"""Tests for the Channel Prediction plugin's per-shank compute, cache and draw loop."""

import dataclasses
import os
import subprocess
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
from qtpy import QtTest, QtWidgets

from ibl_alignment_gui.plugins import channel_prediction as cp
from ibl_alignment_gui.plugins.ephys_atlas import _common, inference, spatial_encoder
from ibl_alignment_gui.plugins.ephys_atlas._common import (
    AUTO_ALIGNMENT_CONFIDENCE_KEY,
    AUTO_ALIGNMENT_KEY,
    PLUGIN_KEY,
    PredictionError,
)
from iblutil.util import Bunch
from tests.app.test_load_worker import _Runner
from tests.plugins._runner import InlineRunner

SHANKS = ('a', 'b')


def _installed(torch: bool, ephysatlas: bool = True):
    """Patch the plugin's checks for torch and ephysatlas to say whether they are installed."""
    real = cp.importlib.util.find_spec
    installed = {'torch': torch, 'ephysatlas': ephysatlas}

    def find_spec(name, *args, **kwargs):
        if name in installed:
            return object() if installed[name] else None
        return real(name, *args, **kwargs)

    return patch.object(cp.importlib.util, 'find_spec', find_spec)


def _fake_controller() -> SimpleNamespace:
    """A controller with two aligned shanks, whose views record what they draw."""
    shank_items = {
        shank: {
            'default': SimpleNamespace(
                name=shank, model=SimpleNamespace(predictions=None), view=MagicMock()
            )
        }
        for shank in SHANKS
    }
    return SimpleNamespace(
        all_shanks=list(SHANKS),
        shank_items=shank_items,
        model=SimpleNamespace(
            configs=['default'],
            get_current_shank=lambda shank, config: SimpleNamespace(align_exists=True),
        ),
    )


DECODER = 'XGBoost region prediction'


class TestReady(unittest.TestCase):
    """ready() allows a prediction only with features, and otherwise says how to load them."""

    def setUp(self):
        self.controller = _fake_controller()
        self.controller.view = None
        self.loader = cp.ChannelPrediction(self.controller)
        self.warning = self.enterContext(patch.object(cp.QtWidgets.QMessageBox, 'warning'))

    def test_with_features(self):
        with patch.object(cp, 'has_features', return_value=True):
            self.assertTrue(self.loader.ready())
        self.warning.assert_not_called()

    def test_without_features(self):
        with patch.object(cp, 'has_features', return_value=False):
            self.assertFalse(self.loader.ready())
        self.warning.assert_called_once()
        self.assertEqual(self.warning.call_args.args[2], cp.NO_FEATURES_MESSAGE)


class TestPlotRegionsModelLookup(unittest.TestCase):
    """plot_predictions gets the option's model once, then computes every shank in one task."""

    def setUp(self):
        self.controller = _fake_controller()
        self.controller.view = None
        self.controller.model.brain_atlas = None
        self.controller.plugins = {PLUGIN_KEY: Bunch()}
        self.controller.run_in_thread = self.runner = InlineRunner()
        self.loader = cp.ChannelPrediction(self.controller)
        # The decoder's computation, per shank, as a stand-in.
        self.compute = MagicMock(return_value=({DECODER: Bunch(region='regions view')}, None))
        self.spec = dataclasses.replace(cp.MODEL_FOR_OPTION[DECODER], compute=self.compute)
        self.enterContext(patch.dict(cp.MODEL_FOR_OPTION, {DECODER: self.spec}))
        self.warning = self.enterContext(patch.object(cp.QtWidgets.QMessageBox, 'warning'))
        self.load = self.enterContext(patch.object(cp, 'load_model', return_value=object()))

    def set_model(self, model=None, source=None) -> None:
        self.controller.plugins[PLUGIN_KEY][self.spec.key] = dict(model=model, source=source)

    def test_cancel_opens_dialog_once(self):
        """With no model loaded, a cancelled load dialog opens once and nothing is computed"""
        with patch.object(cp, 'choose_source', return_value=None) as dialog:
            self.loader.plot_predictions(DECODER)
        dialog.assert_called_once()
        self.assertEqual(self.runner.tasks, [])
        self.compute.assert_not_called()

    def test_model_failure_one_warning(self):
        """A model that cannot be loaded gives one warning and nothing is computed"""
        self.set_model(source=('org/missing', '2026_W39'))
        error = PredictionError('Could not load org/missing@2026_W39:\nnot found')
        self.load.side_effect = error
        self.loader.plot_predictions(DECODER)
        self.load.assert_called_once()
        self.warning.assert_called_once()
        self.assertEqual(self.warning.call_args.args[1:], (DECODER, str(error)))
        self.compute.assert_not_called()

    def test_unexpected_failure_one_warning(self):
        """Any other error getting the model gives one warning and is logged"""
        self.set_model(source=('org/model', '2026_W39'))
        self.load.side_effect = RuntimeError('boom')
        with self.assertLogs('ibl_alignment_gui.plugins.channel_prediction', 'ERROR'):
            self.loader.plot_predictions(DECODER)
        self.warning.assert_called_once()
        self.assertIn('Unexpected error in', self.warning.call_args.args[2])
        self.compute.assert_not_called()

    def test_loaded_model_computes_each_shank(self):
        """With the model loaded, one task computes every shank and they are all drawn"""
        self.set_model(model=object(), source=('org/model', '2026_W39'))
        self.loader.plot_predictions(DECODER)
        self.load.assert_not_called()
        self.assertEqual(len(self.runner.tasks), 1)
        self.assertEqual(self.compute.call_count, len(SHANKS))
        for shank in SHANKS:
            self.controller.shank_items[shank]['default'].view.plot_prediction.assert_called_once()
        self.warning.assert_not_called()

    def test_queued_warning_shown_once(self):
        """A warning queued while computing is shown once when the task ends, then cleared"""
        self.set_model(model=object(), source=('org/model', '2026_W39'))
        state = self.controller.plugins[PLUGIN_KEY]

        def compute(controller, items, model, progress):
            # As features_for_vintage does, on the background thread, for every shank.
            state.setdefault('pending_warnings', []).append('Features of another version')
            return {DECODER: Bunch(region='regions view')}, None

        self.compute.side_effect = compute
        self.loader.plot_predictions(DECODER)
        self.warning.assert_called_once()
        self.assertEqual(self.warning.call_args.args[1:], (DECODER, 'Features of another version'))
        self.assertNotIn('pending_warnings', state)

    def test_version_warning_for_each_model(self):
        """Features of another version warn for every model run on them, not once per session"""
        transformer = 'ProbeTransformer region prediction'
        df = pd.DataFrame({'a': [1.0]})
        df.attrs['features_version'] = '2026_W39'
        shank = SimpleNamespace(
            name='probe00',
            config='default',
            model=SimpleNamespace(loaders={}, raw_data={'features': Bunch(exists=True, df=df)}),
        )

        def compute_for(option):
            def compute(controller, items, model, progress):
                # The models read their features as the backends do; both trained on 2026_W42.
                _common.features_for_vintage(controller, shank, '2026_W42')
                return {option: Bunch(region='regions view')}, None

            return compute

        specs = {
            name: dataclasses.replace(cp.MODEL_FOR_OPTION[name], compute=compute_for(name))
            for name in (DECODER, transformer)
        }
        self.enterContext(patch.dict(cp.MODEL_FOR_OPTION, specs))
        for spec in specs.values():
            self.controller.plugins[PLUGIN_KEY][spec.key] = dict(model=object(), source=None)
        self.loader.plot_predictions(DECODER)
        self.loader.plot_predictions(transformer)
        # One dialog per model, though each ran on two shanks
        self.assertEqual([c.args[1] for c in self.warning.call_args_list], [DECODER, transformer])

    def test_all_cached_no_task(self):
        """When every shank has the option cached, it is drawn without a task"""
        for shank in SHANKS:
            self.controller.shank_items[shank]['default'].model.predictions = Bunch(
                {DECODER: Bunch(region='cached')}
            )
        self.loader.plot_predictions(DECODER)
        self.assertEqual(self.runner.tasks, [])
        self.compute.assert_not_called()
        for shank in SHANKS:
            self.controller.shank_items[shank]['default'].view.plot_prediction.assert_called_once()

    def test_only_uncached_shanks_computed(self):
        """Only the shanks without the option cached are computed; all are drawn"""
        self.set_model(model=object())
        cached = self.controller.shank_items['a']['default']
        cached.model.predictions = Bunch({DECODER: Bunch(region='cached')})
        self.loader.plot_predictions(DECODER)
        computed = [c.args[1] for c in self.compute.call_args_list]
        self.assertEqual(computed, [self.controller.shank_items['b']['default']])
        for shank in SHANKS:
            self.controller.shank_items[shank]['default'].view.plot_prediction.assert_called_once()

    def test_another_task_running(self):
        """While another task runs, the plot is refused with a warning"""
        self.set_model(model=object())
        self.runner.busy = True
        self.loader.plot_predictions(DECODER)
        self.warning.assert_called_once()
        self.compute.assert_not_called()


class TestAllPredictions(unittest.TestCase):
    """plot_all_predictions computes the loaded models in one task and draws every option."""

    TRANSFORMER = 'ProbeTransformer region prediction'

    def setUp(self):
        self.controller = _fake_controller()
        self.controller.view = None
        self.controller.plugins = {PLUGIN_KEY: Bunch()}
        self.controller.run_in_thread = self.runner = InlineRunner()
        self.loader = cp.ChannelPrediction(self.controller)
        # Two models: the decoder is loaded, the transformer is not.
        self.compute = MagicMock(
            return_value=(
                {DECODER: Bunch(region='view'), DECODER + ' (cumulative)': Bunch(region='c')},
                None,
            )
        )
        decoder = dataclasses.replace(cp.MODEL_FOR_OPTION[DECODER], compute=self.compute)
        transformer = cp.MODEL_FOR_OPTION[self.TRANSFORMER]
        self.specs = [decoder, transformer]
        self.enterContext(patch.object(cp, 'available_models', return_value=self.specs))
        self.enterContext(patch.dict(cp.MODEL_FOR_OPTION, {o: decoder for o in decoder.options}))
        self.warning = self.enterContext(patch.object(cp.QtWidgets.QMessageBox, 'warning'))
        self.controller.plugins[PLUGIN_KEY][decoder.key] = dict(model=object(), source=None)
        self.options = [o for spec in self.specs for o in spec.options]
        for shank in SHANKS:
            view = self.controller.shank_items[shank]['default'].view
            view.fig_preds = {option: MagicMock(name=option) for option in self.options}

    def view(self, shank):
        return self.controller.shank_items[shank]['default'].view

    def test_loaded_model_computed_and_drawn(self):
        """The loaded model runs once per shank in one task, and both its options are drawn"""
        self.loader.plot_all_predictions()
        self.assertEqual(len(self.runner.tasks), 1)
        self.assertEqual(self.compute.call_count, len(SHANKS))
        for shank in SHANKS:
            view = self.view(shank)
            drawn = [c.args[2] for c in view.plot_prediction.call_args_list]
            self.assertEqual(drawn, [DECODER, DECODER + ' (cumulative)'])
            figs = [c.args[0] for c in view.plot_prediction.call_args_list]
            self.assertEqual(figs, [view.fig_preds[o] for o in drawn])

    def test_unloaded_model_not_loaded(self):
        """The options of a model that is not loaded say so, and it is not loaded"""
        with patch.object(cp, 'load_model') as load:
            self.loader.plot_all_predictions()
        load.assert_not_called()
        for shank in SHANKS:
            cleared = {c.args[1]: c.args[2] for c in self.view(shank).clear_prediction.mock_calls}
            self.assertEqual(
                cleared,
                {
                    self.TRANSFORMER: cp.NOT_LOADED_MESSAGE,
                    self.TRANSFORMER + ' (cumulative)': cp.NOT_LOADED_MESSAGE,
                },
            )

    def test_cached_not_computed_again(self):
        """Once computed, showing the view again needs no task"""
        self.loader.plot_all_predictions()
        self.loader.plot_all_predictions()
        self.assertEqual(len(self.runner.tasks), 1)
        self.assertEqual(self.compute.call_count, len(SHANKS))

    def test_progress_names_model(self):
        """Each progress message names the model it is for"""
        self.loader.plot_all_predictions()
        self.assertTrue(self.runner.progress)
        for message in self.runner.progress:
            self.assertTrue(message.startswith(DECODER + ', '), message)

    def test_errors_one_warning(self):
        """The models' errors on every shank are reported in one warning"""
        self.compute.side_effect = PredictionError('no features')
        self.loader.plot_all_predictions()
        self.warning.assert_called_once()
        self.assertEqual(self.warning.call_args.args[1], cp.ALL_TITLE)

    def test_model_loaded_from_menu_redraws(self):
        """In the all predictions view, a model loaded from the menu redraws that view"""
        self.controller.show_predictions = True
        self.controller.view = MagicMock()
        cp._show_first_option(self.controller, self.specs[0])
        self.controller.view.trigger_menu_option.assert_not_called()
        self.assertEqual(self.compute.call_count, len(SHANKS))


class TestPredictionTask(unittest.TestCase):
    """The background task: load the model if needed, then compute each shank."""

    def setUp(self):
        self.controller = _fake_controller()
        self.shanks = [self.controller.shank_items[s]['default'] for s in SHANKS]
        self.computed = []

        def compute(controller, items, model, progress):
            progress(0.5, 'halfway')
            self.computed.append(items.name)
            return {DECODER: Bunch(shank=items.name)}, f'extra {items.name}'

        self.spec = dataclasses.replace(cp.MODEL_FOR_OPTION[DECODER], compute=compute)
        self.load = self.enterContext(patch.object(cp, 'load_model', return_value='model'))
        self.reports = []
        self.cancel_at = None

    def progress_callback(self, message, current, total):
        self.reports.append((message, current, total))
        if self.cancel_at is not None and len(self.reports) >= self.cancel_at:
            raise cp.TaskCancelledError()

    def run_task(self, model=None, source=('repo', 'tag')):
        return cp._prediction_task(
            self.controller, self.spec, model, source, self.shanks, self.progress_callback
        )

    def test_loads_then_computes_each_shank(self):
        """The model is loaded, then each shank computed with it, in order"""
        outcome = self.run_task()
        self.assertEqual(self.load.call_args.args[:2], (self.spec, ('repo', 'tag')))
        self.assertEqual((outcome.model, outcome.loaded_from), ('model', ('repo', 'tag')))
        self.assertEqual(self.computed, list(SHANKS))
        self.assertEqual([r[2] for r in outcome.results], ['extra a', 'extra b'])
        self.assertFalse(outcome.cancelled)

    def test_loaded_model_not_loaded_again(self):
        """With the model given, nothing is loaded"""
        outcome = self.run_task(model='loaded', source=None)
        self.load.assert_not_called()
        self.assertIsNone(outcome.loaded_from)
        self.assertEqual(self.computed, list(SHANKS))

    def test_progress_names_shank(self):
        """Each shank's progress is prefixed with its name and position"""
        self.run_task(model='loaded')
        self.assertIn(('a (1/2): halfway', 500, 1000), self.reports)
        self.assertIn(('b (2/2): starting…', 0, 0), self.reports)

    def test_error_on_one_shank(self):
        """A shank's error is kept and the next shank still computed"""
        compute = self.spec.compute

        def failing(controller, items, model, progress):
            if items.name == 'a':
                raise PredictionError('no features on a')
            return compute(controller, items, model, progress)

        self.spec = dataclasses.replace(self.spec, compute=failing)
        outcome = self.run_task(model='loaded')
        self.assertEqual(outcome.errors, ['no features on a'])
        self.assertEqual([r[0].name for r in outcome.results], ['b'])

    def test_unexpected_error_on_one_shank(self):
        """An unexpected error on a shank is logged, reported, and the next shank computed"""
        compute = self.spec.compute

        def failing(controller, items, model, progress):
            if items.name == 'a':
                raise RuntimeError('boom')
            return compute(controller, items, model, progress)

        self.spec = dataclasses.replace(self.spec, compute=failing)
        with self.assertLogs('ibl_alignment_gui.plugins.channel_prediction', 'ERROR'):
            outcome = self.run_task(model='loaded')
        self.assertEqual(len(outcome.errors), 1)
        self.assertIn('Unexpected error in', outcome.errors[0])
        self.assertEqual([r[0].name for r in outcome.results], ['b'])

    def test_cancel_between_shanks(self):
        """After Cancel, the finished shanks are kept and the rest skipped"""
        # Reports: a starting, a halfway, b starting (Cancel).
        self.cancel_at = 3
        outcome = self.run_task(model='loaded')
        self.assertTrue(outcome.cancelled)
        self.assertEqual([r[0].name for r in outcome.results], ['a'])
        self.assertEqual(self.computed, ['a'])

    def test_model_kept_after_cancel(self):
        """A model loaded before Cancel is still handed back, to be stored"""
        self.cancel_at = 1  # the first shank's first report
        outcome = self.run_task()
        self.assertTrue(outcome.cancelled)
        self.assertEqual(outcome.loaded_from, ('repo', 'tag'))
        self.assertEqual(outcome.results, [])


class TestPredictionInThread(unittest.TestCase):
    """plot_predictions with the controller's real background runner and a real thread."""

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_drawn_when_task_ends(self):
        """The views are computed off the main thread and drawn once the task has ended"""
        controller = _fake_controller()
        controller.model.brain_atlas = None
        controller.view = None
        runner = _Runner()
        controller.run_in_thread = runner.run_in_thread
        threads = []

        def compute(c, items, model, progress):
            threads.append(threading.get_ident())
            return {DECODER: Bunch(region='view')}, None

        spec = dataclasses.replace(cp.MODEL_FOR_OPTION[DECODER], compute=compute)
        controller.plugins = {PLUGIN_KEY: Bunch({spec.key: dict(model=object(), source=None)})}
        with patch.dict(cp.MODEL_FOR_OPTION, {DECODER: spec}):
            cp.ChannelPrediction(controller).plot_predictions(DECODER)
            for _ in range(500):
                if not runner.task_running:
                    break
                QtTest.QTest.qWait(10)
        self.assertFalse(runner.task_running)
        self.assertEqual(len(threads), len(SHANKS))
        self.assertNotIn(threading.get_ident(), threads)
        for shank in SHANKS:
            controller.shank_items[shank]['default'].view.plot_prediction.assert_called_once()


class TestModelOptions(unittest.TestCase):
    """The model options offered in Prediction Plots (which models: TestModelRegistry)."""

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def offered(self, torch: bool = True, ephysatlas: bool = True) -> list:
        """The prediction options setup gives the controller."""
        view = QtWidgets.QWidget()
        self.addCleanup(view.deleteLater)
        controller = SimpleNamespace(
            view=view, plugins={}, plugin_options=MagicMock(), prediction_keys=[]
        )
        with _installed(torch=torch, ephysatlas=ephysatlas):
            cp.setup(controller)
        return controller.prediction_keys

    def test_every_option_of_each_model(self):
        """With torch, both options of each decoder and of the auto-alignment, in order"""
        self.assertEqual(
            self.offered(),
            [
                'XGBoost region prediction',
                'XGBoost region prediction (cumulative)',
                'ProbeTransformer region prediction',
                'ProbeTransformer region prediction (cumulative)',
                AUTO_ALIGNMENT_KEY,
                AUTO_ALIGNMENT_CONFIDENCE_KEY,
            ],
        )

    def test_without_torch(self):
        """Without torch, only the XGBoost decoder's options"""
        self.assertEqual(
            self.offered(torch=False),
            ['XGBoost region prediction', 'XGBoost region prediction (cumulative)'],
        )

    def test_without_ephysatlas(self):
        """Without ephysatlas, none"""
        self.assertEqual(self.offered(ephysatlas=False), [])


class TestModelRegistry(unittest.TestCase):
    """The models offered, the Plugins menu built from them, and clearing loaded models."""

    def test_model_for_option(self):
        """Decoder options give their decoder, auto-alignment options the alignment model"""
        for name, spec in inference.DECODER_SPECS.items():
            for option in (name, name + inference.CUMULATIVE_SUFFIX):
                self.assertIs(cp.MODEL_FOR_OPTION[option], spec)
        for option in (AUTO_ALIGNMENT_KEY, AUTO_ALIGNMENT_CONFIDENCE_KEY):
            self.assertIs(cp.MODEL_FOR_OPTION[option], spatial_encoder.ENCODER_SPEC)
        for option in ('Beryl', 'Cosmos'):
            self.assertNotIn(option, cp.MODEL_FOR_OPTION)

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_available_models(self):
        """Without torch only the XGBoost decoder is offered; with torch every model"""
        with _installed(torch=False):
            self.assertEqual([spec.key for spec in cp.available_models()], [DECODER])
        with _installed(torch=True):
            self.assertEqual(cp.available_models(), list(cp.MODELS))

    def menu(self, torch: bool) -> list[str]:
        """The Plugins -> Channel Prediction menu items setup adds."""
        # A real widget as the view: it owns the menu's actions, as the main window does.
        view = QtWidgets.QWidget()
        self.addCleanup(view.deleteLater)
        controller = SimpleNamespace(
            view=view,
            plugins={},
            plugin_options=MagicMock(),
            model=SimpleNamespace(brain_atlas=None),
        )
        with _installed(torch=torch):
            cp.setup(controller)
        menu = controller.plugin_options.addMenu.call_args.args[0]
        return [action.text() for action in menu.actions()]

    def test_menu_with_torch(self):
        """With torch, one load item per model and the results item"""
        self.assertEqual(
            self.menu(torch=True),
            [
                'Load XGBoost region prediction model (Hugging Face)…',
                'Load ProbeTransformer region prediction model (Hugging Face)…',
                'Load alignment model (Hugging Face)…',
                'Show auto-alignment results',
            ],
        )

    def test_menu_without_torch(self):
        """Without torch, only the XGBoost load item"""
        self.assertEqual(
            self.menu(torch=False), ['Load XGBoost region prediction model (Hugging Face)…']
        )

    def test_menu_needs_features(self):
        """The menu is greyed out until a session with features is loaded"""
        view = QtWidgets.QWidget()
        self.addCleanup(view.deleteLater)
        controller = SimpleNamespace(
            view=view, plugins={}, plugin_options=MagicMock(), model=SimpleNamespace()
        )
        with _installed(torch=False):
            cp.setup(controller)
        menu = controller.plugin_options.addMenu.call_args.args[0]
        self.assertFalse(menu.isEnabled())
        self.assertEqual(menu.menuAction().toolTip(), cp.NO_FEATURES_MESSAGE)
        for features in (True, False):
            with self.subTest(features=features):
                with (
                    patch.object(cp, 'has_features', return_value=features),
                    patch.object(cp, 'close_alignment_windows'),
                ):
                    controller.plugins[PLUGIN_KEY]['data_button_pressed']()
                self.assertEqual(menu.isEnabled(), features)
                # Without its hint, Qt reports the menu's own name as its tooltip
                hint = menu.menuAction().toolTip() == cp.NO_FEATURES_MESSAGE
                self.assertEqual(hint, not features)


class TestLazyImports(unittest.TestCase):
    def test_no_heavy_imports(self):
        """The plugin and both backends import without importing torch or the alignment model

        (The GUI's feature loader imports the light ``ephysatlas.data`` on its own.)
        """
        code = (
            'import sys\n'
            'import ibl_alignment_gui.plugins.channel_prediction\n'
            'from ibl_alignment_gui.plugins.ephys_atlas import inference, spatial_encoder\n'
            "print(sorted(m for m in ('torch', 'ephysatlas.alignment') if m in sys.modules))\n"
        )
        run = subprocess.run(
            [sys.executable, '-c', code], capture_output=True, text=True, check=True
        )
        self.assertEqual(run.stdout.strip(), '[]')


if __name__ == '__main__':
    unittest.main()
