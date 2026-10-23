"""Tests for the Channel Prediction plugin's per-shank compute, cache and draw loop."""

import dataclasses
import os
import subprocess
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from qtpy import QtTest, QtWidgets

from ibl_alignment_gui.plugins import channel_prediction as cp
from ibl_alignment_gui.plugins.ephys_atlas import inference, spatial_encoder
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


class TestPlotRegionPanels(unittest.TestCase):
    def setUp(self):
        self.controller = _fake_controller()
        self.items = [self.controller.shank_items[shank]['default'] for shank in SHANKS]
        self.regions = Bunch(region='regions view')
        self.other = Bunch(region='other view')

    def drawn(self, items) -> list:
        """The views drawn on a shank, in order."""
        return [c.args[1] for c in items.view.plot_histology.call_args_list]

    def test_single_view_cached(self):
        """A single view is cached under its option and the model runs once per shank"""
        func = MagicMock(return_value={'Model': self.regions})
        cp._plot_region_panels(self.controller, 'Model', func)
        cp._plot_region_panels(self.controller, 'Model', func)
        self.assertEqual(func.call_count, len(SHANKS))
        for items in self.items:
            self.assertIs(items.model.predictions['Model'], self.regions)
            self.assertEqual(self.drawn(items), [self.regions, self.regions])

    def test_all_views_cached(self):
        """All the views a model returns are cached, so its other option does not run it again"""
        func = MagicMock(return_value={'Model': self.regions, 'Model (other)': self.other})
        cp._plot_region_panels(self.controller, 'Model', func)
        cp._plot_region_panels(self.controller, 'Model (other)', func)
        self.assertEqual(func.call_count, len(SHANKS))
        for items in self.items:
            self.assertEqual(set(items.model.predictions), {'Model', 'Model (other)'})
            self.assertEqual(self.drawn(items), [self.regions, self.other])

    def test_option_missing_from_views(self):
        """When the views lack the requested option, the others are cached and nothing is drawn"""
        func = MagicMock(return_value={'Model (other)': self.other})
        cp._plot_region_panels(self.controller, 'Model', func)
        for items in self.items:
            self.assertEqual(set(items.model.predictions), {'Model (other)'})
            self.assertEqual(self.drawn(items), [])

    def test_cancelled_not_cached(self):
        """A cancelled computation caches and draws nothing, and the next call computes again"""
        func = MagicMock(return_value=None)
        cp._plot_region_panels(self.controller, 'Model', func)
        for items in self.items:
            self.assertEqual(dict(items.model.predictions), {})
            self.assertEqual(self.drawn(items), [])
        cp._plot_region_panels(self.controller, 'Model', func)
        self.assertEqual(func.call_count, 2 * len(SHANKS))

    def test_error_not_cached(self):
        """A PredictionError is returned per shank, nothing is cached, and the next call retries"""
        func = MagicMock(side_effect=PredictionError('no features'))
        errors = cp._plot_region_panels(self.controller, 'Model', func)
        self.assertEqual(errors, ['no features'] * len(SHANKS))
        for items in self.items:
            self.assertNotIn('Model', items.model.predictions)
        func.side_effect, func.return_value = None, {'Model': self.regions}
        cp._plot_region_panels(self.controller, 'Model', func)
        self.assertEqual(func.call_count, 2 * len(SHANKS))

    def test_unexpected_error_reported(self):
        """Any other error is returned per shank with its type and message, logged, not cached"""
        func = MagicMock(side_effect=ValueError('boom'))
        with self.assertLogs('ibl_alignment_gui.plugins.channel_prediction', 'ERROR') as logs:
            errors = cp._plot_region_panels(self.controller, 'Model', func)
        self.assertEqual(len(errors), len(SHANKS))
        for error in errors:
            self.assertIn('Unexpected error in Model: ValueError: boom', error)
            self.assertIn('See the log for details.', error)
        self.assertEqual(len(logs.records), len(SHANKS))
        self.assertIn('Traceback', logs.output[0])
        for items in self.items:
            self.assertNotIn('Model', items.model.predictions)

    def test_unexpected_error_shown_once(self):
        """Through plot_regions, the same error on every shank gives one warning"""
        self.controller.view = None
        self.controller.model.brain_atlas = None
        loader = cp.ChannelPrediction(self.controller)
        loader.func_map['Model'] = MagicMock(side_effect=ValueError('boom'))
        with (
            patch.object(cp.QtWidgets.QMessageBox, 'warning') as warning,
            self.assertLogs('ibl_alignment_gui.plugins.channel_prediction', 'ERROR'),
        ):
            loader.plot_regions('Model')
        warning.assert_called_once()
        self.assertEqual(warning.call_args.args[1], 'Model')
        self.assertEqual(warning.call_args.args[2].count('Unexpected error in Model'), 1)


class TestComputeMappingPredictions(unittest.TestCase):
    def test_keyed_by_mapping(self):
        """Beryl and Cosmos return their regions keyed by their region-plot option"""
        regions = Bunch(
            id=np.array([1, 1]), acronym=np.array(['A', 'A']), rgb=np.array([[1, 2, 3]] * 2)
        )
        controller = SimpleNamespace(
            model=SimpleNamespace(
                brain_atlas=SimpleNamespace(
                    get_labels=lambda xyz, mapping: np.array([1, 1]),
                    regions=SimpleNamespace(get=lambda ids: regions),
                )
            )
        )
        items = SimpleNamespace(
            model=SimpleNamespace(
                align_handle=SimpleNamespace(
                    xyz_samples=None,
                    ephysalign=SimpleNamespace(sampling_trk=np.array([0.0, 1e-5])),
                )
            )
        )
        for mapping in ('Beryl', 'Cosmos'):
            with self.subTest(mapping=mapping):
                views = cp.compute_mapping_predictions(controller, items, mapping)
                self.assertEqual(list(views), [mapping])
                self.assertEqual(views[mapping].axis_label[:, 1].tolist(), ['A'])


DECODER = 'XGBoost region prediction'


class TestPlotRegionsModelLookup(unittest.TestCase):
    """plot_regions gets the option's model once, then computes every shank in one task."""

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
            self.loader.plot_regions(DECODER)
        dialog.assert_called_once()
        self.assertEqual(self.runner.tasks, [])
        self.compute.assert_not_called()

    def test_model_failure_one_warning(self):
        """A model that cannot be loaded gives one warning and nothing is computed"""
        self.set_model(source=('org/missing', '2026_W39'))
        error = PredictionError('Could not load org/missing@2026_W39:\nnot found')
        self.load.side_effect = error
        self.loader.plot_regions(DECODER)
        self.load.assert_called_once()
        self.warning.assert_called_once()
        self.assertEqual(self.warning.call_args.args[1:], (DECODER, str(error)))
        self.compute.assert_not_called()

    def test_unexpected_failure_one_warning(self):
        """Any other error getting the model gives one warning and is logged"""
        self.set_model(source=('org/model', '2026_W39'))
        self.load.side_effect = RuntimeError('boom')
        with self.assertLogs('ibl_alignment_gui.plugins.channel_prediction', 'ERROR'):
            self.loader.plot_regions(DECODER)
        self.warning.assert_called_once()
        self.assertIn('Unexpected error in', self.warning.call_args.args[2])
        self.compute.assert_not_called()

    def test_loaded_model_computes_each_shank(self):
        """With the model loaded, one task computes every shank and they are all drawn"""
        self.set_model(model=object(), source=('org/model', '2026_W39'))
        self.loader.plot_regions(DECODER)
        self.load.assert_not_called()
        self.assertEqual(len(self.runner.tasks), 1)
        self.assertEqual(self.compute.call_count, len(SHANKS))
        for shank in SHANKS:
            self.controller.shank_items[shank]['default'].view.plot_histology.assert_called_once()
        self.warning.assert_not_called()

    def test_mapping_needs_no_model(self):
        """Beryl is computed at once, without a model or a background task"""
        self.loader.func_map['Beryl'] = MagicMock(return_value={'Beryl': Bunch()})
        with patch.object(cp, 'choose_source') as dialog:
            self.loader.plot_regions('Beryl')
        dialog.assert_not_called()
        self.assertEqual(self.runner.tasks, [])
        self.assertEqual(self.loader.func_map['Beryl'].call_count, len(SHANKS))

    def test_all_cached_no_task(self):
        """When every shank has the option cached, it is drawn without a task"""
        for shank in SHANKS:
            self.controller.shank_items[shank]['default'].model.predictions = Bunch(
                {DECODER: Bunch(region='cached')}
            )
        self.loader.plot_regions(DECODER)
        self.assertEqual(self.runner.tasks, [])
        self.compute.assert_not_called()
        for shank in SHANKS:
            self.controller.shank_items[shank]['default'].view.plot_histology.assert_called_once()

    def test_only_uncached_shanks_computed(self):
        """Only the shanks without the option cached are computed; all are drawn"""
        self.set_model(model=object())
        cached = self.controller.shank_items['a']['default']
        cached.model.predictions = Bunch({DECODER: Bunch(region='cached')})
        self.loader.plot_regions(DECODER)
        computed = [c.args[1] for c in self.compute.call_args_list]
        self.assertEqual(computed, [self.controller.shank_items['b']['default']])
        for shank in SHANKS:
            self.controller.shank_items[shank]['default'].view.plot_histology.assert_called_once()

    def test_another_task_running(self):
        """While another task runs, the plot is refused with a warning"""
        self.set_model(model=object())
        self.runner.busy = True
        self.loader.plot_regions(DECODER)
        self.warning.assert_called_once()
        self.compute.assert_not_called()


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
    """plot_regions with the controller's real background runner and a real thread."""

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
            cp.ChannelPrediction(controller).plot_regions(DECODER)
            for _ in range(500):
                if not runner.task_running:
                    break
                QtTest.QTest.qWait(10)
        self.assertFalse(runner.task_running)
        self.assertEqual(len(threads), len(SHANKS))
        self.assertNotIn(threading.get_ident(), threads)
        for shank in SHANKS:
            controller.shank_items[shank]['default'].view.plot_histology.assert_called_once()


class TestModelOptions(unittest.TestCase):
    """Which model options the region dropdown offers: the same online and offline."""

    ALL = [
        'XGBoost region prediction',
        'XGBoost region prediction (cumulative)',
        'ProbeTransformer region prediction',
        'ProbeTransformer region prediction (cumulative)',
        AUTO_ALIGNMENT_KEY,
        AUTO_ALIGNMENT_CONFIDENCE_KEY,
    ]

    def offered(self, offline=True, torch=True, present=()) -> list:
        """The options _refresh_model_options adds, with no model loaded."""
        controller = SimpleNamespace(
            offline=offline,
            plugins={PLUGIN_KEY: Bunch()},
            view=MagicMock(),
            plot_region_ref_panels=MagicMock(),
        )
        controller.view.has_menu_option.side_effect = lambda tab, key: key in present
        with _installed(torch=torch):
            cp._refresh_model_options(controller)
        populate = controller.view.populate_menu_tab
        return populate.call_args.args[2] if populate.called else []

    def test_options_offered_offline_before_loading(self):
        """Offline, with torch and no model loaded, every model's options are offered"""
        self.assertCountEqual(self.offered(offline=True), self.ALL)

    def test_same_online_and_offline(self):
        """Online and offline offer the same options"""
        for torch in (True, False):
            with self.subTest(torch=torch):
                self.assertCountEqual(
                    self.offered(offline=True, torch=torch),
                    self.offered(offline=False, torch=torch),
                )

    def test_torch_models_need_torch(self):
        """Without torch, only the XGBoost options are offered"""
        self.assertCountEqual(self.offered(torch=False), self.ALL[:2])

    def test_options_not_duplicated(self):
        """Options already in the dropdown are not added again"""
        self.assertCountEqual(self.offered(present=self.ALL[:2]), self.ALL[2:])


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
        """With torch, one load item per model, the features file and the results item"""
        self.assertEqual(
            self.menu(torch=True),
            [
                'Load XGBoost region prediction model (Hugging Face)…',
                'Load ProbeTransformer region prediction model (Hugging Face)…',
                'Load alignment model (Hugging Face)…',
                'Load features file…',
                'Show auto-alignment results',
            ],
        )

    def test_menu_without_torch(self):
        """Without torch, only the XGBoost load item and the features file"""
        self.assertEqual(
            self.menu(torch=False),
            ['Load XGBoost region prediction model (Hugging Face)…', 'Load features file…'],
        )

    def test_invalidate_loaded_models(self):
        """Only the loaded models have their predictions cleared"""
        loaded = (spatial_encoder.ENCODER_SPEC.key,)
        controller = SimpleNamespace(
            plugins={
                PLUGIN_KEY: Bunch(
                    {spec.key: dict(model=object()) for spec in cp.MODELS if spec.key in loaded}
                )
            }
        )
        specs = [dataclasses.replace(spec, invalidate=MagicMock()) for spec in cp.MODELS]
        with patch.object(cp, 'MODELS', tuple(specs)):
            cp._invalidate_loaded_models(controller)
        for spec in specs:
            if spec.key in loaded:
                spec.invalidate.assert_called_once_with(controller)
            else:
                spec.invalidate.assert_not_called()


class TestSetLocalFeatures(unittest.TestCase):
    def test_loaded_shanks_load_file(self):
        """A chosen features file is loaded by the shanks whose data is loaded, not the others"""
        loaded = MagicMock(data_loaded=True)
        not_loaded = MagicMock(data_loaded=False)
        controller = SimpleNamespace(
            view=None,
            plugins={PLUGIN_KEY: Bunch()},
            model=SimpleNamespace(shanks={'a': {'default': loaded}, 'b': {'default': not_loaded}}),
        )
        dialog = cp.QtWidgets.QFileDialog
        with (
            patch.object(dialog, 'getOpenFileName', return_value=('/data/f.pqt', '')),
            patch.object(cp, 'FeatureLoaderLocal') as loader,
            patch.object(cp, '_invalidate_loaded_models') as invalidate,
        ):
            loader.return_value.load_features.return_value = Bunch(exists=True)
            cp._set_local_features(controller)
        loaded.load_ephys_features.assert_called_once_with(loader.return_value)
        not_loaded.load_ephys_features.assert_not_called()
        invalidate.assert_called_once_with(controller)
        self.assertEqual(controller.plugins[PLUGIN_KEY]['features_path'], cp.Path('/data/f.pqt'))


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
