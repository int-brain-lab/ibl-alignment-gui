"""Tests for the Channel Prediction plugin's per-shank compute, cache and draw loop."""

import dataclasses
import os
import subprocess
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from qtpy import QtTest, QtWidgets

from ibl_alignment_gui.plugins import channel_prediction as cp
from ibl_alignment_gui.plugins.ephys_atlas import _common, inference, spatial_encoder
from ibl_alignment_gui.plugins.ephys_atlas._common import (
    ALL_SHANKS,
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

    def test_queued_warning_shown_once(self):
        """A warning queued while computing is shown once when the task ends, then cleared"""
        self.set_model(model=object(), source=('org/model', '2026_W39'))
        state = self.controller.plugins[PLUGIN_KEY]

        def compute(controller, items, model, progress):
            # As features_for_vintage does, on the background thread, for every shank.
            state.setdefault('pending_warnings', []).append('Features of another version')
            return {DECODER: Bunch(region='regions view')}, None

        self.compute.side_effect = compute
        self.loader.plot_regions(DECODER)
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
        self.loader.plot_regions(DECODER)
        self.loader.plot_regions(transformer)
        # One dialog per model, though each ran on two shanks
        self.assertEqual([c.args[1] for c in self.warning.call_args_list], [DECODER, transformer])

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
    """Adding the model options to the region dropdown (which models: TestModelRegistry)."""

    ALL = [
        'XGBoost region prediction',
        'XGBoost region prediction (cumulative)',
        'ProbeTransformer region prediction',
        'ProbeTransformer region prediction (cumulative)',
        AUTO_ALIGNMENT_KEY,
        AUTO_ALIGNMENT_CONFIDENCE_KEY,
    ]

    def offered(self, present=()) -> list:
        """The options _refresh_model_options adds, with torch and no model loaded."""
        controller = SimpleNamespace(
            plugins={PLUGIN_KEY: Bunch()},
            view=MagicMock(),
            plot_region_ref_panels=MagicMock(),
        )
        controller.view.has_menu_option.side_effect = lambda tab, key: key in present
        with _installed(torch=True):
            cp._refresh_model_options(controller)
        populate = controller.view.populate_menu_tab
        return populate.call_args.args[2] if populate.called else []

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


def _handler(loaded: bool = True, yaml_file: str | None = None) -> MagicMock:
    """A shank handler, with its data loaded or not, and a features file from the session YAML."""
    handler = MagicMock(data_loaded=loaded, shank_idx=0)
    features = cp.FeatureLoaderLocal(yaml_file) if yaml_file else None
    handler.loaders = {'features': features, 'geom': MagicMock()}
    return handler


class TestSetLocalFeatures(unittest.TestCase):
    """ "Load features file…": choosing, checking and applying features files per shank."""

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.invalidate = self.enterContext(patch.object(cp, '_invalidate_loaded_models'))
        self.warning = self.enterContext(patch.object(cp.QtWidgets.QMessageBox, 'warning'))

        # Every file has features, except one named "bad"; no file is read.
        def load_features(loader, shank_sites=None):
            return Bunch(exists=loader.features_path.name != 'bad')

        self.enterContext(
            patch.object(
                cp.FeatureLoaderLocal, 'load_features', autospec=True, side_effect=load_features
            )
        )

    def controller(self, shanks: dict) -> SimpleNamespace:
        return SimpleNamespace(
            view=None, plugins={PLUGIN_KEY: Bunch()}, model=SimpleNamespace(shanks=shanks)
        )

    def choose(self, chosen: dict | None):
        """Patch the per-shank dialog to return ``chosen`` on OK, or to be cancelled (None)."""
        dialog = MagicMock()
        dialog.exec.return_value = (
            QtWidgets.QDialog.Rejected if chosen is None else QtWidgets.QDialog.Accepted
        )
        dialog.chosen.return_value = chosen or {}
        return patch.object(cp, 'FeaturesFilesDialog', return_value=dialog)

    def loaded_file(self, handler) -> Path | None:
        calls = handler.load_ephys_features.call_args_list
        return calls[-1].args[0].features_path if calls else None

    def paths(self, controller) -> dict:
        return controller.plugins[PLUGIN_KEY]['features_paths']

    def test_single_shank_plain_dialog(self):
        """One shank and one configuration: a plain file dialog, used for that shank"""
        handler = _handler()
        controller = self.controller({'a': {'default': handler}})
        with (
            patch.object(cp.QtWidgets.QFileDialog, 'getOpenFileName', return_value=('/f.pqt', '')),
            patch.object(cp, 'FeaturesFilesDialog') as dialog,
        ):
            cp._set_local_features(controller)
        dialog.assert_not_called()
        self.assertEqual(self.paths(controller), {ALL_SHANKS: Path('/f.pqt')})
        self.assertEqual(self.loaded_file(handler), Path('/f.pqt'))
        self.invalidate.assert_called_once_with(controller)

    def test_one_file_for_all_shanks(self):
        """One file for all shanks: loaded by the shanks whose data is loaded, not the others"""
        loaded, not_loaded = _handler(), _handler(loaded=False)
        controller = self.controller({'a': {'default': loaded}, 'b': {'default': not_loaded}})
        with self.choose({ALL_SHANKS: Path('/f.pqt')}):
            cp._set_local_features(controller)
        self.assertEqual(self.loaded_file(loaded), Path('/f.pqt'))
        not_loaded.load_ephys_features.assert_not_called()
        self.assertEqual(self.paths(controller), {ALL_SHANKS: Path('/f.pqt')})
        self.invalidate.assert_called_once_with(controller)

    def test_file_per_shank(self):
        """Each shank loads its own file"""
        a, b = _handler(), _handler()
        controller = self.controller({'a': {'default': a}, 'b': {'default': b}})
        chosen = {('a', 'default'): Path('/a.pqt'), ('b', 'default'): Path('/b.pqt')}
        with self.choose(chosen):
            cp._set_local_features(controller)
        self.assertEqual(
            (self.loaded_file(a), self.loaded_file(b)), (Path('/a.pqt'), Path('/b.pqt'))
        )
        self.assertEqual(self.paths(controller), chosen)

    def test_file_per_configuration(self):
        """A file for one configuration leaves the shank's other configuration as it is"""
        dense, sparse = _handler(), _handler()
        controller = self.controller({'a': {'dense': dense, 'sparse': sparse}})
        with self.choose({('a', 'dense'): Path('/dense.pqt')}):
            cp._set_local_features(controller)
        self.assertEqual(self.loaded_file(dense), Path('/dense.pqt'))
        sparse.load_ephys_features.assert_not_called()

    def test_file_kept_for_other_shanks(self):
        """Choosing a file for one shank keeps the files chosen before for the others"""
        a, b = _handler(), _handler()
        controller = self.controller({'a': {'default': a}, 'b': {'default': b}})
        controller.plugins[PLUGIN_KEY]['features_paths'] = {('b', 'default'): Path('/b.pqt')}
        with self.choose({('a', 'default'): Path('/a.pqt')}):
            cp._set_local_features(controller)
        b.load_ephys_features.assert_not_called()
        self.assertEqual(
            self.paths(controller),
            {('a', 'default'): Path('/a.pqt'), ('b', 'default'): Path('/b.pqt')},
        )

    def test_invalid_file_applies_nothing(self):
        """A file without features for its shank: a warning naming it, and nothing changes"""
        a, b = _handler(), _handler()
        controller = self.controller({'a': {'default': a}, 'b': {'default': b}})
        with self.choose({('a', 'default'): Path('/a.pqt'), ('b', 'default'): Path('/bad')}):
            cp._set_local_features(controller)
        self.warning.assert_called_once()
        self.assertIn('b (default)', self.warning.call_args.args[2])
        a.load_ephys_features.assert_not_called()
        self.assertNotIn('features_paths', controller.plugins[PLUGIN_KEY])
        self.invalidate.assert_not_called()

    def test_unreadable_file_reported(self):
        """A file that cannot be read is reported like one without features, and nothing changes"""
        a, b = _handler(), _handler()
        controller = self.controller({'a': {'default': a}, 'b': {'default': b}})
        with (
            self.choose({('a', 'default'): Path('/notes.txt')}),
            patch.object(
                cp.FeatureLoaderLocal, 'load_features', side_effect=OSError('not parquet')
            ),
        ):
            cp._set_local_features(controller)
        self.warning.assert_called_once()
        self.assertIn('a (default): /notes.txt could not be read', self.warning.call_args.args[2])
        self.assertNotIn('features_paths', controller.plugins[PLUGIN_KEY])

    def test_cancel_changes_nothing(self):
        """Cancelling the dialog changes nothing"""
        a, b = _handler(), _handler()
        controller = self.controller({'a': {'default': a}, 'b': {'default': b}})
        with self.choose(None):
            cp._set_local_features(controller)
        a.load_ephys_features.assert_not_called()
        self.assertNotIn('features_paths', controller.plugins[PLUGIN_KEY])
        self.invalidate.assert_not_called()

    def test_current_file(self):
        """A row opens with the file chosen before, else the file for all, else the YAML's file"""
        handler = _handler(yaml_file='/yaml.pqt')
        controller = self.controller({})
        key = ('a', 'default')
        self.assertEqual(cp._current_features_file(controller, key, handler), Path('/yaml.pqt'))
        controller.plugins[PLUGIN_KEY]['features_paths'] = {ALL_SHANKS: Path('/all.pqt')}
        self.assertEqual(cp._current_features_file(controller, key, handler), Path('/all.pqt'))
        controller.plugins[PLUGIN_KEY]['features_paths'][key] = Path('/own.pqt')
        self.assertEqual(cp._current_features_file(controller, key, handler), Path('/own.pqt'))


class TestFeaturesFilesDialog(unittest.TestCase):
    """The per-shank dialog: which rows it returns."""

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        rows = [
            (('a', 'default'), Path('/a.pqt')),
            (('b', 'default'), Path('/b.pqt')),
            (('c', 'default'), None),
        ]
        self.dialog = cp.FeaturesFilesDialog(None, rows)
        self.addCleanup(self.dialog.deleteLater)

    def field(self, key) -> QtWidgets.QLineEdit:
        return self.dialog._fields[key]

    def test_opens_with_current_files(self):
        """Each row opens with the shank's current file"""
        self.assertEqual(self.field(('a', 'default')).text(), '/a.pqt')
        self.assertEqual(self.field(('c', 'default')).text(), '')

    def test_only_changed_rows(self):
        """Only changed rows are returned; rows left alone or emptied keep the current file"""
        self.field(('a', 'default')).setText('/new.pqt')
        self.field(('b', 'default')).setText('')
        self.assertEqual(self.dialog.chosen(), {('a', 'default'): Path('/new.pqt')})

    def test_one_file_for_all(self):
        """With the box ticked, its one file is returned for all shanks"""
        self.dialog._all.setChecked(True)
        self.dialog._all_field.setText('/all.pqt')
        self.assertEqual(self.dialog.chosen(), {ALL_SHANKS: Path('/all.pqt')})
        self.dialog._all_field.setText('')
        self.assertEqual(self.dialog.chosen(), {})


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
