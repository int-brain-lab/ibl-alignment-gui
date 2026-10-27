"""Tests for loading the Channel Prediction models: the load dialog and the cached model.

Loading from the Plugins menu, selecting a model's option and the shared loading steps run with
the spec of each region decoder (``inference``) and of the auto-alignment encoder
(``spatial_encoder``). Model loading, the dialogs, the background tasks and the prediction caches
are replaced by stand-ins: no model, network, thread or window is used, and ``ephysatlas`` does
not need to be installed.
"""

import dataclasses
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from qtpy import QtWidgets

from ibl_alignment_gui.app.load_worker import TaskCancelledError
from ibl_alignment_gui.plugins import channel_prediction as cp
from ibl_alignment_gui.plugins.ephys_atlas import _common, inference, spatial_encoder
from ibl_alignment_gui.plugins.ephys_atlas._common import (
    AUTO_ALIGNMENT_CONFIDENCE_KEY,
    AUTO_ALIGNMENT_KEY,
    PLUGIN_KEY,
    ModelSourceDialog,
    ModelSpec,
    PredictionError,
)
from iblutil.util import Bunch
from tests.plugins._runner import InlineRunner

DECODER = 'XGBoost region prediction'
NEW_SOURCE = ('org/new-model', '2026_W40')


def _fake_controller(features: bool = True) -> SimpleNamespace:
    """A controller with the plugin state and one aligned shank, with or without features.

    Background tasks run at once (:class:`InlineRunner`); the view records the options selected.
    """
    handler = SimpleNamespace(raw_data={'features': Bunch(exists=features)})
    items = SimpleNamespace(
        name='probe00', model=SimpleNamespace(predictions=None), view=MagicMock()
    )
    view = MagicMock()
    return SimpleNamespace(
        plugins={PLUGIN_KEY: Bunch()},
        view=view,
        run_in_thread=InlineRunner(),
        all_shanks=['probe00'],
        shank_items={'probe00': {'default': items}},
        model=SimpleNamespace(
            shanks={'probe00': {'default': handler}},
            configs=['default'],
            get_current_shank=lambda shank, config: SimpleNamespace(align_exists=True),
        ),
    )


class _ModelLoadingTests:
    """The stand-ins, and the tests of what differs between models, run by each backend's subclass.

    A subclass names the backend's spec, state key and defaults, and fakes its model loading in
    ``patch_loader``. The loading flow itself (:class:`_LoadingFlowTests`) is the same code for
    every model, so it runs with the decoder only.
    """

    spec: ModelSpec
    key: str
    default_repo: str
    default_revision: str
    title: str

    @classmethod
    def setUpClass(cls):
        # Run without a display server unless one has been chosen explicitly.
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.controller = _fake_controller()
        self.loads = []  # the (repo, revision) of every model load
        self.load_error = None  # raised by the fake loader when set
        # Has what both backends read from a loaded model.
        self.model = SimpleNamespace(
            index={'model_class': 'FakeModel', 'vintage': '2026_W40'},
            inputs={'features': ['a']},
            config={'classes': [1]},
            conf_model=object(),
        )
        self.dialogs = []  # the pre-filled fields of every dialog opened
        self.dialog_result = QtWidgets.QDialog.Rejected
        self.dialog_source = None
        self.enterContext(patch.object(_common, 'ModelSourceDialog', self._fake_dialog_class()))
        self.warning = self.enterContext(patch.object(QtWidgets.QMessageBox, 'warning'))
        self.critical = self.enterContext(patch.object(QtWidgets.QMessageBox, 'critical'))
        # The spec as used by the tests: clearing the cached predictions is recorded, not done.
        self.invalidate = MagicMock()
        # Running the model on a shank is recorded and gives one view, so loading is tested alone.
        self.compute = MagicMock(return_value=({self.spec.options[0]: Bunch()}, None))
        self.test_spec = dataclasses.replace(
            self.spec, invalidate=self.invalidate, compute=self.compute, on_computed=None
        )
        self.patch_loader()

    # -- stand-ins --------------------------------------------------------------------------

    def _fake_load(self, repo: str, revision: str):
        self.loads.append((repo, revision))
        if self.load_error is not None:
            raise self.load_error
        return self.model

    def _fake_dialog_class(self) -> type:
        test = self

        class FakeDialog:
            def __init__(self, *args):
                test.dialogs.append(test.prefill(*args))

            def exec(self):
                return test.dialog_result

            def source(self):
                return test.dialog_source

        return FakeDialog

    @staticmethod
    def prefill(parent, spec, repo, revision, local_dir) -> tuple:
        """What the dialog was opened with: ``(title, repo, revision, local_dir)``."""
        return spec.title, repo, revision, local_dir

    def choose(self, source: tuple[str, str] | None) -> None:
        """Make the dialog return OK with ``source``, or Cancel when ``source`` is None."""
        self.dialog_source = source
        self.dialog_result = (
            QtWidgets.QDialog.Rejected if source is None else QtWidgets.QDialog.Accepted
        )

    def menu_load(self) -> None:
        """Load the model from its Plugins menu item."""
        cp._load_model(self.controller, self.test_spec)

    def plot(self) -> None:
        """Select the model's first prediction option."""
        cp._predict(self.controller, self.test_spec, self.test_spec.options[0])

    def shown(self) -> bool:
        """Whether, after loading from the menu, the model's first option was selected."""
        calls = self.controller.view.trigger_menu_option.call_args_list
        return [c.args for c in calls] == [('prediction', self.test_spec.options[0])]

    def set_state(self, model=None, source=None) -> None:
        self.controller.plugins[PLUGIN_KEY][self.key] = dict(model=model, source=source)

    def state(self) -> dict:
        return self.controller.plugins[PLUGIN_KEY][self.key]

    def stored_model(self):
        """The model kept in the plugin state, or None."""
        return (self.controller.plugins[PLUGIN_KEY].get(self.key) or {}).get('model')

    # -- what differs between the models ----------------------------------------------------

    def test_prefills_defaults(self):
        """The first time, the dialog offers the default repository and release tag"""
        self.menu_load()
        self.assertEqual(
            self.dialogs, [(self.title, self.default_repo, self.default_revision, None)]
        )

    def test_load_model(self):
        """load_model loads the source with the spec, reporting progress, and stores nothing"""
        reports = []
        model = _common.load_model(self.test_spec, NEW_SOURCE, lambda f, m: reports.append(f))
        self.assertIs(model, self.model)
        self.assertEqual(self.loads, [NEW_SOURCE])
        self.assertEqual(reports[-1], 1.0)
        self.assertNotIn(self.key, self.controller.plugins[PLUGIN_KEY])

    # -- ModelSpec ----------------------------------------------------------------------------

    def test_spec(self):
        """The spec holds the state key, dialog title and defaults the dialog uses"""
        self.assertEqual(
            (self.spec.key, self.spec.title, self.spec.default_source()),
            (self.key, self.title, (self.default_repo, self.default_revision)),
        )


class _LoadingFlowTests:
    """The loading flow: the same code for every model, so run with one model only."""

    # -- loading from the Plugins menu ---------------------------------------------------------

    def test_no_features_warns(self):
        """Without features the user is warned and no dialog opens"""
        self.controller = _fake_controller(features=False)
        self.menu_load()
        self.warning.assert_called_once()
        self.assertEqual(self.dialogs, [])
        self.assertEqual(self.loads, [])
        self.assertFalse(self.shown())

    def test_prefills_previous_hub_source(self):
        """The dialog offers the Hub source loaded before"""
        self.set_state(model=object(), source=('org/old-model', '2026_W30'))
        self.menu_load()
        self.assertEqual(self.dialogs, [(self.title, 'org/old-model', '2026_W30', None)])

    def test_prefills_previous_local_source(self):
        """A local release loaded before goes in the local field, with the default repository"""
        with tempfile.TemporaryDirectory() as local:
            self.set_state(model=object(), source=(local, '2026_W30'))
            self.menu_load()
            self.assertEqual(
                self.dialogs, [(self.title, self.default_repo, '2026_W30', Path(local))]
            )

    def test_cancel_without_model(self):
        """Cancelling with no model loaded loads nothing and shows nothing"""
        self.choose(None)
        self.menu_load()
        self.assertEqual(self.loads, [])
        self.assertIsNone(self.stored_model())
        self.assertFalse(self.shown())

    def test_cancel_keeps_loaded_model(self):
        """Cancelling keeps the model already loaded, and shows it"""
        loaded = object()
        self.set_state(model=loaded, source=('org/old-model', '2026_W30'))
        self.choose(None)
        self.menu_load()
        self.assertIs(self.state()['model'], loaded)
        self.assertEqual(self.loads, [])
        self.assertTrue(self.shown())

    def test_loads_new_source(self):
        """A new source is loaded in the background, stored, clears the predictions and shown"""
        self.choose(NEW_SOURCE)
        self.menu_load()
        self.assertEqual(self.loads, [NEW_SOURCE])
        self.assertIs(self.state()['model'], self.model)
        self.assertEqual(tuple(self.state()['source']), NEW_SOURCE)
        self.invalidate.assert_called_once()
        self.assertTrue(self.shown())

    def test_same_source_not_reloaded(self):
        """Choosing the source already loaded does not load it again"""
        loaded = object()
        self.set_state(model=loaded, source=NEW_SOURCE)
        self.choose(NEW_SOURCE)
        self.menu_load()
        self.assertEqual(self.loads, [])
        self.assertEqual(self.controller.run_in_thread.tasks, [])
        self.assertIs(self.state()['model'], loaded)
        self.invalidate.assert_not_called()
        self.assertTrue(self.shown())

    def test_load_failure(self):
        """A failed load is reported with the source and error, and nothing is stored"""
        self.load_error = RuntimeError('network down')
        self.choose(NEW_SOURCE)
        self.menu_load()
        self.critical.assert_called_once()
        message = self.critical.call_args.args[2]
        self.assertIn('org/new-model@2026_W40', message)
        self.assertIn('network down', message)
        self.assertIsNone(self.stored_model())
        self.invalidate.assert_not_called()
        self.assertFalse(self.shown())

    def test_cancel_during_load_keeps_model(self):
        """Cancelling the load stores nothing and keeps the current model"""
        current = object()
        self.set_state(model=current, source=('org/old-model', '2026_W30'))
        self.choose(NEW_SOURCE)
        self.controller.run_in_thread.cancel_at = 1  # Cancel at the load's first report
        self.menu_load()
        self.assertIs(self.state()['model'], current)
        self.assertEqual(tuple(self.state()['source']), ('org/old-model', '2026_W30'))
        self.invalidate.assert_not_called()
        self.critical.assert_not_called()
        self.assertFalse(self.shown())

    def test_load_shows_progress(self):
        """Loading runs as a cancellable task titled with the spec's title, reporting progress"""
        self.choose(NEW_SOURCE)
        self.menu_load()
        tasks = self.controller.run_in_thread.tasks
        self.assertEqual(
            [(t['title'], t['cancellable']) for t in tasks], [(self.spec.title, True)]
        )
        self.assertIn('downloaded on first use', self.controller.run_in_thread.progress[0])

    def test_another_task_running(self):
        """While another task runs, loading is refused with a warning"""
        self.choose(NEW_SOURCE)
        self.controller.run_in_thread.busy = True
        self.menu_load()
        self.warning.assert_called_once()
        self.assertEqual(self.loads, [])

    # -- choose_source, load_model, store_model ----------------------------------------------

    def test_choose_source(self):
        """The chosen source is returned; Cancel returns None; nothing is loaded"""
        self.choose(NEW_SOURCE)
        self.assertEqual(_common.choose_source(self.controller, self.test_spec), NEW_SOURCE)
        self.choose(None)
        self.assertIsNone(_common.choose_source(self.controller, self.test_spec))
        self.assertEqual(self.loads, [])

    def test_load_model_failure(self):
        """A failed load raises PredictionError naming the source and the reason"""
        self.load_error = RuntimeError('network down')
        with self.assertRaisesRegex(PredictionError, 'org/new-model@2026_W40(.|\n)*network down'):
            _common.load_model(self.test_spec, NEW_SOURCE, lambda f, m: None)

    def test_load_model_cancel_passes_through(self):
        """Cancel while loading is not turned into a load failure"""
        self.load_error = TaskCancelledError()
        with self.assertRaises(TaskCancelledError):
            _common.load_model(self.test_spec, NEW_SOURCE, lambda f, m: None)

    def test_store_model(self):
        """store_model keeps the model and its source and clears its old predictions"""
        model = object()
        _common.store_model(self.controller, self.test_spec, model, list(NEW_SOURCE))
        self.assertEqual(self.state(), dict(model=model, source=NEW_SOURCE))
        self.invalidate.assert_called_once_with(self.controller)

    # -- selecting one of the model's region-plot options --------------------------------------

    def test_cached_model(self):
        """A loaded model is used without a dialog or a load"""
        loaded = object()
        self.set_state(model=loaded, source=NEW_SOURCE)
        self.plot()
        self.assertEqual(self.dialogs, [])
        self.assertEqual(self.loads, [])
        self.assertIs(self.compute.call_args.args[2], loaded)

    def test_preset_source_loads_without_dialog(self):
        """A preset source (as the launcher example sets) is loaded without the dialog"""
        self.set_state(source=NEW_SOURCE)
        self.plot()
        self.assertEqual(self.loads, [NEW_SOURCE])
        self.assertEqual(self.dialogs, [])
        self.assertIs(self.state()['model'], self.model)
        self.assertIs(self.compute.call_args.args[2], self.model)

    def test_no_model_opens_dialog(self):
        """With no model, the dialog opens and the chosen model is loaded and used"""
        self.choose(NEW_SOURCE)
        self.plot()
        self.assertEqual(len(self.dialogs), 1)
        self.assertEqual(self.loads, [NEW_SOURCE])
        self.assertIs(self.stored_model(), self.model)
        self.assertIs(self.compute.call_args.args[2], self.model)

    def test_no_model_cancelled(self):
        """With no model, cancelling the dialog loads and computes nothing"""
        self.choose(None)
        self.plot()
        self.assertEqual(self.loads, [])
        self.compute.assert_not_called()

    def test_preset_source_failure(self):
        """A preset source that fails to load gives one warning, without a dialog"""
        self.load_error = RuntimeError('network down')
        self.set_state(source=NEW_SOURCE)
        self.plot()
        self.warning.assert_called_once()
        self.assertRegex(
            self.warning.call_args.args[2], 'org/new-model@2026_W40(.|\n)*network down'
        )
        self.critical.assert_not_called()
        self.assertEqual(self.dialogs, [])
        self.assertIsNone(self.state()['model'])
        self.compute.assert_not_called()


class TestDecoderModelLoading(_LoadingFlowTests, _ModelLoadingTests, unittest.TestCase):
    """The shared loading with a region decoder's spec."""

    spec = inference.DECODER_SPECS[DECODER]
    key = DECODER
    default_repo = inference.DECODERS[DECODER].repo
    default_revision = '2026_W41'  # the decoder's release in the stand-in ephysatlas.models
    title = f'Load {DECODER} model'

    def patch_loader(self):
        # The decoders read their default release from ephysatlas.models and load a model with
        # ephysatlas.load_pretrained.
        ephysatlas = types.ModuleType('ephysatlas')
        ephysatlas.load_pretrained = lambda repo, revision=None: self._fake_load(repo, revision)
        models = types.ModuleType('ephysatlas.models')
        models.DECODER_VINTAGES = {self.default_repo: self.default_revision}
        stubs = {'ephysatlas': ephysatlas, 'ephysatlas.models': models}
        self.enterContext(patch.dict(sys.modules, stubs))


class TestEncoderModelLoading(_ModelLoadingTests, unittest.TestCase):
    """The shared loading with the auto-alignment encoder's spec."""

    spec = spatial_encoder.ENCODER_SPEC
    key = spatial_encoder.MODEL_NAME
    default_repo = 'int-brain-lab/ea-encoder-channel'
    default_revision = '2026_W39'
    title = 'Load alignment model'

    def patch_loader(self):
        # The encoder imports these from ephysatlas.alignment when it loads a model.
        alignment = types.ModuleType('ephysatlas.alignment')
        alignment.ChannelModel = self._fake_load
        alignment.CHANNEL_MODEL_REPO_ID = self.default_repo
        alignment.VINTAGE = self.default_revision
        stubs = {'ephysatlas': types.ModuleType('ephysatlas'), 'ephysatlas.alignment': alignment}
        self.enterContext(patch.dict(sys.modules, stubs))


class TestDecoderSpecs(unittest.TestCase):
    """What the decoder specs do beyond the shared loading."""

    def test_invalidates_only_itself(self):
        """A decoder's spec clears its own cached predictions, not the other decoder's"""
        keys = [name + suffix for name in inference.DECODERS for suffix in ('', ' (cumulative)')]
        items = SimpleNamespace(model=SimpleNamespace(predictions=Bunch.fromkeys(keys, object())))
        controller = SimpleNamespace(
            all_shanks=['a'],
            shank_items={'a': {'default': items}},
            model=SimpleNamespace(
                configs=['default'],
                get_current_shank=lambda shank, config: SimpleNamespace(align_exists=True),
            ),
        )
        inference.DECODER_SPECS[DECODER].invalidate(controller)
        other = 'ProbeTransformer region prediction'
        self.assertEqual(set(items.model.predictions), {other, other + ' (cumulative)'})


class TestModelSpecs(unittest.TestCase):
    """The registry facts each model's spec carries."""

    SPECS = (*inference.DECODER_SPECS.values(), spatial_encoder.ENCODER_SPEC)

    def test_options(self):
        """Each model offers its own region-plot options"""
        for name, spec in inference.DECODER_SPECS.items():
            self.assertEqual(spec.options, (name, name + ' (cumulative)'))
        self.assertEqual(
            spatial_encoder.ENCODER_SPEC.options,
            (AUTO_ALIGNMENT_KEY, AUTO_ALIGNMENT_CONFIDENCE_KEY),
        )

    def test_compute(self):
        """Each model computes with its backend: a decoder runs decode_shank with its own name"""
        for name, spec in inference.DECODER_SPECS.items():
            self.assertIs(spec.compute.func, inference.decode_shank)
            self.assertEqual((spec.compute.args, spec.compute.keywords), ((), {'name': name}))
        self.assertIs(spatial_encoder.ENCODER_SPEC.compute, spatial_encoder.align_shank)

    def test_unique(self):
        """No two models share a key, an option or a manifest model class"""
        options = [option for spec in self.SPECS for option in spec.options]
        for values in (
            [spec.key for spec in self.SPECS],
            options,
            [spec.model_class for spec in self.SPECS],
        ):
            self.assertEqual(len(values), len(set(values)))


class TestModelSourceDialog(unittest.TestCase):
    """The shared load dialog itself, built from a stand-in spec (offscreen)."""

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.spec = ModelSpec(
            key='Fake model',
            title='Load fake model',
            default_source=lambda: ('org/default-model', '2026_W39'),
            note='A note about the fake model.',
            load=None,
            invalidate=None,
            options=('Fake model',),
            compute=None,
            needs_torch=False,
            model_class='FakeModel',
        )

    def _dialog(self, repo='org/old-model', revision='2026_W30', local_dir=None):
        dialog = ModelSourceDialog(None, self.spec, repo, revision, local_dir)
        self.addCleanup(dialog.deleteLater)
        return dialog

    def _field(self, dialog, text: str) -> QtWidgets.QLineEdit:
        """The line edit holding ``text``."""
        return next(e for e in dialog.findChildren(QtWidgets.QLineEdit) if e.text() == text)

    def test_shows_spec(self):
        """The title and note are the spec's, and the fields open with the given source"""
        dialog = self._dialog(local_dir=Path('/data/release'))
        self.assertEqual(dialog.windowTitle(), 'Load fake model')
        notes = [label.text() for label in dialog.findChildren(QtWidgets.QLabel)]
        self.assertIn('A note about the fake model.', notes)
        texts = [e.text() for e in dialog.findChildren(QtWidgets.QLineEdit)]
        self.assertEqual(texts, ['org/old-model', '2026_W30', str(Path('/data/release'))])

    def test_source_from_fields(self):
        """The source is the edited repository and tag, or the local folder when given"""
        dialog = self._dialog()
        self._field(dialog, 'org/old-model').setText('org/new-model')
        self._field(dialog, '2026_W30').setText('2026_W40')
        self.assertEqual(dialog.source(), ('org/new-model', '2026_W40'))
        self._field(dialog, '').setText('/data/release')
        self.assertEqual(dialog.source(), ('/data/release', '2026_W40'))

    def test_empty_fields_use_defaults(self):
        """Cleared repository and tag fields fall back to the spec's defaults"""
        dialog = self._dialog()
        self._field(dialog, 'org/old-model').clear()
        self._field(dialog, '2026_W30').clear()
        self.assertEqual(dialog.source(), ('org/default-model', '2026_W39'))


if __name__ == '__main__':
    unittest.main()
