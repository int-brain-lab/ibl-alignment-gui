"""Tests for loading the Channel Prediction models: the load dialog and the cached model.

The shared ``load_model_dialog`` and ``get_model`` run with the spec of each region decoder
(``inference``) and of the auto-alignment encoder (``spatial_encoder``). Model loading, the dialogs
and the prediction caches are replaced by stand-ins: no model, network or window is used, and
``ephysatlas`` does not need to be installed.
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

DECODER = 'XGBoost region prediction'
NEW_SOURCE = ('org/new-model', '2026_W40')


def _fake_controller(features: bool = True) -> SimpleNamespace:
    """A controller with the plugin state and one shank that does or does not have features."""
    handler = SimpleNamespace(raw_data={'features': Bunch(exists=features)})
    return SimpleNamespace(
        plugins={PLUGIN_KEY: Bunch()},
        view=None,
        model=SimpleNamespace(shanks={'probe00': {'default': handler}}),
    )


class _ModelLoadingTests:
    """The tests, run by each backend's subclass below.

    A subclass names the backend's spec, state key and defaults, and fakes its model loading in
    ``patch_loader``.
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
        self.test_spec = dataclasses.replace(self.spec, invalidate=self.invalidate)
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

    def load_model_dialog(self) -> bool:
        return _common.load_model_dialog(self.controller, self.test_spec)

    def get_model(self):
        return _common.get_model(self.controller, self.test_spec)

    def set_state(self, model=None, source=None) -> None:
        self.controller.plugins[PLUGIN_KEY][self.key] = dict(model=model, source=source)

    def state(self) -> dict:
        return self.controller.plugins[PLUGIN_KEY][self.key]

    # -- load_model_dialog --------------------------------------------------------------------

    def test_no_features_warns(self):
        """Without features the user is warned and no dialog opens"""
        self.controller = _fake_controller(features=False)
        self.assertFalse(self.load_model_dialog())
        self.warning.assert_called_once()
        self.assertEqual(self.dialogs, [])
        self.assertEqual(self.loads, [])

    def test_prefills_defaults(self):
        """The first time, the dialog offers the default repository and release tag"""
        self.load_model_dialog()
        self.assertEqual(
            self.dialogs, [(self.title, self.default_repo, self.default_revision, None)]
        )

    def test_prefills_previous_hub_source(self):
        """The dialog offers the Hub source loaded before"""
        self.set_state(model=object(), source=('org/old-model', '2026_W30'))
        self.load_model_dialog()
        self.assertEqual(self.dialogs, [(self.title, 'org/old-model', '2026_W30', None)])

    def test_prefills_previous_local_source(self):
        """A local release loaded before goes in the local field, with the default repository"""
        with tempfile.TemporaryDirectory() as local:
            self.set_state(model=object(), source=(local, '2026_W30'))
            self.load_model_dialog()
            self.assertEqual(
                self.dialogs, [(self.title, self.default_repo, '2026_W30', Path(local))]
            )

    def test_cancel_without_model(self):
        """Cancelling with no model loaded loads nothing"""
        self.choose(None)
        self.assertFalse(self.load_model_dialog())
        self.assertEqual(self.loads, [])
        self.assertIsNone(self.state()['model'])

    def test_cancel_keeps_loaded_model(self):
        """Cancelling keeps the model already loaded"""
        loaded = object()
        self.set_state(model=loaded, source=('org/old-model', '2026_W30'))
        self.choose(None)
        self.assertTrue(self.load_model_dialog())
        self.assertIs(self.state()['model'], loaded)
        self.assertEqual(self.loads, [])

    def test_loads_new_source(self):
        """A new source is loaded, stored and clears the cached predictions"""
        self.choose(NEW_SOURCE)
        self.assertTrue(self.load_model_dialog())
        self.assertEqual(self.loads, [NEW_SOURCE])
        self.assertIs(self.state()['model'], self.model)
        self.assertEqual(tuple(self.state()['source']), NEW_SOURCE)
        self.invalidate.assert_called_once()

    def test_same_source_not_reloaded(self):
        """Choosing the source already loaded does not load it again"""
        loaded = object()
        self.set_state(model=loaded, source=NEW_SOURCE)
        self.choose(NEW_SOURCE)
        self.assertTrue(self.load_model_dialog())
        self.assertEqual(self.loads, [])
        self.assertIs(self.state()['model'], loaded)
        self.invalidate.assert_not_called()

    def test_load_failure(self):
        """A failed load is reported with the source and error, and nothing is stored"""
        self.load_error = RuntimeError('network down')
        self.choose(NEW_SOURCE)
        self.assertFalse(self.load_model_dialog())
        self.critical.assert_called_once()
        message = self.critical.call_args.args[2]
        self.assertIn('org/new-model@2026_W40', message)
        self.assertIn('network down', message)
        self.assertIsNone(self.state()['model'])
        self.invalidate.assert_not_called()

    # -- get_model ----------------------------------------------------------------------------

    def test_cached_model(self):
        """A loaded model is returned without a dialog or a load"""
        loaded = object()
        self.set_state(model=loaded, source=NEW_SOURCE)
        self.assertIs(self.get_model(), loaded)
        self.assertEqual(self.dialogs, [])
        self.assertEqual(self.loads, [])

    def test_preset_source_loads_without_dialog(self):
        """A preset source (as the launcher example sets) is loaded without the dialog"""
        self.set_state(source=NEW_SOURCE)
        self.assertIs(self.get_model(), self.model)
        self.assertEqual(self.loads, [NEW_SOURCE])
        self.assertEqual(self.dialogs, [])
        self.assertIs(self.state()['model'], self.model)

    def test_no_model_opens_dialog(self):
        """With no model, the dialog opens and the chosen model is returned"""
        self.choose(NEW_SOURCE)
        self.assertIs(self.get_model(), self.model)
        self.assertEqual(len(self.dialogs), 1)
        self.assertEqual(self.loads, [NEW_SOURCE])

    def test_no_model_cancelled(self):
        """With no model, cancelling the dialog returns None"""
        self.choose(None)
        self.assertIsNone(self.get_model())
        self.assertEqual(self.loads, [])

    def test_preset_source_failure(self):
        """A preset source that fails to load raises PredictionError, without a dialog"""
        self.load_error = RuntimeError('network down')
        self.set_state(source=NEW_SOURCE)
        with self.assertRaisesRegex(PredictionError, 'org/new-model@2026_W40(.|\n)*network down'):
            self.get_model()
        self.critical.assert_not_called()
        self.assertIsNone(self.state()['model'])

    def test_load_shows_progress(self):
        """Loading runs behind a progress dialog titled with the spec's title"""
        self.choose(NEW_SOURCE)
        with patch.object(
            _common, 'run_with_progress', wraps=_common.run_with_progress
        ) as run_with_progress:
            self.load_model_dialog()
        run_with_progress.assert_called_once()
        self.assertEqual(run_with_progress.call_args.args[1], self.spec.title)

    # -- ModelSpec ----------------------------------------------------------------------------

    def test_spec(self):
        """The spec holds the state key, dialog title and defaults the dialog uses"""
        self.assertEqual(
            (self.spec.key, self.spec.title, self.spec.default_source()),
            (self.key, self.title, (self.default_repo, self.default_revision)),
        )

    def test_spec_load(self):
        """The spec's load function loads the source and reports progress to the end"""
        reports = []
        model = self.spec.load(NEW_SOURCE, lambda fraction, message: reports.append(fraction))
        self.assertIs(model, self.model)
        self.assertEqual(self.loads, [NEW_SOURCE])
        self.assertEqual(reports[-1], 1.0)


class TestDecoderModelLoading(_ModelLoadingTests, unittest.TestCase):
    """The shared loading with a region decoder's spec."""

    spec = inference.DECODER_SPECS[DECODER]
    key = DECODER
    default_repo = inference.DECODERS[DECODER].repo
    default_revision = inference.MODEL_REVISION
    title = f'Load {DECODER} model'

    def patch_loader(self):
        # The decoders import ephysatlas.load_pretrained when they load a model.
        ephysatlas = types.ModuleType('ephysatlas')
        ephysatlas.load_pretrained = lambda repo, revision=None: self._fake_load(repo, revision)
        self.enterContext(patch.dict(sys.modules, {'ephysatlas': ephysatlas}))


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

    def test_needs_torch(self):
        """Only the XGBoost decoder runs without torch"""
        self.assertEqual(
            {spec.key: spec.needs_torch for spec in self.SPECS},
            {
                'XGBoost region prediction': False,
                'ProbeTransformer region prediction': True,
                'Encoding': True,
            },
        )

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
