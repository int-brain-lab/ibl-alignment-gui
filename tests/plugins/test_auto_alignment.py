"""Tests for the auto-alignment: its two region-panel views and how they are computed and cached.

The model, the alignment and the result window are replaced by stand-ins: no model is used, nothing
is drawn on screen and ``ephysatlas`` does not need to be installed.
"""

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from ibl_alignment_gui.app.load_worker import TaskCancelledError
from ibl_alignment_gui.plugins import channel_prediction as cp
from ibl_alignment_gui.plugins.ephys_atlas import spatial_encoder
from ibl_alignment_gui.plugins.ephys_atlas._common import (
    AUTO_ALIGNMENT_CONFIDENCE_KEY,
    AUTO_ALIGNMENT_KEY,
    PLUGIN_KEY,
    PredictionError,
)
from iblutil.util import Bunch
from tests.plugins._runner import InlineRunner

RED, BLUE = [255, 0, 0], [0, 0, 255]
GREY = list(spatial_encoder.NO_VALUE_RGB)
OPTIONS = {AUTO_ALIGNMENT_KEY, AUTO_ALIGNMENT_CONFIDENCE_KEY}


def _fake_atlas(ids) -> SimpleNamespace:
    """An atlas whose track labels are ``ids``; region 1 is 'A' (red), region 2 is 'B' (blue)."""
    table = {1: ('A', RED), 2: ('B', BLUE)}

    def get(region_ids):
        region_ids = np.asarray(region_ids)
        return Bunch(
            id=region_ids,
            acronym=np.array([table[i][0] for i in region_ids]),
            rgb=np.array([table[i][1] for i in region_ids]),
        )

    return SimpleNamespace(
        get_labels=lambda track, mapping: np.asarray(ids), regions=SimpleNamespace(get=get)
    )


class TestPredictRegions(unittest.TestCase):
    def test_predict_regions(self):
        """One segment per run of the same region along the track, with labels and colours"""
        atlas = _fake_atlas([1, 1, 2, 2, 2])
        view = spatial_encoder.predict_regions(
            atlas, np.zeros((5, 3)), np.array([0.0, 10.0, 20.0, 30.0, 40.0])
        )
        np.testing.assert_allclose(view.region, [[0, 10], [20, 40]])
        # Depths go through metres and back, so positions are compared with a tolerance.
        np.testing.assert_allclose(view.axis_label[:, 0].astype(float), [5, 30])
        self.assertEqual(view.axis_label[:, 1].tolist(), ['A', 'B'])
        np.testing.assert_array_equal(view.colour, [RED, BLUE])


class TestPredictConfidence(unittest.TestCase):
    # Two channels at each of three depths (µm), the track reaching beyond the probe both ways.
    AXIAL = np.array([20.0, 20.0, 40.0, 40.0, 60.0, 60.0])
    TRACK = np.array([-100.0, 1000.0])

    def _view(self, p_good, valid=None):
        valid = np.ones(len(p_good), dtype=bool) if valid is None else np.asarray(valid, bool)
        return spatial_encoder.predict_confidence(
            np.asarray(p_good, dtype=float), valid, self.AXIAL, self.TRACK
        )

    def test_predict_confidence_groups_and_greys(self):
        """Depths of the same rounded confidence merge; the track beyond the probe is grey"""
        view = self._view([0.9, 0.9, 0.9, 0.9, 0.1, 0.1])
        np.testing.assert_allclose(view.region, [[-100, 10], [10, 50], [50, 70], [70, 1000]])
        self.assertEqual(view.colour[0].tolist(), GREY)
        self.assertEqual(view.colour[-1].tolist(), GREY)
        high, low = view.colour[1], view.colour[2]
        self.assertGreater(high[1], high[0])  # green: high confidence
        self.assertGreater(low[0], low[1])  # red: low confidence
        self.assertFalse(view.boundaries)

    def test_predict_confidence_unscored_channels(self):
        """Channels that were not aligned have no value and are drawn grey"""
        view = self._view([0.9, 0.9, 0.9, 0.9, 0.1, 0.1], valid=[1, 1, 1, 1, 0, 0])
        np.testing.assert_allclose(view.region, [[-100, 10], [10, 50], [50, 70], [70, 1000]])
        self.assertEqual(view.colour[2].tolist(), GREY)

    def test_predict_confidence_no_confidence(self):
        """Without any confidence value the probe is one grey segment and has no labels"""
        view = self._view([np.nan] * 6)
        np.testing.assert_allclose(view.region, [[-100, 10], [10, 70], [70, 1000]])
        self.assertTrue(all(c == GREY for c in view.colour.tolist()))
        self.assertEqual(view.axis_label.shape, (0, 2))

    def test_predict_confidence_labels(self):
        """The mean confidence is labelled in the middle of every 500 µm"""
        axial = np.arange(0.0, 1500.0, 20.0)
        view = spatial_encoder.predict_confidence(
            np.full(axial.size, 0.5), np.ones(axial.size, bool), axial, axial
        )
        self.assertEqual(
            view.axis_label.tolist(), [[250.0, 'p 0.50'], [750.0, 'p 0.50'], [1250.0, 'p 0.50']]
        )


def _no_progress(fraction, message) -> None:
    """A progress callback that ignores the reports."""


class TestAlignShank(unittest.TestCase):
    """``align_shank`` and how the plugin runs and caches it, with the alignment faked."""

    N_TRACK = 6

    def setUp(self):
        self.items = SimpleNamespace(
            name='probe00',
            config='default',
            model=SimpleNamespace(
                align_handle=SimpleNamespace(xyz_samples=np.zeros((self.N_TRACK, 3))),
                predictions=None,
            ),
            view=MagicMock(),
        )
        self.controller = SimpleNamespace(
            plugins={PLUGIN_KEY: Bunch()},
            view=None,
            run_in_thread=InlineRunner(),  # background tasks run at once
            all_shanks=['probe00'],
            shank_items={'probe00': {'default': self.items}},
            model=SimpleNamespace(
                brain_atlas=_fake_atlas([1, 1, 1, 2, 2, 2]),
                configs=['default'],
                get_current_shank=lambda shank, config: SimpleNamespace(align_exists=True),
            ),
        )
        # Four channels, two per depth, aligned to the track; the top two are confident.
        self.probe = dict(
            pid='pid', features=None, planned_xyz=None, axial_um=np.array([60.0, 60, 20, 20])
        )
        self.result = SimpleNamespace(
            p_good=np.array([0.9, 0.9, 0.1, 0.1]),
            valid=np.ones(4, dtype=bool),
            diagnostics={'channel_to_trace': np.array([0, 1, 2, 3])},
        )
        track = np.zeros((self.N_TRACK, 3), dtype=np.float32)
        self.model = object()
        # The alignment itself (features, inputs and align_histology), faked.
        self.align = MagicMock(return_value=(self.result, self.probe, track))
        self.open_window = MagicMock()
        for name, value in (('_align', self.align), ('_open_result_window', self.open_window)):
            self.enterContext(patch.object(spatial_encoder, name, value))
        # The alignment model is loaded.
        self.controller.plugins[PLUGIN_KEY][spatial_encoder.ENCODER_SPEC.key] = dict(
            model=self.model, source=('repo', '2026_W39')
        )

    def plot(self, option) -> None:
        cp.ChannelPrediction(self.controller).plot_predictions(option)

    def drawn(self) -> list:
        return [c.args[1] for c in self.items.view.plot_prediction.call_args_list]

    def test_align_returns_both_views(self):
        """align_shank returns the regions and confidence views, and what the window needs"""
        views, extra = spatial_encoder.align_shank(
            self.controller, self.items, self.model, _no_progress
        )
        self.assertEqual(set(views), OPTIONS)
        self.assertEqual(views[AUTO_ALIGNMENT_KEY].axis_label[:, 1].tolist(), ['A', 'B'])
        self.assertFalse(views[AUTO_ALIGNMENT_CONFIDENCE_KEY].boundaries)
        self.assertEqual(extra, (self.result, self.probe))
        self.align.assert_called_once_with(self.controller, self.items, self.model, _no_progress)
        self.open_window.assert_not_called()  # opened afterwards, on the main thread

    def test_window_opened_after_compute(self):
        """Running the model opens the shank's result window once, from on_computed"""
        self.plot(AUTO_ALIGNMENT_KEY)
        self.open_window.assert_called_once_with(
            self.controller, self.items, self.result, self.probe
        )

    def test_align_cancelled(self):
        """A cancelled alignment caches and draws nothing and opens no window"""
        self.align.side_effect = TaskCancelledError()
        self.plot(AUTO_ALIGNMENT_KEY)
        self.assertEqual(dict(self.items.model.predictions), {})
        self.assertEqual(self.drawn(), [])
        self.open_window.assert_not_called()

    def test_aligns_once_for_both_options(self):
        """Selecting one option aligns once and caches both; the other is drawn without aligning"""
        self.plot(AUTO_ALIGNMENT_KEY)
        self.plot(AUTO_ALIGNMENT_CONFIDENCE_KEY)
        self.plot(AUTO_ALIGNMENT_KEY)
        self.assertEqual(self.align.call_count, 1)
        predictions = self.items.model.predictions
        self.assertEqual(set(predictions), OPTIONS)
        self.assertEqual(
            self.drawn(),
            [
                predictions[AUTO_ALIGNMENT_KEY],
                predictions[AUTO_ALIGNMENT_CONFIDENCE_KEY],
                predictions[AUTO_ALIGNMENT_KEY],
            ],
        )

    def test_nothing_stored_on_shank(self):
        """Apart from the cached views, nothing about the alignment is kept on the shank"""
        self.plot(AUTO_ALIGNMENT_KEY)
        self.assertEqual(set(vars(self.items.model)), {'align_handle', 'predictions'})


def _fake_model(vintage: str | None = '2026_W39') -> SimpleNamespace:
    """An alignment model with two features."""
    index = {'vintage': vintage} if vintage else {}
    return SimpleNamespace(encoder=SimpleNamespace(index=index), features=['a', 'b'])


def _features_table() -> pd.DataFrame:
    """Bottom channel first, as a table may be ordered; one feature is not finite."""
    return pd.DataFrame(
        {
            'a': [1.0, np.nan],
            'b': [2.0, np.inf],
            'axial_um': [20.0, 60.0],
            'x_target': [1.0, 2.0],
            'y_target': [3.0, 4.0],
            'z_target': [5.0, 6.0],
        }
    )


class TestAlignmentInputs(unittest.TestCase):
    """``alignment_inputs``: a features table to the alignment's arrays, no GUI involved."""

    def test_top_first_and_non_finite_zero(self):
        """Rows are ordered top of the probe first, and non-finite features set to 0"""
        probe = spatial_encoder.alignment_inputs(_fake_model(), _features_table())
        np.testing.assert_array_equal(probe['axial_um'], [60.0, 20.0])
        np.testing.assert_array_equal(probe['features'], [[0.0, 0.0], [1.0, 2.0]])

    def test_planned_trajectory(self):
        """The planned trajectory is taken from the target columns when the table has them"""
        df = _features_table()
        probe = spatial_encoder.alignment_inputs(_fake_model(), df)
        np.testing.assert_array_equal(probe['planned_xyz'], [[2, 4, 6], [1, 3, 5]])
        probe = spatial_encoder.alignment_inputs(_fake_model(), df.drop(columns='z_target'))
        self.assertIsNone(probe['planned_xyz'])

    def test_missing_feature(self):
        """A table lacking a feature the model needs raises PredictionError naming it"""
        with self.assertRaisesRegex(PredictionError, r"column\(s\): \['b'\](.|\n)*2026_W39"):
            spatial_encoder.alignment_inputs(_fake_model(), _features_table().drop(columns='b'))


class TestAlignmentInputsVintage(unittest.TestCase):
    def test_missing_feature_suggests_features_vintage(self):
        """With the table's vintage known, the error suggests that release of the model"""
        df = _features_table().drop(columns='b')
        df.attrs['features_version'] = '2026_W26'
        with self.assertRaisesRegex(PredictionError, 'load a 2026_W26 release of the model'):
            spatial_encoder.alignment_inputs(_fake_model(), df)


class TestAlign(unittest.TestCase):
    """``_align``: the shank's features and the alignment, with both faked."""

    def setUp(self):
        self.items = SimpleNamespace(
            name='probe00',
            model=SimpleNamespace(
                pid='pid', align_handle=SimpleNamespace(xyz_samples=np.zeros((3, 3)))
            ),
        )
        self.controller = SimpleNamespace(model=SimpleNamespace())
        self.model = _fake_model()
        self.features_for_vintage = self.enterContext(
            patch.object(spatial_encoder, 'features_for_vintage', return_value=_features_table())
        )
        # The alignment, imported from ephysatlas when it runs.
        alignment = types.ModuleType('ephysatlas.alignment')
        self.align_histology = alignment.align_histology = MagicMock(return_value='result')
        stubs = {'ephysatlas': types.ModuleType('ephysatlas'), 'ephysatlas.alignment': alignment}
        self.enterContext(patch.dict(sys.modules, stubs))

    def align(self):
        return spatial_encoder._align(self.controller, self.items, self.model, _no_progress)

    def test_features_of_model_vintage(self):
        """The features come from the table of the model's vintage, reporting progress"""
        self.align()
        self.features_for_vintage.assert_called_once_with(
            self.controller, self.items, '2026_W39', _no_progress
        )

    def test_aligns_with_session_pid(self):
        """The alignment gets the inputs, the track top first and the session's pid"""
        result, probe, _ = self.align()
        self.assertEqual(result, 'result')
        self.assertEqual(probe['pid'], 'pid')
        kwargs = self.align_histology.call_args.kwargs
        self.assertEqual((kwargs['pid'], kwargs['progress']), ('pid', _no_progress))
        np.testing.assert_array_equal(self.align_histology.call_args.args[1], probe['features'])

    def test_no_features(self):
        """No features for the probe raises PredictionError"""
        self.features_for_vintage.return_value = None
        with self.assertRaisesRegex(PredictionError, 'No ephys-atlas features'):
            self.align()

    def test_no_vintage(self):
        """A release naming no vintage is refused before any features are loaded"""
        self.model = _fake_model(vintage=None)
        with self.assertRaisesRegex(PredictionError, 'vintage'):
            self.align()
        self.features_for_vintage.assert_not_called()

    def test_cancel_passes_through(self):
        """Cancel during the alignment is not turned into an error"""
        self.align_histology.side_effect = TaskCancelledError()
        with self.assertRaises(TaskCancelledError):
            self.align()

    def test_alignment_error_reported(self):
        """An error from the alignment becomes a PredictionError with its message"""
        self.align_histology.side_effect = ValueError('too few usable channels')
        with self.assertRaisesRegex(PredictionError, 'too few usable channels'):
            self.align()


if __name__ == '__main__':
    unittest.main()
