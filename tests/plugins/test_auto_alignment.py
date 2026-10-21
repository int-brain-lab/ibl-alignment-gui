"""Tests for the auto-alignment: its two region-panel views and how they are computed and cached.

The model, the alignment and the result window are replaced by stand-ins: no model is used, nothing
is drawn on screen and ``ephysatlas`` does not need to be installed.
"""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from ibl_alignment_gui.plugins import channel_prediction as cp
from ibl_alignment_gui.plugins.ephys_atlas._common import (
    AUTO_ALIGNMENT_CONFIDENCE_KEY,
    AUTO_ALIGNMENT_KEY,
    PLUGIN_KEY,
)
from iblutil.util import Bunch
from tests.plugins._stubs import SPATIAL_ENCODER, import_spatial_encoder

spatial_encoder = import_spatial_encoder()

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


class TestAlignShank(unittest.TestCase):
    """``align_shank`` and how the plugin caches its views, with the alignment faked."""

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
        self.align = MagicMock(return_value=self.result)  # stands in for run_with_progress
        self.open_window = MagicMock()
        for name, value in (
            ('get_model', MagicMock(return_value=object())),
            ('_probe_inputs', MagicMock(return_value=self.probe)),
            ('run_with_progress', self.align),
            ('_open_result_window', self.open_window),
        ):
            self.enterContext(patch.object(spatial_encoder, name, value))
        # The plugin imports spatial_encoder lazily: make that import find this module.
        self.enterContext(patch.dict(sys.modules, {SPATIAL_ENCODER: spatial_encoder}))

    def plot(self, option) -> None:
        cp._plot_region_panels(self.controller, option, cp.compute_spatial_encoder_predictions)

    def drawn(self) -> list:
        return [c.args[1] for c in self.items.view.plot_histology.call_args_list]

    def test_align_returns_both_views(self):
        """align_shank returns the regions and confidence views and opens the result window"""
        views = spatial_encoder.align_shank(self.controller, self.items)
        self.assertEqual(set(views), OPTIONS)
        self.assertEqual(views[AUTO_ALIGNMENT_KEY].axis_label[:, 1].tolist(), ['A', 'B'])
        self.assertFalse(views[AUTO_ALIGNMENT_CONFIDENCE_KEY].boundaries)
        self.open_window.assert_called_once_with(
            self.controller, self.items, self.result, self.probe
        )

    def test_align_cancelled(self):
        """A cancelled alignment returns None and opens no window"""
        self.align.side_effect = spatial_encoder.AlignmentCancelledError()
        self.assertIsNone(spatial_encoder.align_shank(self.controller, self.items))
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

    def test_invalidate_clears_both(self):
        """Invalidating clears both views, so the next selection aligns again"""
        self.plot(AUTO_ALIGNMENT_KEY)
        spatial_encoder.invalidate_predictions(self.controller)
        self.assertEqual(dict(self.items.model.predictions), {})
        self.plot(AUTO_ALIGNMENT_CONFIDENCE_KEY)
        self.assertEqual(self.align.call_count, 2)

    def test_nothing_stored_on_shank(self):
        """Apart from the cached views, nothing about the alignment is kept on the shank"""
        self.plot(AUTO_ALIGNMENT_KEY)
        self.assertEqual(set(vars(self.items.model)), {'align_handle', 'predictions'})


class TestAutoAlignmentOptions(unittest.TestCase):
    def test_both_options_registered(self):
        """Both options are offered for the alignment model and use the same compute function"""
        options = {key: regions for key, regions, _ in cp._MODEL_OPTIONS}
        self.assertEqual(options['Encoding'], [AUTO_ALIGNMENT_KEY, AUTO_ALIGNMENT_CONFIDENCE_KEY])
        controller = SimpleNamespace(model=SimpleNamespace(brain_atlas=None))
        func_map = cp.ChannelPrediction(controller).func_map
        for option in OPTIONS:
            self.assertIs(func_map[option], cp.compute_spatial_encoder_predictions)


if __name__ == '__main__':
    unittest.main()
