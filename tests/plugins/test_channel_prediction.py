"""Tests for the Channel Prediction plugin's per-shank compute, cache and draw loop."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from ibl_alignment_gui.plugins import channel_prediction as cp
from ibl_alignment_gui.plugins.ephys_atlas._common import PredictionError
from iblutil.util import Bunch

SHANKS = ('a', 'b')


def _fake_controller() -> SimpleNamespace:
    """A controller with two aligned shanks, whose views record what they draw."""
    shank_items = {
        shank: {
            'default': SimpleNamespace(model=SimpleNamespace(predictions=None), view=MagicMock())
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


if __name__ == '__main__':
    unittest.main()
