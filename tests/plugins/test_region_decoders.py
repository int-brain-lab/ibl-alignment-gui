"""Tests for the Hugging Face region decoders of the Channel Prediction plugin.

They load the released decoders (``ephysatlas.load_pretrained``, tag ``MODEL_REVISION``) and run
them on a small features fixture, so they are skipped when ephysatlas or torch are not installed.
"""

import importlib.util
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from ibl_alignment_gui.plugins import channel_prediction as cp
from ibl_alignment_gui.plugins.ephys_atlas import inference
from ibl_alignment_gui.plugins.ephys_atlas._common import PredictionError
from iblutil.util import Bunch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

# Per-channel features of one recorded probe (96 channels, every 4th of a Neuropixels 1.0 probe),
# reduced to the columns the 2026_W39 decoders need plus ``axial_um``, indexed (pid, channel).
FEATURES_FILE = Path(__file__).parents[1].joinpath('fixtures', 'region_decoders', 'features.pqt')

TRANSFORMER = 'ProbeTransformer region prediction'

HAS_DEPS = all(importlib.util.find_spec(m) for m in ('ephysatlas', 'torch', 'xgboost'))


def _fake_controller(models: dict) -> SimpleNamespace:
    """A controller holding only the plugin state the decoders read, with models pre-loaded."""
    state = Bunch({name: dict(model=model, source=None) for name, model in models.items()})
    return SimpleNamespace(plugins={'Channel Prediction': state}, view=None)


def _fake_shank(df: pd.DataFrame) -> SimpleNamespace:
    """A shank whose features come from ``df`` (no ONE loader, so they are used as is)."""
    model = SimpleNamespace(raw_data={'features': Bunch(df=df, exists=True)}, loaders={})
    return SimpleNamespace(model=model)


class TestRunDecoder(unittest.TestCase):
    """run_decoder with a stand-in model: no GUI and no ephysatlas needed."""

    @staticmethod
    def _model(features: list[str], predict) -> SimpleNamespace:
        return SimpleNamespace(
            inputs={'features': features}, index={'vintage': 'V1'}, predict=predict
        )

    def test_flat_table(self):
        """A flat table gets the (pid, channel) index and 'outside', and depths are returned"""
        df = pd.DataFrame({'a': [1.0, 2.0], 'axial_um': [20.0, 40.0]})
        seen = {}

        def predict(x):
            seen['df'] = x
            return pd.DataFrame({'predicted_atlas_id': [1, 2]}, index=x.index)

        out, depths = inference.run_decoder(self._model(['a', 'outside'], predict), df)
        self.assertEqual(list(seen['df'].index.names), ['pid', 'channel'])
        self.assertTrue((seen['df']['outside'] == 0).all())
        np.testing.assert_array_equal(out['predicted_atlas_id'], [1, 2])
        np.testing.assert_array_equal(depths, [20.0, 40.0])

    def test_dropped_channels(self):
        """Depths are those of the channels the model predicted"""
        df = pd.DataFrame({'a': [1.0, np.nan, 3.0], 'axial_um': [20.0, 40.0, 60.0]})

        def predict(x):
            kept = x.dropna()
            return pd.DataFrame({'predicted_atlas_id': [1] * len(kept)}, index=kept.index)

        _, depths = inference.run_decoder(self._model(['a'], predict), df)
        np.testing.assert_array_equal(depths, [20.0, 60.0])

    def test_missing_feature(self):
        """A model KeyError becomes a PredictionError naming the vintage"""

        def predict(x):
            raise KeyError("missing ['b']")

        df = pd.DataFrame({'a': [1.0], 'axial_um': [0.0]})
        with self.assertRaisesRegex(PredictionError, 'V1'):
            inference.run_decoder(self._model(['a', 'b'], predict), df)


@unittest.skipUnless(HAS_DEPS, 'needs ephysatlas, torch and xgboost')
class TestRegionDecoders(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Imported here so the module still imports (and skips) without ephysatlas installed.
        from ephysatlas import load_pretrained  # noqa: PLC0415

        cls.models = {
            name: load_pretrained(decoder.repo, revision=inference.MODEL_REVISION)
            for name, decoder in inference.DECODERS.items()
        }
        cls.controller = _fake_controller(cls.models)
        cls.df = pd.read_parquet(FEATURES_FILE)

    def test_predict_each_decoder(self):
        """Every channel gets a region of the model's classes, with its depth."""
        for name, model in self.models.items():
            with self.subTest(name=name):
                out, depths, _ = inference._predict_df(self.controller, _fake_shank(self.df), name)
                self.assertEqual(len(out), len(self.df))
                np.testing.assert_array_equal(depths, self.df['axial_um'])
                self.assertTrue(np.isin(out['predicted_atlas_id'], model.config['classes']).all())

    def test_transformer_flat_index_matches_indexed(self):
        """A flat features table is predicted as one probe, as the (pid, channel)-indexed one."""
        # pid / channel as columns and a plain row index, like a flat features file.
        df_flat = self.df.reset_index()
        out_flat, depths_flat, _ = inference._predict_df(
            self.controller, _fake_shank(df_flat), TRANSFORMER
        )
        out_indexed, depths_indexed, _ = inference._predict_df(
            self.controller, _fake_shank(self.df), TRANSFORMER
        )
        np.testing.assert_array_equal(depths_flat, depths_indexed)
        np.testing.assert_array_equal(
            out_flat['predicted_atlas_id'], out_indexed['predicted_atlas_id']
        )

    def test_predict_cumulative(self):
        """Both views from one run; cumulative probabilities end at 1, one colour per class."""
        for name, model in self.models.items():
            with self.subTest(name=name):
                views = inference.decode_shank(self.controller, _fake_shank(self.df), name)
                self.assertEqual(set(views), {name, name + inference.CUMULATIVE_SUFFIX})
                view = views[name + inference.CUMULATIVE_SUFFIX]
                n_classes = len(model.config['classes'])
                self.assertEqual(view.probability.shape, (len(self.df), n_classes))
                np.testing.assert_allclose(view.probability[:, -1], 1.0, atol=1e-4)
                self.assertEqual(view.depths.size, len(self.df))
                self.assertEqual(len(view.colours), n_classes)
                np.testing.assert_array_equal(view.regions, model.config['classes'])

    def test_missing_feature_raises(self):
        """A features table without a required column raises PredictionError."""
        for name in self.models:
            with self.subTest(name=name):
                df = self.df.drop(columns=['rms_ap'])
                with self.assertRaises(PredictionError):
                    inference.decode_shank(self.controller, _fake_shank(df), name)


RED, BLUE, GREEN = [255, 0, 0], [0, 0, 255], [0, 255, 0]
DECODER = 'XGBoost region prediction'


class TestDecoderViews(unittest.TestCase):
    """The decoder views and decode_shank, with a stand-in model output (no ephysatlas)."""

    def setUp(self):
        table = {1: ('A', RED), 2: ('B', BLUE), 3: ('C', GREEN)}

        def get(ids):
            ids = np.asarray(ids)
            return Bunch(
                id=ids,
                acronym=np.array([table[i][0] for i in ids]),
                rgb=np.array([table[i][1] for i in ids]),
            )

        # The atlas regions of the classes come from ephysatlas: stand in for them.
        self.enterContext(
            patch.object(inference, 'classifier_regions', lambda: SimpleNamespace(get=get))
        )
        self.out = pd.DataFrame(
            {
                'predicted_atlas_id': [1, 1, 2],
                'p_A': [0.7, 0.6, 0.1],
                'p_B': [0.2, 0.3, 0.8],
                'p_C': [0.1, 0.1, 0.1],
            }
        )
        self.depths = np.array([0.0, 10.0, 20.0])
        self.model = SimpleNamespace(config={'classes': [1, 2, 3]})
        self.predict_df = self.enterContext(
            patch.object(
                inference,
                '_predict_df',
                MagicMock(return_value=(self.out, self.depths, self.model)),
            )
        )

    def test_predict(self):
        """The most probable region per channel, as region segments in microns"""
        view = inference.predict(self.out, self.depths)
        np.testing.assert_allclose(view.region, [[0, 10], [20, 20]])
        self.assertEqual(view.axis_label[:, 1].tolist(), ['A', 'B'])
        np.testing.assert_array_equal(view.colour, [RED, BLUE])

    def test_predict_cumulative(self):
        """Cumulative probabilities ending at 1, with each class's region and colour"""
        view = inference.predict_cumulative(self.out, self.depths, self.model)
        np.testing.assert_allclose(view.probability[:, -1], 1.0)
        np.testing.assert_allclose(view.probability[0], [0.7, 0.9, 1.0])
        np.testing.assert_array_equal(view.regions, [1, 2, 3])
        np.testing.assert_array_equal(view.colours, [RED, BLUE, GREEN])
        np.testing.assert_array_equal(view.depths, self.depths)

    def test_decode_shank_both_views_one_run(self):
        """decode_shank runs the model once and returns both views under their options"""
        views = inference.decode_shank(None, None, DECODER)
        self.assertEqual(set(views), {DECODER, DECODER + inference.CUMULATIVE_SUFFIX})
        self.predict_df.assert_called_once()

    def test_decode_shank_cancelled(self):
        """A cancelled model load gives no views"""
        self.predict_df.return_value = None
        self.assertIsNone(inference.decode_shank(None, None, DECODER))

    def test_both_options_one_run_per_shank(self):
        """Through the plugin, the second option of a decoder is drawn without running it again"""
        items = SimpleNamespace(model=SimpleNamespace(predictions=None), view=MagicMock())
        controller = SimpleNamespace(
            all_shanks=['a'],
            shank_items={'a': {'default': items}},
            model=SimpleNamespace(
                brain_atlas=None,
                configs=['default'],
                get_current_shank=lambda s, c: SimpleNamespace(align_exists=True),
            ),
        )
        func_map = cp.ChannelPrediction(controller).func_map
        cumulative = DECODER + inference.CUMULATIVE_SUFFIX
        self.assertIs(func_map[DECODER], func_map[cumulative])
        cp._plot_region_panels(controller, DECODER, func_map[DECODER])
        cp._plot_region_panels(controller, cumulative, func_map[cumulative])
        self.predict_df.assert_called_once()
        items.view.plot_histology.assert_called_once()
        items.view.plot_histology_cumulative.assert_called_once()


if __name__ == '__main__':
    unittest.main()
