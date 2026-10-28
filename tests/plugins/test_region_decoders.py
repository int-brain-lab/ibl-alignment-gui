"""Tests for the Hugging Face region decoders of the Channel Prediction plugin.

They load the released decoders (``ephysatlas.load_pretrained``, each at its default release in
``ephysatlas.models.DECODER_VINTAGES``) and run them on a small features fixture, so they are
skipped when ephysatlas or torch are not installed.
"""

import importlib.util
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from ibl_alignment_gui.plugins.ephys_atlas import inference
from ibl_alignment_gui.plugins.ephys_atlas._common import PredictionError
from iblutil.util import Bunch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

# Per-channel features of one recorded probe (96 channels, every 4th of a Neuropixels 1.0 probe),
# computed with ibleatools' compute_denoised_features (features version 2026_W42) and reduced to
# the columns the 2026_W42 decoders need plus ``axial_um``, indexed (pid, channel). Its attrs hold
# the features version, the snippet windows and the ibleatools version that computed them.
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
    return SimpleNamespace(name='probe00', config='default', model=model)


def _no_progress(fraction, message) -> None:
    """A progress callback that ignores the reports."""


class TestDecode(unittest.TestCase):
    """_decode with the features and run_decoder faked: the steps it reports."""

    def setUp(self):
        self.model = SimpleNamespace(index={'vintage': 'V1'})
        self.features = self.enterContext(
            patch.object(inference, 'features_for_vintage', return_value='features')
        )
        self.run_decoder = self.enterContext(
            patch.object(inference, 'run_decoder', return_value=('out', 'depths'))
        )
        self.reports = []

    def decode(self):
        progress = lambda fraction, message: self.reports.append(message)  # noqa: E731
        return inference._decode('controller', 'items', self.model, progress)

    def test_features_then_predict(self):
        """The features of the model's vintage are passed to the decoder; prediction is reported"""
        self.assertEqual(self.decode(), ('out', 'depths'))
        self.assertEqual(self.features.call_args.args[:3], ('controller', 'items', 'V1'))
        self.run_decoder.assert_called_once_with(self.model, 'features')
        self.assertEqual(self.reports, ['Predicting the regions…'])

    def test_no_features(self):
        """No features for the shank raises PredictionError before predicting"""
        self.features.return_value = None
        with self.assertRaisesRegex(PredictionError, 'No ephys-atlas features'):
            self.decode()
        self.run_decoder.assert_not_called()


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
        """A model KeyError for a missing feature becomes a PredictionError naming it"""

        def predict(x):
            raise KeyError("missing ['b']")

        df = pd.DataFrame({'a': [1.0], 'axial_um': [0.0]})
        with self.assertRaises(PredictionError) as raised:
            inference.run_decoder(self._model(['a', 'b'], predict), df)
        message = str(raised.exception)
        self.assertIn("missing required column(s): ['b']", message)
        self.assertIn('The model (V1)', message)

    def test_missing_feature_suggests_features_vintage(self):
        """With the table's vintage known, the error suggests that release of the model"""

        def predict(x):
            raise KeyError("missing ['b']")

        df = pd.DataFrame({'a': [1.0], 'axial_um': [0.0]})
        df.attrs['features_version'] = 'V0'
        with self.assertRaisesRegex(PredictionError, 'load a V0 release of the model'):
            inference.run_decoder(self._model(['a', 'b'], predict), df)

    def test_other_key_error_raised(self):
        """A model KeyError that is not about a missing feature is raised as is"""

        def predict(x):
            raise KeyError('bug')

        df = pd.DataFrame({'a': [1.0], 'axial_um': [0.0]})
        with self.assertRaisesRegex(KeyError, 'bug'):
            inference.run_decoder(self._model(['a'], predict), df)


@unittest.skipUnless(HAS_DEPS, 'needs ephysatlas, torch and xgboost')
class TestRegionDecoders(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Imported here so the module still imports (and skips) without ephysatlas installed.
        from ephysatlas import load_pretrained  # noqa: PLC0415
        from ephysatlas.models import DECODER_VINTAGES  # noqa: PLC0415

        cls.models = {
            name: load_pretrained(decoder.repo, revision=DECODER_VINTAGES[decoder.repo])
            for name, decoder in inference.DECODERS.items()
        }
        cls.controller = _fake_controller(cls.models)
        cls.df = pd.read_parquet(FEATURES_FILE)

    def test_predict_each_decoder(self):
        """Every channel gets a region of the model's classes, with its depth."""
        for name, model in self.models.items():
            with self.subTest(name=name):
                out, depths = inference._decode(
                    self.controller, _fake_shank(self.df), model, _no_progress
                )
                self.assertEqual(len(out), len(self.df))
                np.testing.assert_array_equal(depths, self.df['axial_um'])
                self.assertTrue(np.isin(out['predicted_atlas_id'], model.config['classes']).all())

    def test_transformer_flat_index_matches_indexed(self):
        """A flat features table is predicted as one probe, as the (pid, channel)-indexed one."""
        # pid / channel as columns and a plain row index, like a flat features file.
        df_flat = self.df.reset_index()
        model = self.models[TRANSFORMER]
        out_flat, depths_flat = inference._decode(
            self.controller, _fake_shank(df_flat), model, _no_progress
        )
        out_indexed, depths_indexed = inference._decode(
            self.controller, _fake_shank(self.df), model, _no_progress
        )
        np.testing.assert_array_equal(depths_flat, depths_indexed)
        np.testing.assert_array_equal(
            out_flat['predicted_atlas_id'], out_indexed['predicted_atlas_id']
        )

    def test_predict_cumulative(self):
        """Both views from one run; cumulative probabilities end at 1, one colour per class."""
        for name, model in self.models.items():
            with self.subTest(name=name):
                views, _ = inference.decode_shank(
                    self.controller, _fake_shank(self.df), model, _no_progress, name
                )
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
        for name, model in self.models.items():
            with self.subTest(name=name):
                df = self.df.drop(columns=['rms_ap'])
                with self.assertRaises(PredictionError):
                    inference.decode_shank(
                        self.controller, _fake_shank(df), model, _no_progress, name
                    )

    def test_features_version_warning(self):
        """Features of another version than a decoder's queue one warning; its own queue none."""
        other = self.df.copy()
        other.attrs['features_version'] = '2026_W39'
        for df, n_warnings in ((other, 1), (self.df, 0)):
            for name, model in self.models.items():
                version = df.attrs['features_version']
                with self.subTest(name=name, features_version=version):
                    # A fresh controller: the queued warnings live in the plugin state.
                    controller = _fake_controller(self.models)
                    out, _ = inference._decode(controller, _fake_shank(df), model, _no_progress)
                    self.assertEqual(len(out), len(df))  # still predicted
                    warnings = controller.plugins['Channel Prediction'].get('pending_warnings', [])
                    self.assertEqual(len(warnings), n_warnings)
                    for warning in warnings:
                        self.assertIn(version, warning)
                        self.assertIn(model.index['vintage'], warning)


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
        self.decode = self.enterContext(
            patch.object(inference, '_decode', MagicMock(return_value=(self.out, self.depths)))
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
        views, extra = inference.decode_shank(None, None, self.model, _no_progress, DECODER)
        self.assertEqual(set(views), {DECODER, DECODER + inference.CUMULATIVE_SUFFIX})
        self.assertIsNone(extra)
        self.decode.assert_called_once_with(None, None, self.model, _no_progress)


if __name__ == '__main__':
    unittest.main()
