"""Tests for the Hugging Face region decoders of the Channel Prediction plugin.

They load the released decoders (``ephysatlas.load_pretrained``, tag ``MODEL_REVISION``) and run
them on a small features fixture, so they are skipped when ephysatlas or torch are not installed.
"""

import importlib.util
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from ibl_alignment_gui.plugins.ephys_atlas import inference
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
                regions, depths = inference.predict(self.controller, _fake_shank(self.df), name)
                self.assertEqual(regions.id.size, len(self.df))
                np.testing.assert_array_equal(depths, self.df['axial_um'])
                self.assertTrue(np.isin(regions.id, model.config['classes']).all())

    def test_transformer_flat_index_matches_indexed(self):
        """A flat features table is predicted as one probe, as the (pid, channel)-indexed one."""
        # pid / channel as columns and a plain row index, like a flat features file.
        df_flat = self.df.reset_index()
        regions_flat, depths_flat = inference.predict(
            self.controller, _fake_shank(df_flat), TRANSFORMER
        )
        regions_indexed, depths_indexed = inference.predict(
            self.controller, _fake_shank(self.df), TRANSFORMER
        )
        np.testing.assert_array_equal(depths_flat, depths_indexed)
        np.testing.assert_array_equal(regions_flat.id, regions_indexed.id)

    def test_predict_cumulative(self):
        """Cumulative probabilities end at 1, with one region id and colour per class."""
        for name, model in self.models.items():
            with self.subTest(name=name):
                cprobas, depths, colours, region_ids = inference.predict_cumulative(
                    self.controller, _fake_shank(self.df), name
                )
                n_classes = len(model.config['classes'])
                self.assertEqual(cprobas.shape, (len(self.df), n_classes))
                np.testing.assert_allclose(cprobas[:, -1], 1.0, atol=1e-4)
                self.assertEqual(depths.size, len(self.df))
                self.assertEqual(len(colours), n_classes)
                np.testing.assert_array_equal(region_ids, model.config['classes'])

    def test_missing_feature_warns_and_returns_none(self):
        """A features table without a required column shows a warning instead of raising."""
        for name in self.models:
            with self.subTest(name=name):
                df = self.df.drop(columns=['rms_ap'])
                with patch.object(inference.QtWidgets.QMessageBox, 'warning') as warning:
                    result = inference.predict(self.controller, _fake_shank(df), name)
                self.assertIsNone(result)
                warning.assert_called_once()
