"""Tests for the helpers shared by the Channel Prediction plugin and its model backends."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from qtpy import QtWidgets

from ibl_alignment_gui.loaders.data_loader import FeatureLoaderLocal, FeatureLoaderOne
from ibl_alignment_gui.plugins.ephys_atlas._common import (
    PLUGIN_KEY,
    PredictionError,
    _get_features_df,
    features_for_vintage,
    features_vintage,
    get_region_boundaries,
    missing_features_error,
    model_vintage,
)
from iblutil.util import Bunch

RED, BLUE = [255, 0, 0], [0, 0, 255]


class TestGetRegionBoundaries(unittest.TestCase):
    def test_regions_along_track(self):
        """One segment per run of the same region, with its label and colour, in microns"""
        regions = Bunch(
            id=np.array([1, 1, 2, 2, 2]),
            acronym=np.array(['A', 'A', 'B', 'B', 'B']),
            rgb=np.array([RED] * 2 + [BLUE] * 3),
        )
        depths_m = np.array([0.0, 10.0, 20.0, 30.0, 40.0]) / 1e6
        view = get_region_boundaries(regions, depths_m)
        np.testing.assert_allclose(view.region, [[0, 10], [20, 40]])
        # Depths go from metres back to microns, so positions are compared with a tolerance.
        np.testing.assert_allclose(view.axis_label[:, 0].astype(float), [5, 30])
        self.assertEqual(view.axis_label[:, 1].tolist(), ['A', 'B'])
        np.testing.assert_array_equal(view.colour, [RED, BLUE])

    def test_single_region(self):
        """A track within one region is one segment"""
        regions = Bunch(
            id=np.array([3, 3, 3]), acronym=np.array(['C'] * 3), rgb=np.array([BLUE] * 3)
        )
        view = get_region_boundaries(regions, np.array([0.0, 10.0, 20.0]) / 1e6)
        np.testing.assert_allclose(view.region, [[0, 20]])
        self.assertEqual(view.axis_label[:, 1].tolist(), ['C'])


class TestFeaturesForVintage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_vintage_load_failure(self):
        """A vintage table that cannot be loaded raises PredictionError and is not cached"""
        # An IBL-mode shank whose GUI features are another vintage than the model's.
        loader = FeatureLoaderOne.__new__(FeatureLoaderOne)
        loader.feature_version = '2026_W26'
        loader.pid = 'pid'
        loader.one = None
        loader.multi_area = False
        items = SimpleNamespace(
            name='probe00',
            config='default',
            model=SimpleNamespace(loaders={'features': loader}, raw_data={}),
        )
        controller = SimpleNamespace(plugins={PLUGIN_KEY: Bunch()})
        error = FileNotFoundError('[Errno 2] No such file or directory: df_all_cols_merged.pqt')
        with (
            patch.object(FeatureLoaderOne, 'load_features', side_effect=error),
            self.assertRaisesRegex(
                PredictionError, r'2026_W39 features could not be loaded(.|\n)*No such file'
            ),
        ):
            features_for_vintage(controller, items, '2026_W39')
        self.assertEqual(items.model.raw_data['model_features'], {})
        self.assertIsNone(QtWidgets.QApplication.overrideCursor())

    def test_vintage_without_features_refused(self):
        """A vintage with no features is refused, suggesting the session's vintage; not reloaded"""
        loader = FeatureLoaderOne.__new__(FeatureLoaderOne)
        loader.feature_version = '2026_W26'
        loader.pid = 'pid'
        loader.one = None
        loader.multi_area = True
        items = SimpleNamespace(
            name='probe00',
            config='default',
            model=SimpleNamespace(loaders={'features': loader}, raw_data={}),
        )
        controller = SimpleNamespace(plugins={PLUGIN_KEY: Bunch()})
        with patch.object(
            FeatureLoaderOne, 'load_features', return_value=Bunch(exists=False)
        ) as load:
            for _ in range(2):
                with self.assertRaises(PredictionError) as raised:
                    features_for_vintage(controller, items, '2026_W39')
        load.assert_called_once()
        message = str(raised.exception)
        self.assertIn('No 2026_W39 features are available', message)
        self.assertIn(
            'load a 2026_W26 release of the model (Plugins -> Channel Prediction -> Load ... '
            'model, release tag 2026_W26)',
            message,
        )

    def test_progress_only_when_loading(self):
        """Loading the vintage table is reported; the cached table and GUI features are not"""
        loader = FeatureLoaderOne.__new__(FeatureLoaderOne)
        loader.feature_version = '2026_W26'
        loader.pid = 'pid'
        loader.one = None
        loader.multi_area = False
        gui_features = pd.DataFrame({'a': [1.0]})
        raw_data = {'features': Bunch(exists=True, df=gui_features)}
        items = SimpleNamespace(
            name='probe00',
            config='default',
            model=SimpleNamespace(loaders={'features': loader}, raw_data=raw_data),
        )
        controller = SimpleNamespace(plugins={PLUGIN_KEY: Bunch()})
        reports = []

        def progress(fraction, message):
            reports.append((fraction, message))

        table = Bunch(exists=True, df='W39 features')
        with patch.object(FeatureLoaderOne, 'load_features', return_value=table):
            for _ in range(2):  # loaded, then cached
                self.assertEqual(
                    features_for_vintage(controller, items, '2026_W39', progress), 'W39 features'
                )
            # The GUI's own vintage needs no loading.
            self.assertIs(
                features_for_vintage(controller, items, '2026_W26', progress), gui_features
            )
        self.assertEqual(reports, [(None, 'Loading the 2026_W39 features for probe00…')])

    @staticmethod
    def _local_features(attrs):
        """A shank whose features come from a local file, with the file's ``attrs``."""
        df = pd.DataFrame({'rms_lf': [1.0, 2.0]})
        df.attrs.update(attrs)
        items = SimpleNamespace(
            name='probe00',
            config='default',
            model=SimpleNamespace(loaders={}, raw_data={'features': Bunch(exists=True, df=df)}),
        )
        controller = SimpleNamespace(plugins={PLUGIN_KEY: Bunch()})
        return controller, items, df

    def test_other_features_version_warning_queued_each_call(self):
        """Local features of another version than the model's queue a warning each time used"""
        controller, items, df = self._local_features({'features_version': '2026_W42'})
        with patch.object(QtWidgets.QMessageBox, 'warning') as warning:
            for _ in range(2):  # e.g. two models trained on the same vintage
                self.assertIs(features_for_vintage(controller, items, '2026_W39'), df)
        warning.assert_not_called()  # this may run on a background thread: no Qt here
        messages = controller.plugins[PLUGIN_KEY]['pending_warnings']
        self.assertEqual(len(messages), 2)
        for message in messages:
            self.assertIn('2026_W42', message)
            self.assertIn('2026_W39', message)

    def test_same_or_unknown_features_version_no_warning(self):
        """Local features of the model's version, or without a version, queue nothing"""
        for attrs in ({'features_version': '2026_W39'}, {}):
            with self.subTest(attrs=attrs):
                controller, items, df = self._local_features(attrs)
                self.assertIs(features_for_vintage(controller, items, '2026_W39'), df)
                self.assertNotIn('pending_warnings', controller.plugins[PLUGIN_KEY])


class TestVintageTable(unittest.TestCase):
    def test_online_features_use_the_vintage_table(self):
        """Features from Alyx are replaced by the model's vintage table, not the GUI's features"""
        loader = FeatureLoaderOne.__new__(FeatureLoaderOne)
        loader.feature_version = '2026_W42'
        loader.pid = 'pid'
        loader.one = None
        loader.multi_area = False
        items = SimpleNamespace(
            name='b',
            config='default',
            model=SimpleNamespace(
                loaders={'features': loader},
                raw_data={'features': Bunch(exists=True, df='GUI features')},
            ),
        )
        controller = SimpleNamespace(plugins={PLUGIN_KEY: Bunch()})
        table = Bunch(exists=True, df='W39 features')
        with patch.object(FeatureLoaderOne, 'load_features', return_value=table):
            self.assertEqual(features_for_vintage(controller, items, '2026_W39'), 'W39 features')


class TestGetFeaturesDf(unittest.TestCase):
    def test_the_shank_features(self):
        """The features are those the shank loaded with the session, or None without any"""
        controller = SimpleNamespace(plugins={PLUGIN_KEY: Bunch()})
        for features, expected in [
            (Bunch(exists=True, df='features'), 'features'),
            (Bunch(exists=False), None),
            (None, None),
        ]:
            with self.subTest(features=features):
                items = SimpleNamespace(model=SimpleNamespace(raw_data={'features': features}))
                self.assertEqual(_get_features_df(controller, items), expected)


class TestModelVintage(unittest.TestCase):
    def test_manifest_vintage(self):
        """The vintage is read from the release's manifest"""
        self.assertEqual(model_vintage({'vintage': '2026_W39'}), '2026_W39')

    def test_no_vintage_refused(self):
        """A release whose manifest names no vintage is refused"""
        for index in (None, {}, {'vintage': ''}):
            with self.subTest(index=index), self.assertRaisesRegex(PredictionError, 'vintage'):
                model_vintage(index)


class TestMissingFeaturesError(unittest.TestCase):
    def test_message(self):
        """The error names the missing columns and the vintage"""
        message = str(missing_features_error(['a', 'b'], '2026_W39'))
        self.assertIn("missing required column(s): ['a', 'b']", message)
        self.assertIn('features computed with the 2026_W39 feature set', message)

    def test_suggests_features_vintage(self):
        """With the features' vintage known, it suggests loading that release of the model"""
        message = str(missing_features_error(['a'], '2026_W39', '2026_W26'))
        self.assertIn(
            'Alternatively, load a 2026_W26 release of the model (Plugins -> Channel Prediction '
            '-> Load ... model, release tag 2026_W26) to match your features.',
            message,
        )

    def test_generic_suggestion(self):
        """Without the features' vintage (or with the model's own), it suggests a matching one"""
        for data_vintage in (None, '2026_W39'):
            with self.subTest(data_vintage=data_vintage):
                message = str(missing_features_error(['a'], '2026_W39', data_vintage))
                self.assertIn(
                    'Alternatively, load a release of the model whose vintage matches your '
                    'features',
                    message,
                )

    def test_features_vintage_from_file(self):
        """A file's features_version attribute is kept in the parquet file and read back"""
        df = pd.DataFrame({'channel': [0], 'a': [1.0]})
        self.assertIsNone(features_vintage(df))
        df.attrs['features_version'] = '2026_W26'
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'features.pqt'
            df.to_parquet(path)
            read = FeatureLoaderLocal(path).load_features()['df']
        self.assertEqual(features_vintage(read), '2026_W26')


if __name__ == '__main__':
    unittest.main()
