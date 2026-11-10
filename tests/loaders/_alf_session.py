"""A small ALF session written to disk, for the loader tests."""

from pathlib import Path

import numpy as np
import pandas as pd

import one.alf.io as alfio
from ibl_alignment_gui.utils.parse_yaml import DatasetPaths


def write_session(root: Path) -> DatasetPaths:
    """Write a small ALF session to disk and return its dataset paths.

    Four channels, three clusters on channels 0, 2 and 1, and six spikes, two per cluster.
    Cluster 2 fires below the minimum rate of DataLoader.filter_spikes_by_fr (50 per hour).
    """
    paths = DatasetPaths(
        spike_sorting=root.joinpath('alf', 'probe00'),
        processed_ephys=root.joinpath('raw_ephys_data', 'probe00'),
        task=root.joinpath('alf', 'task'),
        raw_task=root.joinpath('raw_passive_data'),
    )
    for folder in (paths.spike_sorting, paths.processed_ephys, paths.task, paths.raw_task):
        folder.mkdir(parents=True)

    alfio.save_object_npy(
        paths.spike_sorting,
        {
            'clusters': np.array([0, 1, 2, 0, 1, 2], dtype=np.int32),
            'times': np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            'depths': np.array([10, 20, 30, 40, 50, 60], dtype=np.float32),
            'amps': np.arange(1, 7, dtype=np.float32) * 1e-6,
        },
        'spikes',
    )
    alfio.save_object_npy(
        paths.spike_sorting,
        {
            'channels': np.array([0, 2, 1]),
            'peakToTrough': np.array([0.1, 0.2, 0.3]),
            'waveforms': np.arange(3 * 5 * 4, dtype=np.float32).reshape(3, 5, 4),
        },
        'clusters',
    )
    pd.DataFrame({'firing_rate': [1.0, 1.0, 0.001], 'label': [1.0, 0.0, 1.0]}).to_parquet(
        paths.spike_sorting.joinpath('clusters.metrics.pqt')
    )
    alfio.save_object_npy(
        paths.spike_sorting,
        {
            'rawInd': np.arange(4),
            'localCoordinates': np.c_[np.zeros(4), np.arange(4) * 20.0],
        },
        'channels',
    )

    alfio.save_object_npy(
        paths.processed_ephys,
        {'rms': np.arange(12.0).reshape(3, 4), 'timestamps': np.array([1.5, 4.5, 7.5])},
        'ephysTimeRmsAP',
        namespace='iblqc',
    )
    # The LF RMS uses the older attribute name and has no timestamps
    alfio.save_object_npy(
        paths.processed_ephys,
        {'amps': np.arange(12.0, 24.0).reshape(3, 4)},
        'ephysTimeRmsLF',
        namespace='iblqc',
    )
    alfio.save_object_npy(
        paths.processed_ephys,
        {'power': np.arange(20.0).reshape(5, 4), 'freqs': np.arange(5.0)},
        'ephysSpectralDensityLF',
        namespace='iblqc',
    )

    alfio.save_object_npy(paths.task, {'times': np.array([0.0, 1.0])}, 'passiveRFM')
    alfio.save_object_npy(paths.task, {'valveOn': np.array([2.0, 3.0])}, 'passiveStims')
    # Real sessions store these as a .table.csv, but ONE loads tables with a pandas call that
    # now warns, and warnings are errors in the tests
    alfio.save_object_npy(
        paths.task,
        {
            'start': np.array([1.0, 2.0, 3.0]),
            'position': np.array([35, -35, 35]),
            'contrast': np.array([1.0, 1.0, 0.0]),
        },
        'passiveGabor',
    )
    np.arange(15 * 15 * 2, dtype=np.uint8).tofile(
        paths.raw_task.joinpath('_iblrig_RFMapStim.raw.bin')
    )
    return paths
