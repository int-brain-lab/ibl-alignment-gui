"""A small SpikeInterface SortingAnalyzer of a generated recording, saved to disk for the tests."""

import warnings
from pathlib import Path

import numpy as np


def write_analyzer(folder: Path, extensions=None, n_shanks: int | None = None, **kwargs) -> Path:
    """
    Save a small SortingAnalyzer of a generated recording, with the given extensions computed.

    Parameters
    ----------
    folder : Path
        The folder to save the analyzer to.
    extensions : list of str or dict or None
        The extensions to compute, or a dict of extension names to their parameters. Defaults to
        the extensions SortingSourceSI needs.
    n_shanks : int or None
        When given, the recording is on a probe with this many shanks of 4 channels, 250 um
        apart. Otherwise it is on the generated probe, which sets no shanks.
    **kwargs
        Passed to generate_ground_truth_recording.

    Returns
    -------
    Path
        The analyzer folder.
    """
    import spikeinterface as si  # noqa: PLC0415
    from probeinterface import generate_multi_shank  # noqa: PLC0415

    extensions = extensions or ['random_spikes', 'templates', 'spike_amplitudes']
    kwargs = {'durations': [4.0], 'num_channels': 8, 'num_units': 4, 'seed': 0} | kwargs
    if n_shanks is not None:
        kwargs['num_channels'] = 4 * n_shanks
    # SpikeInterface warns that the generated recording cannot be saved with the analyzer
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        recording, sorting = si.generate_ground_truth_recording(**kwargs)
        if n_shanks is not None:
            probe = generate_multi_shank(
                num_shank=n_shanks, shank_pitch=[250, 0], num_columns=1, num_contact_per_column=4
            )
            probe.set_device_channel_indices(np.arange(4 * n_shanks))
            recording.set_probe(probe, in_place=True)
        analyzer = si.create_sorting_analyzer(sorting, recording, sparse=False, verbose=False)
        analyzer.compute(extensions, verbose=False)
        analyzer.save_as(format='binary_folder', folder=folder)

    return folder
