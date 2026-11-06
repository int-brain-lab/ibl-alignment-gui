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


def write_shank_analyzers(folder: Path, n_shanks: int = 2) -> tuple[list[Path], np.ndarray]:
    """
    Save one SortingAnalyzer per shank of a recording on a multi-shank probe.

    The probe has 4 channels per shank, with the contact ids probeinterface gives Neuropixels
    probes ('s0e0' ...), and each analyzer holds the channels and units of one shank.

    Parameters
    ----------
    folder : Path
        The folder to save the analyzers to, as shank_0, shank_1 ...
    n_shanks : int
        The number of shanks.

    Returns
    -------
    list of Path
        The analyzer folder of each shank.
    np.ndarray
        The contact id of each channel of the whole recording, in channel order.
    """
    import spikeinterface as si  # noqa: PLC0415
    from probeinterface import generate_multi_shank  # noqa: PLC0415

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        recording, sorting = si.generate_ground_truth_recording(
            durations=[4.0], num_channels=4 * n_shanks, num_units=4 * n_shanks, seed=0
        )
        probe = generate_multi_shank(
            num_shank=n_shanks, shank_pitch=[250, 0], num_columns=1, num_contact_per_column=4
        )
        contact_ids = np.array([f's{shank}e{i}' for shank in range(n_shanks) for i in range(4)])
        probe.set_contact_ids(contact_ids)
        probe.set_device_channel_indices(np.arange(4 * n_shanks))
        recording.set_probe(probe, group_mode='by_shank')

        paths = []
        for group, shank_recording in sorted(recording.split_by('group').items()):
            path = folder.joinpath(f'shank_{group}')
            # The units sorted on the shank are those whose main channel is on it
            on_shank = np.isin(
                sorting.get_property('main_channel_id'), shank_recording.channel_ids
            )
            shank_sorting = sorting.select_units(sorting.unit_ids[on_shank])
            analyzer = si.create_sorting_analyzer(
                shank_sorting, shank_recording, sparse=False, verbose=False
            )
            analyzer.compute(['random_spikes', 'templates', 'spike_amplitudes'], verbose=False)
            analyzer.save_as(format='binary_folder', folder=path)
            paths.append(path)

    return paths, contact_ids
