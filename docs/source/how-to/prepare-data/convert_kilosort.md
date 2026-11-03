(howto-convert-kilosort)=
# How to convert SpikeGLX and kilosort output

This guide shows you how to create the spike sorting, RMS and power spectrum files the GUI reads,
from a SpikeGLX recording sorted with kilosort, pykilosort or iblsorter.

## Prerequisites

- The `ibl` extra installed (see {ref}`howto-install-extras`).
- The spike sorting output folder from kilosort, pykilosort or iblsorter, and the folder with the
  SpikeGLX `.ap.bin` (or `.ap.cbin`) and `.meta` files. Neuropixels 1.0 recordings also need the
  `.lf.bin`; Neuropixels 2.0 probes record the LF band in the AP file, so it is computed from
  there.

## Steps

1. Choose an output folder that is **not** the spike sorting output folder, so that none of the
   spike sorter's files are overwritten.

2. Run:

   ```python
   import logging
   from pathlib import Path

   from ibl_alignment_gui.convertors.extract_ephys import extract_data

   # Progress is reported through the logger
   logging.basicConfig(level=logging.INFO)

   ks_path = Path('/path/to/spike/sorting/output')
   ephys_path = Path('/path/to/raw/ephys/data')
   out_path = Path('/path/to/output')

   extract_data(ks_path, ephys_path, out_path)
   ```

   The RMS and power spectra are computed on samples taken through the recording, in parallel
   threads, so an hour-long recording takes a few minutes. To read the whole recording instead,
   or to use fewer threads on a shared machine, compute them with `extract_rmsmap` after
   `extract_data`:

   ```python
   from ibl_alignment_gui.convertors.extract_ephys import extract_rmsmap

   ap_file = ephys_path / 'my_recording_g0_t0.imec0.ap.bin'
   extract_rmsmap(ap_file, out_folder=out_path, spectra=False, sample_spacing_secs=None, n_workers=2)
   ```

3. Check that the output folder holds `spikes.*`, `clusters.*` and `channels.*` files, and the
   `_iblqc_ephysTimeRms*` and `_iblqc_ephysSpectralDensityLF.*` files, each with a `rawInd`
   file.

4. In the session YAML, use `out_path` as both the `spike_sorting` and the `processed_ephys`
   folder, and `ephys_path` as `raw_ephys` (see {ref}`howto-single-probe-yaml`).

## Files made with an older version

Older versions of `extract_ephys` saved the RMS and power spectra with the channels sorted by
their position on the probe, and without the `rawInd` files that record the order. Recompute them
with the steps above when:

- the GUI warns that the channels of these files *may be in the wrong order*. It does so when the
  raw file does not store the channels in shank and depth order, as on Neuropixels 2.0 four shank
  probes, where the plots then show the wrong channels;
- the probe is a Neuropixels 2.0. Even without the warning, the left and right columns of sites
  may be swapped in the probe plots.

## Related

- {ref}`ref-input-datasets`: the contents of each file
- {ref}`howto-convert-spikeinterface`: for other recording systems or sorters
