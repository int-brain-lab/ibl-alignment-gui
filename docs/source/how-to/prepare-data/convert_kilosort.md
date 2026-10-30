(howto-convert-kilosort)=
# How to convert SpikeGLX and kilosort output

This guide shows you how to create the spike sorting, RMS and power spectrum files the GUI reads,
from a SpikeGLX recording sorted with kilosort, pykilosort or iblsorter.

## Prerequisites

- The `ibl` extra installed (see {ref}`howto-install-extras`).
- The spike sorting output folder from kilosort, pykilosort or iblsorter, and the folder with the
  SpikeGLX `.ap.bin`, `.lf.bin` and `.meta` files.

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

3. Check that the output folder holds `spikes.*`, `clusters.*` and `channels.*` files, and the
   `_iblqc_ephysTimeRms*` and `_iblqc_ephysSpectralDensityLF.*` files.

4. In the session YAML, use `out_path` as both the `spike_sorting` and the `processed_ephys`
   folder, and `ephys_path` as `raw_ephys` (see {ref}`howto-single-probe-yaml`).

## Related

- {ref}`ref-input-datasets`: the contents of each file
- {ref}`howto-convert-spikeinterface`: for other recording systems or sorters
