(howto-open-without-spike-sorting)=
# How to open a session that has not been spike sorted

This guide shows you how to look at the raw data, RMS and power spectra along the track before spike
sorting has been run, for example to check a recording or to start an alignment early.

## Prerequisites

- The SpikeGLX `.ap.meta` file, and optionally the `.ap.bin` and `.lf.bin` files.
- The traced trajectory, `xyz_picks.json`.
- Optionally, the `_iblqc_*` RMS and power spectrum files.

## Steps

1. Write a session YAML with no `spike_sorting` dataset. `raw_ephys` points at the SpikeGLX files,
   and the RMS and power spectrum files are read from the same folder unless you give a
   `processed_ephys` folder:

   ```yaml
   probes:
     probe00:
       datasets:
         raw_ephys:
           path: /path/to/spikeglx
         picks:
           path: /path/to/picks
         output:
           path: /path/to/output
   ```

2. Load the session. The probe geometry is read from the `.ap.meta` file. The raw data, RMS and
   power spectrum plots are shown; the plots that need spike sorting are empty.

## Related

- {ref}`ref-probe-geometry`: how the geometry is found
- {ref}`ref-session-yaml`: which datasets are required
