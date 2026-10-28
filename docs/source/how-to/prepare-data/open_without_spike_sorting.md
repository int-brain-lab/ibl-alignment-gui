(howto-open-without-spike-sorting)=
# How to open a session that has not been spike sorted

This guide shows you how to look at the raw data, RMS and power spectra along the track before spike
sorting has been run, for example to check a recording or to start an alignment early.

## Prerequisites

- The SpikeGLX `.ap.meta` file. Without spike sorting it is the only source of the probe geometry,
  so the session cannot be opened without it.
- The `_iblqc_*` RMS and power spectrum files, which are what you align against without spike
  sorting.
- The traced trajectory, `xyz_picks.json`.
- Optionally, the `.ap.bin` and `.lf.bin` files, for snippets of the raw signal.

## Steps

1. Write a session YAML with no `spike_sorting` dataset. `raw_ephys` points at the folder with the
   SpikeGLX `.ap.meta` file, and the RMS and power spectrum files are read from the same folder
   unless you give a `processed_ephys` folder. Without spike sorting, `raw_ephys`, `picks` and
   `output` must all be given:

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

2. Load the session. The probe geometry is read from the `.ap.meta` file. The RMS and power
   spectrum plots are offered, and the raw data snippets when the `.bin` files are there. The plots
   that need spike sorting are not offered.

## Related

- {ref}`ref-probe-geometry`: how the geometry is found
- {ref}`ref-session-yaml`: which datasets are required
