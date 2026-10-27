(howto-single-probe-yaml)=
# How to write a session YAML for one probe

This guide shows you how to list the dataset folders of a single-shank probe, when they are not all
in one folder.

## Steps

1. Create a `.yaml` file. Give a top-level `path` for the folder most datasets sit under, so the
   dataset paths can be relative:

   ```yaml
   path: /mnt/data/subject01/2025-01-15

   probes:
     probe00:
       datasets:
         spike_sorting:
           path: probe00/kilosort
         processed_ephys:
           path: probe00/kilosort
         raw_ephys:
           path: probe00/spikeglx
         picks:
           path: probe00/picks
         output:
           path: probe00/alignment
   ```

2. Add any optional datasets you have: `histology` or `features` (see {ref}`ref-session-yaml`).
   `picks` and `output` must always be given. `processed_ephys` can be left out if the RMS and
   power spectrum files are in the `raw_ephys` folder.

3. Open the file (see {ref}`howto-open-session`) and check that each figure shows data.

## Related

- {ref}`ref-session-yaml`: every key, and which datasets are required
- {ref}`explanation-session-yaml-resolution`: how relative paths are resolved
