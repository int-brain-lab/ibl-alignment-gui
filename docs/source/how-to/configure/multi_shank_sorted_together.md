(howto-multi-shank-sorted-together)=
# How to load shanks that were sorted together

This guide shows you how to load a probe whose shanks were spike sorted together, so that one spike
sorting output holds every shank.

## Prerequisites

- One traced trajectory per shank.

## Steps

1. Name the trajectory files with the shank number, counting from 1, and put them in the `picks`
   folder:

   ```text
   xyz_picks_shank1.json
   xyz_picks_shank2.json
   xyz_picks_shank3.json
   xyz_picks_shank4.json
   ```

2. Give the probe a single entry in the session YAML, as for a single-shank probe:

   ```yaml
   path: /path/to/session

   probes:
     probe00:
       datasets:
         spike_sorting:
           path: probe00/kilosort
         raw_ephys:
           path: probe00/spikeglx
         picks:
           path: probe00/picks
         output:
           path: probe00/alignment
   ```

3. Open the session. The GUI finds the shanks from the probe geometry and shows them side by side
   in a grid.

The results are written per shank, as `channel_locations_shank1.json` and so on, so all the shanks
can share one `output` folder.

## Related

- {ref}`howto-shanks-sorted-separately`: when each shank was sorted on its own
- {ref}`ref-probe-geometry`: how the shanks are found
