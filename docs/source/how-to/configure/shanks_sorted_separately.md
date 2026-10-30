(howto-shanks-sorted-separately)=
# How to load shanks that were sorted separately

This guide shows you how to load a multi-shank probe whose shanks were spike sorted one at a time,
so that each shank has its own spike sorting output.

## Steps

1. Give each shank its own probe entry in the session YAML, with its own folders. Each shank is
   then a single-shank probe to the GUI, so its trajectory is named `xyz_picks.json`, with no
   shank suffix.

2. Give every entry its own `output` folder. The results of each entry are written as
   `channel_locations.json` with no suffix, so entries sharing an `output` folder overwrite each
   other's results.

   ```yaml
   path: /path/to/session

   probes:
     shank_0:
       datasets:
         spike_sorting:
           path: shank_0/spike_sorting
         processed_ephys:
           path: shank_0/processed_ephys
         raw_ephys:
           path: shank_0/raw_ephys
         picks:
           path: shank_0/picks
         output:
           path: shank_0/output

     shank_1:
       datasets:
         spike_sorting:
           path: shank_1/spike_sorting
         processed_ephys:
           path: shank_1/processed_ephys
         raw_ephys:
           path: shank_1/raw_ephys
         picks:
           path: shank_1/picks
         output:
           path: shank_1/output

     # shank_2 and shank_3 follow the same pattern
   ```

   When every shank uses the same folder names, a probe `path` and `defaults` shorten this (see
   {ref}`howto-share-paths-with-defaults`).

3. Open the session. The shanks appear in the order they are listed in the file.

## Related

- {ref}`howto-multi-shank-sorted-together`
- {ref}`explanation-shanks-probes-configurations`
