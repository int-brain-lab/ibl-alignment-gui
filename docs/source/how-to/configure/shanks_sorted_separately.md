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

## SpikeInterface analyzers

Shanks sorted separately with SpikeInterface, one `SortingAnalyzer` per shank, are set up the
same way: each analyzer is the `spike_sorting` of its own probe entry. The RMS, power spectra and
snippets can come from one run of `extract_si_qc` on the whole probe's recording, shared by every
entry, or from one run per shank:

```yaml
probes:
  shank_0:
    datasets:
      spike_sorting:
        path: sorting/shank_0_analyzer
      processed_ephys:
        path: extracted          # one folder for the whole probe
      picks:
        path: shank_0/picks
      output:
        path: shank_0/output

  shank_1:
    datasets:
      spike_sorting:
        path: sorting/shank_1_analyzer
      processed_ephys:
        path: extracted
      picks:
        path: shank_1/picks
      output:
        path: shank_1/output
```

Each shank finds its own channels in the shared files by their probe contact ids (see
{ref}`howto-convert-spikeinterface`). This needs the analyzers to have the contact ids that
probeinterface gives Neuropixels probes, which analyzers made from a SpikeInterface reader of
the raw data have. An analyzer without them, and without SpikeGLX channel ids, would show the
first channels of the probe instead of its own. For such analyzers, run `extract_si_qc` once per
shank on that shank's recording, and give each entry its own `processed_ephys` folder:

```python
ap_shanks = ap.split_by('group')
lf_shanks = lf.split_by('group')
for group, ap_shank in ap_shanks.items():
    extract_si_qc(ap_shank, f'extracted/shank_{group}', lf_recording=lf_shanks[group])
```

A shank whose analyzer holds a contact that is not in the extracted files does not load: the
files were written from another recording.

Analyzers of the whole probe, including those sorted shank by shank and then aggregated, as
SpikeInterface's `run_sorter_by_property` does, are loaded as one probe (see
{ref}`howto-multi-shank-sorted-together`).

## Related

- {ref}`howto-multi-shank-sorted-together`
- {ref}`explanation-shanks-probes-configurations`
