(howto-give-shanks-features)=
# How to give each shank its features file

This guide shows you how to tell the GUI which features file to use for each shank, so the models
can run. The file is given with the session: in the session YAML, or in the data folder.

## In the session YAML

Add the file as the `features` dataset of the probe. The path is to the file, not a folder, and is
resolved like any other dataset path:

```yaml
path: /path/to/session

probes:
  probe00:
    datasets:
      spike_sorting:
        path: probe00/spike_sorting
      picks:
        path: probe00/picks
      output:
        path: probe00/alignment
      features:
        path: probe00/features/raw_ephys_features.pqt
```

This is the best choice for a session you will come back to.

## In the data folder

When you open a data folder instead of a YAML, put the features file in that folder, with
`features` in its name, such as `ephys_atlas_features.pqt`. It is picked up when the folder is
opened. Keep only one such file in the folder: with several, the GUI cannot tell which to use, so
it uses none and logs the files it found.

## Multi-shank probes

- Shanks sorted together share one probe entry, and one features file holds all their channels.
  Give it once; the GUI splits it by channel for each shank.
- Shanks sorted separately each have their own probe entry, so give each its own `features`
  dataset.

The features are read when the session is loaded. After changing the file, load the session
again.

## Related

- {ref}`ref-channel-prediction`: where features are looked for, in order
