(howto-give-shanks-features)=
# How to give each shank its features file

This guide shows you how to tell the GUI which features file to use for each shank, so the models
can run.

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

## From the menu

1. Open `Plugins → Channel Prediction → Load features file…`.
2. Choose the files:

   - With one shank, choose the file.
   - With several shanks or configurations, there is one row per shank and configuration. Change
     the rows that need a different file; rows left unchanged keep their current features.
   - For a probe whose shanks were sorted together, whose one file holds every shank's channels,
     tick **One file for all shanks (split by channel)** and choose that file.

3. Click **OK**. The files are checked before any is used; if one has no features for its shank,
   or cannot be read, it is reported and nothing changes.

A file chosen from the menu takes precedence over the YAML, but only until new data is loaded.

## Related

- {ref}`ref-channel-prediction`: where features are looked for, in order
