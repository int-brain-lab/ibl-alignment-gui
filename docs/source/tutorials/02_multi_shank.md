(tutorial-multi-shank)=
# Aligning a multi-shank probe

In this tutorial you will align a four-shank Neuropixels 2.0 probe, using neighbouring shanks to
check each other. You will load the data from a session YAML file.

:::{admonition} TODO
:class: warning

Screenshot of the grid layout with all four shanks of Sample data 2 aligned.
`_static/tutorials/02_multi_shank/result.png`
:::

It takes about 20 minutes.

## Before you start

- Complete {ref}`tutorial-first-alignment`. This tutorial assumes you can add reference line pairs
  and apply a fit.
- Extract the **Sample data 2** folder from the
  [sample data](https://ibl.flatironinstitute.org/public/ephys_alignment_sample_data.zip): a
  four-shank Neuropixels 2.0 (2.4) probe with 96 channels per shank.

## 1. Point the session YAML at your copy of the data

The datasets of Sample data 2 are spread over several folders, so they are listed in a session
YAML instead of being opened as one folder.

1. Open `sample_data_2.yaml`, in the Sample data 2 folder, in a text editor.
2. Change the `path` on the first line to the folder you extracted Sample data 2 into:

   ```yaml
   path: /home/user/Downloads/sample_data2      # <- change this

   probes:

     probe00a:
       datasets:
         spike_sorting:
           path: spike_sorting/probe00a
         processed_ephys:
           path: processed_ephys_data/probe00a
         picks:
           path: spike_sorting/probe00a
         output:
           path: spike_sorting/probe00a
   ```

   Every other path in the file is relative to this one.

3. Save the file.

Each shank of this probe was spike sorted separately, so each one has its own entry under
`probes`: `probe00a` to `probe00d`.

## 2. Open the session

1. Click the `...` button and choose **Open session YAML…**.
2. Choose the YAML file you edited.

The data loads with all four shanks side by side in a grid, each titled with its name from the
YAML. The titles of the active shank are red. This data has no histology images, so the Slice
figure shows the Allen atlas template (`CCF`) and annotation.

:::{admonition} TODO
:class: warning

Screenshot of the grid layout after loading Sample data 2.
`_static/tutorials/02_multi_shank/grid.png`
:::

## 3. Move between shanks

1. Press {kbd}`Right` and {kbd}`Left` to step through the shanks. The red titles move to show
   which shank is active.
2. Press {kbd}`T`. The layout changes to one tab per shank, and only the active shank is shown.
3. Press {kbd}`Right` and {kbd}`Left` again: each press moves to the next tab, with that shank's
   data and track.
4. Press {kbd}`T` to go back to the grid.

Stay in the grid layout for the rest of the tutorial.

## 4. Align the first shank

:::{admonition} TODO
:class: warning

Landmarks for `probe00a` (2–3): for each, the feature, the plot it is clearest in, its approximate
depth, and the matching region boundary.
:::

1. Add reference line pairs to `probe00a` at [TODO: landmarks], and press {kbd}`Enter` after
   each.

Double-clicking in a shank's figures makes it the active shank and adds the lines to it. Each
shank is aligned on its own: the lines and the fit of `probe00a` do not change the other shanks.

## 5. Use the neighbouring shank as a check

The shanks are fixed 250 µm apart on the probe, so they pass through similar anatomy.

1. Look at `probe00b` next to the shank you aligned. [TODO: what the learner should notice, e.g.
   the same boundary visible at a similar depth in the electrophysiology.]
2. Align `probe00b`, starting with [TODO: the landmark shared with `probe00a`].

The Fit figure now shows a line for each aligned shank. Lines with very different slopes are a sign
that one of the shanks may be matched to the wrong boundary.

:::{admonition} TODO
:class: warning

Screenshot of the grid with two shanks aligned, and the Fit figure showing both fits.
`_static/tutorials/02_multi_shank/two_shanks.png`
:::

## 6. Upload

1. Press {kbd}`Shift+U`. A dialog lists the shanks.
2. Tick `probe00a` and `probe00b`, and click **OK**.
3. Click **Yes** for each shank.

Each shank's results are written to its own `output` folder from the YAML, which for this data is
the shank's spike sorting folder: `spike_sorting/probe00a` and `spike_sorting/probe00b`. Because
each shank is a separate entry, the files have no shank suffix: `channel_locations.json` and
`prev_alignments.json`.

## What you've learned

You have:

- loaded a session from a session YAML
- moved between shanks, and switched between the tabbed and grid layouts
- aligned two shanks of one probe, using one to check the other
- uploaded a chosen set of shanks

## Next steps

- {ref}`explanation-shanks-probes-configurations`: how shanks, probes and configurations relate
- {ref}`howto-configure`: session YAMLs for your own data, including
  {ref}`two versions of a recording side by side <howto-compare-two-configurations>`
