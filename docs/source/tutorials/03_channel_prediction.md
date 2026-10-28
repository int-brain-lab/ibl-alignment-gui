(tutorial-channel-prediction)=
# Predicting brain regions from the electrophysiology

In this tutorial you will use the Channel Prediction plugin on the probe from
{ref}`tutorial-first-alignment`. You will compare the regions a model predicts from the
electrophysiology with the histology, run the automatic alignment, and use what the models show
to place your own reference lines.

:::{admonition} TODO
:class: warning

Screenshot of the prediction panel showing `XGBoost region prediction` next to the ephys plots.
`_static/tutorials/03_channel_prediction/result.png`
:::

The models are downloaded the first time you use them, which can take a few minutes.

## Before you start

- Complete {ref}`tutorial-first-alignment`.
- Install the `ephysatlas` extra (see {ref}`howto-install-extras`). It also installs `torch`,
  which the auto-alignment model needs.
- Download the `sample_data_1_prediction` folder ([TODO: download link]) and put it next to the
  `sample_data1` folder, in the same parent folder. It holds:

  - `ephys_atlas_features_2026_W42.pqt`: the electrophysiology features of every channel of
    Sample data 1, computed from the raw data with
    [`ibleatools`](https://github.com/int-brain-lab/ibleatools)
  - `sample_data_1_prediction.yaml`: a session YAML that loads Sample data 1 with these features

## 1. Check the plugin is installed

1. Start the GUI with `alignment-gui`.
2. Open the `Plugins` menu. It has a `Channel Prediction` entry, greyed out until you load a
   session with features in the next step.

If it does not, the `ephysatlas` extra is not installed in the environment you started the GUI
from.

## 2. Load the data with its features

The features are given to the GUI in a session YAML, alongside the data you opened in
{ref}`tutorial-first-alignment`.

1. Open `sample_data_1_prediction.yaml` in a text editor:

   ```yaml
   path: /path/to/ephys_alignment_sample_data

   probes:
     probe00:
       datasets:
         # The recording, histology and alignments of Sample data 1
         spike_sorting:
           path: sample_data1
         histology:
           path: sample_data1
         output:
           path: sample_data1
         # The features the models run on
         features:
           path: sample_data_1_prediction/ephys_atlas_features_2026_W42.pqt
   ```

2. Set `path` to the full path of the folder that holds `sample_data1` and
   `sample_data_1_prediction`, and save the file.

   The other paths are relative to it. The spike sorting and histology come from `sample_data1`,
   and the trajectory and the RMS and power spectra are read from there too, because they are
   saved beside the spike sorting. The results are written to `sample_data1` as well, so you carry
   on from your alignment in {ref}`tutorial-first-alignment`. Only the features come from
   `sample_data_1_prediction`.

3. In the GUI, click `...`, choose **Open session YAML…**, and choose
   `sample_data_1_prediction.yaml`.

The data loads as before, and your alignment from {ref}`tutorial-first-alignment` is selected in
the alignment dropdown. With the features loaded, the `Feature Plots` and `Prediction Plots` menus
and `Plugins → Channel Prediction` are now available. Before, they were greyed out: hovering over
one says why.

:::{admonition} TODO
:class: warning

Screenshot of Sample data 1 loaded from `sample_data_1_prediction.yaml`, with the `Prediction Plots`
menu open.
`_static/tutorials/03_channel_prediction/loaded.png`
:::

## 3. Look at the features

1. Open the `Feature Plots` menu and choose `Ephys Atlas`. Every feature is shown side by side,
   normalised along the probe.
2. Press {kbd}`Alt+3` until the Probe plots panel shows an `Ephys Atlas - <feature>` entry.
   [TODO: name a feature where a boundary is clear, and what to look for.]

These are the numbers the models work from: the same kinds of signal you looked at by eye in
{ref}`tutorial-first-alignment`, computed for every channel.

## 4. Predict the regions

1. Choose `original` from the alignment dropdown, so the histology shows the track before any
   alignment.
2. Open the `Prediction Plots` menu and choose `XGBoost region prediction`.
3. The model's dialog opens. Keep the default repository and release, and click **OK**. The model
   is downloaded the first time, then the prediction runs.

A panel appears beside the ephys plots, showing the region the model predicts for each channel
from the electrophysiology alone. Each predicted region is labelled with its acronym.

4. Compare the prediction with the regions in the Histology figure. Look for boundaries where the
   two agree, and where they do not. Press {kbd}`Alt+M` to show the histology at the `Beryl` or
   `Cosmos` level, with fewer, larger regions, and again to go back to `Allen`.

:::{admonition} TODO
:class: warning

What the learner should see for Sample data 1: where the prediction and the histology agree, and
where they differ.
`_static/tutorials/03_channel_prediction/prediction.png`
:::

5. Choose `XGBoost region prediction (cumulative)`. Each channel now shows the probability of
   every region, stacked. Where one colour fills the width, the model is confident; where several
   share it, it is unsure.

## 5. Run the automatic alignment

1. Choose `Auto-alignment` from the `Prediction Plots` menu, and click **OK** in the model's
   dialog.
2. A progress dialog shows the alignment running. When it finishes, a result window opens.

The result window shows how the model matched the recording to the track:

- the track on coronal and sagittal slices, with the alignment it found
- the cost matrix: every recorded channel against every point along the track, with the best path
  through it
- stripes along the track: the atlas regions, the confidence, and the recorded and predicted
  features

:::{admonition} TODO
:class: warning

Screenshot of the auto-alignment result window for Sample data 1.
`_static/tutorials/03_channel_prediction/auto_alignment.png`
:::

3. Back in the main window, the prediction panel shows the atlas regions at the depths the
   automatic alignment found. Choose `Auto-alignment (confidence)` to see how confident it is
   along the probe: green is high, red is low.
4. Press {kbd}`Shift+P` to see all the predictions side by side, and compare the two models with
   the auto-alignment. Choose any image plot to go back.

To see the result window again later, use `Plugins → Channel Prediction → Show auto-alignment
results`.

## 6. Use the predictions in your own alignment

The predictions do not change your alignment. They are a guide for placing reference lines.

1. Choose `Cosmos` with {kbd}`Alt+M`.
2. [TODO: pick one boundary where the auto-alignment is confident and differs from the original
   track, and describe it.] Add a reference line pair there, with the Ephys line on the
   electrophysiology feature and the Histology line on the boundary.
3. Press {kbd}`Enter`, and switch back to `Auto-alignment` to compare.
4. Upload with {kbd}`Shift+U` when you are happy with it.

## What you've learned

You have:

- loaded a session and its features from a session YAML
- predicted regions from the electrophysiology and compared them with the histology
- read the confidence of a prediction
- run the automatic alignment and used it to guide your own

## Next steps

- {ref}`explanation-channel-prediction-models`: what the models do, and their limits
- {ref}`howto-compute-features`: compute features for your own recordings
- {ref}`howto-choose-model-release`: use a different model release
