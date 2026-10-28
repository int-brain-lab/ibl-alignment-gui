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

It takes about 30 minutes, plus the time to download the models the first time.

## Before you start

- Complete {ref}`tutorial-first-alignment`.
- Install the `ephysatlas` extra (see {ref}`howto-install-extras`). It also installs `torch`,
  which the auto-alignment model needs.
- Download the features file for Sample data 1, [TODO: file name and download link], and put it
  in the Sample data 1 folder. It holds the electrophysiology features of every channel, computed
  from the raw data with [`ibleatools`](https://github.com/int-brain-lab/ibleatools).

## 1. Check the plugin is installed

1. Start the GUI with `alignment-gui`.
2. Open the `Plugins` menu. It has a `Channel Prediction` entry.

If it does not, the `ephysatlas` extra is not installed in the environment you started the GUI
from.

## 2. Load the data with its features

The features are added as a dataset in a session YAML, alongside the folder you opened in
{ref}`tutorial-first-alignment`.

1. Create a file called `sample_data_1.yaml` in the Sample data 1 folder:

   ```yaml
   path: /path/to/sample_data1      # <- the Sample data 1 folder

   probes:
     probe00:
       datasets:
         spike_sorting:
           path: .
         processed_ephys:
           path: .
         picks:
           path: .
         output:
           path: .
         histology:
           path: .
         features:
           path: [TODO: features file name]
   ```

   `.` is the Sample data 1 folder itself, so this is the same data you opened directly in
   {ref}`tutorial-first-alignment`, plus the features.

2. Click `...`, choose **Open session YAML…**, and choose `sample_data_1.yaml`.

The data loads as before. Your alignment from {ref}`tutorial-first-alignment` is selected in the
alignment dropdown.

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
from the electrophysiology alone. Hover over a region to see its name.

4. Compare the prediction with the regions in the Histology figure. Look for boundaries where the
   two agree, and where they do not.

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

- added a features file to a session
- predicted regions from the electrophysiology and compared them with the histology
- read the confidence of a prediction
- run the automatic alignment and used it to guide your own

## Next steps

- {ref}`explanation-channel-prediction-models`: what the models do, and their limits
- {ref}`howto-compute-features`: compute features for your own recordings
- {ref}`howto-choose-model-release`: use a different model release
