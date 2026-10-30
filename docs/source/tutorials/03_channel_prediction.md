(tutorial-channel-prediction)=
# Predicting brain regions from the electrophysiology

In this tutorial you will use the Channel Prediction plugin on the probe from
{ref}`tutorial-first-alignment`. You will compare the regions a model predicts from the
electrophysiology with the histology, run the automatic alignment, and use what the models show
to place your own reference lines.

```{image} ../_static/tutorials/03_prediction/result.png
:alt: The GUI with a region prediction shown in a panel beside the ephys plots
:align: center
```

The models are downloaded the first time you use them, which can take a few minutes.

## Before you start

- Complete {ref}`tutorial-first-alignment`.
- Install the `ephysatlas` extra (see {ref}`howto-install-extras`). It also installs `torch`,
  which the auto-alignment model needs.
- Find the `sample_data_1_prediction` folder in the
  [sample data](https://ibl.flatironinstitute.org/public/ephys_alignment_sample_data.zip) you
  extracted for {ref}`tutorial-first-alignment`, next to the `sample_data1` folder. It holds:

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

```{image} ../_static/tutorials/03_prediction/loaded.png
:alt: Sample data 1 loaded from sample_data_1_prediction.yaml, with the Prediction Plots menu open
:align: center
```

## 3. Look at the features

1. Open the `Feature Plots` menu and choose `Ephys Atlas`. The ephys plots are replaced by every
   feature side by side, each normalised along the probe.

```{image} ../_static/tutorials/03_prediction/features.png
:alt: Every ephys-atlas feature side by side along the probe, with the reference lines of the alignment from the first tutorial
:align: center
```

2. Look at where the features change along the probe. Many of them change together at the
   reference lines of your alignment from {ref}`tutorial-first-alignment`: at the top of the
   thalamus, at about 1500 µm, and at the top of CA1, at about 2700 µm.

These are the numbers the models work from: the same kinds of signal you looked at by eye in
{ref}`tutorial-first-alignment`, computed for every channel. Choose any image plot to go back to
the ephys plots.

## 4. Predict the regions

1. Choose `original` from the alignment dropdown, so the histology shows the track before any
   alignment.
2. Open the `Prediction Plots` menu and choose `XGBoost region prediction`.
3. The model's dialog opens. Keep the default repository and release, and click **OK**. The model
   is downloaded the first time, then the prediction runs.

```{image} ../_static/tutorials/03_prediction/model_load_dialog.png
:alt: The dialog for loading the XGBoost region prediction model, with the default Hugging Face repository and release
:align: center
```

A panel appears beside the ephys plots, showing the region the model predicts for each channel
from the electrophysiology alone. Each predicted region is labelled with its acronym.

```{image} ../_static/tutorials/03_prediction/decoder_prediction.png
:alt: The XGBoost region prediction panel, with the thalamus, hippocampus and isocortex predicted along the probe
:align: center
```

4. Compare the prediction with the regions in the Histology figure. The model predicts at a
   coarser level than the histology shows, so press {kbd}`Alt+M` until the histology is shown in
   `Cosmos`, with fewer, larger regions, to compare them. Press {kbd}`Alt+M` until it is back to
   `Allen` before you go on.
5. Choose `XGBoost region prediction (cumulative)`. Each channel now shows the probability of
   every region, stacked. Where one colour fills the width, the model is confident; where several
   share it, it is unsure.

```{image} ../_static/tutorials/03_prediction/decoder_cumulative.png
:alt: The cumulative XGBoost region prediction, with the probabilities of the regions stacked along the probe
:align: center
```

6. Choose `ProbeTransformer region prediction`, and load it in the same way. It is a second model,
   trained on the same features. You will compare the two in step 6.

## 5. Run the automatic alignment

1. Choose `Auto-alignment` from the `Prediction Plots` menu, and click **OK** in the model's
   dialog.
2. A progress dialog shows the alignment running. When it finishes, a result window opens.

```{image} ../_static/tutorials/03_prediction/auto_alignment_popup.png
:alt: The auto-alignment result window, with the track on two slices, the cost matrix and the recorded and predicted features
:align: center
```

The result window shows how the model matched the recording to the track:

- the track on coronal and sagittal slices, with the alignment it found
- the cost matrix: every recorded channel against every point along the track, with the best path
  through it
- stripes along the track: the atlas regions, the confidence, and the recorded and predicted
  features

3. Close the result window. Back in the main window, the prediction panel shows the atlas region
   of each channel where the automatic alignment placed it on the track.

```{image} ../_static/tutorials/03_prediction/auto_align_prediction.png
:alt: The Auto-alignment panel, with the atlas regions where the alignment placed each channel
:align: center
```

4. Choose `Auto-alignment (confidence)` to see how confident the alignment is along the probe:
   green is high, red is low.

```{image} ../_static/tutorials/03_prediction/auto_align_confidence.png
:alt: The Auto-alignment (confidence) panel, coloured from green for high confidence to red for low
:align: center
```

To see the result window again later, use `Plugins → Channel Prediction → Show auto-alignment
results`.

## 6. Compare all the predictions

1. Press {kbd}`Shift+P`. The ephys plots are replaced by every prediction side by side: the two
   region decoders, each with its cumulative probabilities, and the auto-alignment with its
   confidence.

```{image} ../_static/tutorials/03_prediction/all_predictions.png
:alt: All the predictions side by side, next to the histology
:align: center
```

2. Compare the models. Boundaries that the models agree on, and that the auto-alignment is
   confident about, are the ones to trust most.

## 7. Use the predictions in your own alignment

The predictions do not change your alignment. They are a guide for placing reference lines, and
you can place the lines in the all predictions view.

1. Add a reference line pair at each boundary the models agree on: the Ephys line on the
   predicted boundary, and the Histology line on the matching boundary in the Histology figure.
   Here they agree on three: the top of the brain, the boundary between the isocortex and the
   hippocampus, and the boundary between the hippocampus and the thalamus.
2. Press {kbd}`Enter` to apply the fit.

```{image} ../_static/tutorials/03_prediction/alignment_from_predictions.png
:alt: Three reference line pairs placed on the boundaries the predictions agree on, with the fit applied
:align: center
```

3. Choose any image plot to go back to the ephys plots, and check the alignment against the
   features you used in {ref}`tutorial-first-alignment`.
4. Upload with {kbd}`Shift+U` when you are happy with it.

## What you've learned

You have:

- loaded a session and its features from a session YAML
- predicted regions from the electrophysiology and compared them with the histology
- read the confidence of a prediction
- run the automatic alignment
- compared all the predictions side by side, and placed reference lines where they agree

## Next steps

- {ref}`explanation-channel-prediction-models`: what the models do, and their limits
- {ref}`howto-compute-features`: compute features for your own recordings
- {ref}`howto-choose-model-release`: use a different model release
