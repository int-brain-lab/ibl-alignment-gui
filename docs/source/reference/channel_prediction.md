(ref-channel-prediction)=
# Channel Prediction plugin

This page lists the models, menu entries and feature files of the Channel Prediction plugin.
Requires the `ephysatlas` extra (see {ref}`ref-optional-dependencies`). For how to use it, see
{ref}`howto-channel-prediction`.

## Models

| Model | Hugging Face repository | Default release | Prediction plot entries | Needs `torch` |
|---|---|---|---|---|
| XGBoost region decoder | `int-brain-lab/ea-decoder-channel-xgboost` | `2026_W39` | `XGBoost region prediction`, `XGBoost region prediction (cumulative)` | No |
| ProbeTransformer region decoder | `int-brain-lab/ea-decoder-channel-transformer` | `2026_W39` | `ProbeTransformer region prediction`, `ProbeTransformer region prediction (cumulative)` | Yes |
| Alignment model | `int-brain-lab/ea-encoder-channel`, set by [`ibleatools`](https://github.com/int-brain-lab/ibleatools) | Set by `ibleatools` | `Auto-alignment`, `Auto-alignment (confidence)` | Yes |

Each model is loaded with `ephysatlas.load_pretrained`, from a Hugging Face repository and release
tag or from a local release directory holding an `ephysatlas_model.json`. A downloaded release is
cached in `~/.cache/ephysatlas/models`; a local release takes precedence over the Hub.

A release whose `ephysatlas_model.json` names no `vintage` is refused, because the features to run
it on are unknown.

## Menu entries

`Plugins → Channel Prediction`:

`Load XGBoost region prediction model (Hugging Face)…`
: Choose the repository and release, or a local directory, for the XGBoost decoder.

`Load ProbeTransformer region prediction model (Hugging Face)…`
: The same for the ProbeTransformer. Only shown when `torch` is installed.

`Load alignment model (Hugging Face)…`
: The same for the alignment model. Only shown when `torch` is installed.

`Show auto-alignment results`
: Reopen the auto-alignment result windows. Greyed out until a shank has been aligned.

## Features

The models read per-channel features (see {ref}`ref-input-datasets`), loaded with the session:

- from a session YAML, the file given as the `features` dataset
- from a data folder, the one parquet file in the folder with `features` in its name (`.pqt` or
  `.parquet`); with several such files, none is used and the GUI logs which it found
- in `alignment-gui-ibl`, the ephys-atlas feature table on Alyx of the vintage the model was
  trained on

A file holding every shank of a probe sorted together is split by channel for each shank.

The features must be of the model's vintage. Missing columns are reported when the model runs. If
the Alyx table of the model's vintage has no features for the insertion, the model reports it
instead of running on another vintage's features, and suggests a release of the model trained on
the session's vintage when one exists.

## Results

| Menu | Entries added |
|---|---|
| `Prediction Plots` | `None`, the entries of each model (table above), and `All predictions (side by side)`. Each region decoder shows the most probable region per channel, and its `(cumulative)` stacked class probabilities. |
| `Feature Plots` | `Ephys Atlas`: every feature, normalised across the shank's channels, side by side |
| `Probe Plots` | `Ephys Atlas - <feature>`: one entry per feature, on the probe geometry |

The prediction entries are offered as soon as a session is loaded. Without features, selecting
one shows a warning saying how to load a features file. Selecting one whose model has not been
loaded opens the model's load dialog. A prediction is shown in a panel beside the ephys
plots, placed by depth along the probe (see {ref}`ref-interface-ephys`).

`All predictions (side by side)` shows only the models already loaded, computed for the shanks
that need them in one go; the panels of a model that is not loaded say so.

Selecting `Auto-alignment` or `Auto-alignment (confidence)` aligns each shank, with a progress
dialog, and opens a result window per shank showing:

- the histology track, the planned trajectory when the features table has one, and the inferred
  alignment, on coronal and sagittal slices
- the cost of every recorded channel against every track sample, with the alignment path
- stripes along the track: the atlas regions, then the confidence and the recorded features where
  the probe was aligned, next to the features the model predicts along the whole track

`Auto-alignment` shows the atlas regions along the track at the probe depths the alignment found.
`Auto-alignment (confidence)` shows the alignment confidence along the probe: green for high, red
for low, and grey beyond the probe and for channels that could not be aligned.

Each shank is aligned once. Switching between the two entries does not align again or reopen the
windows. Loading a new model closes the windows and starts again; loading a new session closes
them.

## Launching with models preset

`examples/launch_with_local_prediction.py` opens a session with the model sources already set, so
the load dialogs are skipped. The features come from the session YAML:

```console
python examples/launch_with_local_prediction.py \
    --yaml /path/to/session.yaml \
    --model-dir /path/to/ea-decoder-channel-xgboost/<vintage>
```

`--model-dir` can be given once per model. The `model_class` in the release's
`ephysatlas_model.json` decides which model it is. Models without a `--model-dir` are downloaded
from the Hub at their default release.
