(howto-choose-model-release)=
# How to use a specific model release

This guide shows you how to run a model release other than the default, or one stored on your own
disk.

## Prerequisites

- The `ephysatlas` extra installed (see {ref}`howto-install-extras`).
- A session with features loaded (see {ref}`howto-give-shanks-features`). Until then
  `Plugins → Channel Prediction` is greyed out.

## Steps

1. Open `Plugins → Channel Prediction` and choose the model's `Load … model (Hugging Face)…` entry.
2. In the dialog, either:

   - give the Hugging Face repository and the release tag, such as `2026_W42`, or
   - choose a local release directory, one that holds an `ephysatlas_model.json`.

3. Click **OK**. The release is downloaded once and cached in `~/.cache/ephysatlas/models`.

4. Select the model's entry in `Prediction Plots` to run it.

Loading a model clears its previous predictions and closes any auto-alignment result windows.

The release must be trained on the vintage of your features (see
{ref}`howto-fix-vintage-mismatch`).

## Related

- {ref}`ref-channel-prediction`: repositories and default releases
- {ref}`howto-launch-with-models-preset`: set the release once instead of in the dialog
