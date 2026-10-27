(howto-channel-prediction)=
# Channel prediction

These guides show you how to run the region prediction and auto-alignment models of the Channel
Prediction plugin on your own data. If you haven't used the plugin before, start with the tutorial
{ref}`tutorial-channel-prediction`. What the models do is explained in
{ref}`explanation-channel-prediction-models`, and every menu entry is listed in
{ref}`ref-channel-prediction`.

The plugin needs the `ephysatlas` extra (see {ref}`howto-install-extras`).

The guides cover three kinds of task:

Getting the features ready
: Compute the per-channel features the models run on, and tell the GUI which features file
  belongs to each shank.

Choosing the models
: Run a model release other than the default, or open a session with the models and features
  already chosen so you can skip the dialogs.

Fixing problems
: Get a model running when it reports that your features don't match it.

```{toctree}
:maxdepth: 1

compute_features
give_shanks_features
choose_model_release
launch_with_models_preset
fix_vintage_mismatch
```
