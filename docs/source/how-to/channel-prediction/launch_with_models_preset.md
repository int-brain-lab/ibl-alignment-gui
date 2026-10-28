(howto-launch-with-models-preset)=
# How to skip the model dialogs

This guide shows you how to open a session with the models and features already chosen, so you can
select a prediction straight away.

## Prerequisites

- The `ephysatlas` extra installed (see {ref}`howto-install-extras`).
- The example script `examples/launch_with_local_prediction.py` from the repository.

## Steps

1. Run the script with the session YAML, and a `--model-dir` for every model you have stored
   locally:

   ```console
   python examples/launch_with_local_prediction.py \
       --yaml /path/to/session.yaml \
       --model-dir /path/to/ea-decoder-channel-xgboost/<vintage> \
       --model-dir /path/to/ea-encoder-channel/<vintage>
   ```

   Each `--model-dir` is a release directory holding an `ephysatlas_model.json`, which says which
   model it is. Models without a `--model-dir` are downloaded from the Hugging Face Hub at their
   default release.

2. If the YAML has no `features` dataset, add `--features /path/to/features.pqt`.

3. When the window opens, select a prediction from `Prediction Plots`, such as
   `XGBoost region prediction` or `Auto-alignment`. It runs without asking for a model.

## Related

- {ref}`howto-choose-model-release`
