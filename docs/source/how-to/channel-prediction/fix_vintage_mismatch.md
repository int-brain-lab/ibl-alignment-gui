(howto-fix-vintage-mismatch)=
# How to fix a feature vintage mismatch

This guide shows you how to get a model running when it reports that your features don't match it.

A model can only run on features of the vintage it was trained on. The error names the model's
vintage, for example:

```text
Features file is missing required column(s): [...]

The model (2026_W39) needs features computed with the 2026_W39 feature set.
```

## Steps

Choose one of:

**Use a model release that matches your features**

1. Find the vintage of your features. If the file records it, the error says which release to load.
2. Load that release of the model from `Plugins → Channel Prediction → Load … model`, with the
   vintage as the release tag (see {ref}`howto-choose-model-release`).

**Compute features of the model's vintage**

1. Compute the features again with a version of `ibleatools` that produces the model's vintage
   (see {ref}`howto-compute-features`).
2. Load the new file (see {ref}`howto-give-shanks-features`).

A model release whose `ephysatlas_model.json` names no `vintage` cannot be run. Use another release.

## Related

- {ref}`explanation-channel-prediction-models`: what a vintage is
