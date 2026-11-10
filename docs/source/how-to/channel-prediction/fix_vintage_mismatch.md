(howto-fix-vintage-mismatch)=
# How to fix a feature vintage mismatch

This guide shows you how to get a model running when it reports that your features don't match it.

A model can only run on features of the vintage it was trained on, the `vintage` in its release's
`ephysatlas_model.json`. The error names the vintage of your features, when they record it, and the
vintage the selected release requires, for example:

```text
You have features of vintage 2026_W26, but the selected release 0.0.1 of the model
int-brain-lab/ea-encoder-channel requires features of vintage 2026_W42.

You can either:

1. Load a release of the model trained on features of vintage 2026_W26: ...

2. Re-extract your features with vintage 2026_W42.
```

## Steps

Choose one of:

**Use a model release that matches your features**

1. Find the vintage of your features. If the file records it, the error names it.
2. Find a release of the model trained on that vintage: one whose `ephysatlas_model.json` gives it
   as `vintage`. The release tag names the model, not its features, so it need not be the vintage.
3. Load that release from `Plugins → Channel Prediction → Load … model` (see
   {ref}`howto-choose-model-release`).

**Compute features of the model's vintage**

1. Compute the features again with a version of
   [`ibleatools`](https://github.com/int-brain-lab/ibleatools) that produces the model's vintage
   (see {ref}`howto-compute-features`).
2. Point the session at the new file, and load the session again (see
   {ref}`howto-give-shanks-features`).

A model release whose `ephysatlas_model.json` names no `vintage` cannot be run. Use another release.

## Related

- {ref}`explanation-channel-prediction-models`: what a vintage is
