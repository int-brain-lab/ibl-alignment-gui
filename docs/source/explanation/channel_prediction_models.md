(explanation-channel-prediction-models)=
# The channel prediction models

The channel prediction models are trained in the IBL Ephys Atlas project, and run in the GUI
through [`ibleatools`](https://github.com/int-brain-lab/ibleatools). They use the electrophysiology
of each channel to predict where it is: the region decoders without the histology at all, and the
alignment model by matching the recording to the traced track. This page explains what the models
do, how to read their output, and their limits. The model list and menu entries are in
{ref}`ref-channel-prediction`.

## Features

All the models work from per-channel features computed from the raw AP and LF data, such as
measures of signal amplitude and of LFP power in several frequency bands. These are the kind of
signals used by eye to find landmarks, summarised as numbers for every channel.

The features are computed with `ibleatools`, not by the GUI. A released set of feature definitions
is called a vintage, labelled by year and week, such as `2026_W26`. A model is trained on the
features of one vintage, and can only be run on features of that vintage: a different vintage may
lack columns the model needs, or compute them differently. This is why the GUI checks the
vintage of the features against the model, and refuses rather than running on mismatched features.

## Region decoders

The region decoders classify each channel into a brain region from its features.

XGBoost
: A gradient-boosted tree classifier that looks at each channel on its own.

ProbeTransformer
: A transformer that looks at all the channels of the shank together, with their depths, so the
  prediction for a channel can use the channels around it.

Both give a probability for every region. The `region prediction` entry shows the most probable
region per channel. The `(cumulative)` entry stacks the probabilities of all regions, which shows
where the model is unsure: a channel where two regions have similar probabilities is one the model
cannot place.

## Auto-alignment

The alignment model works the other way round. Instead of predicting regions from the recording,
it predicts the features the recording should show at every point along the traced track, from
where that point is in the atlas. The recorded features are then matched against the predicted
ones to find where along the track each channel fits best.

The result window shows this matching as a cost matrix, every recorded channel against every point
on the track, with the path of the best match drawn through it. The path gives an alignment of the
probe along the track, and from it a region for every channel.

A second model estimates how trustworthy the alignment is at each channel. This is the
`Auto-alignment (confidence)` view: green where the match is confident, red where it is not.

## Using the predictions

The predictions are a second opinion, not ground truth. They are most useful:

- as a starting point, to see roughly where the boundaries are before placing reference lines
- as a check, when a manual alignment and the predictions disagree on a boundary
- to find candidate landmarks in a recording where the features are hard to read by eye

They are least reliable in regions that were rare in the training data, in recordings whose
quality differs from the training data, and where the confidence is low. A boundary placed only
because a model predicted it is no stronger than the model's confidence at that point.
