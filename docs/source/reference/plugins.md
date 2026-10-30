(ref-plugins)=
# Plugins

This page lists the plugins that come with the GUI. Most are opened from the `Plugins` menu. To
write your own, see {ref}`howto-extend`.

| Plugin | Opened from | Available in |
|---|---|---|
| Cluster features | Clicking a cluster in an image plot | All modes |
| Range controller | `Plugins → Range Controller` | All modes |
| 3D features | `Plugins → 3D features` | All modes |
| Channel prediction | `Plugins → Channel Prediction` | All modes, with the `ephysatlas` extra and a session with features |
| Ephys features | `Plugins → Ephys Features` | `alignment-gui-ibl` only |
| Upload dialog | Uploading with more than one shank | All modes |
| QC dialog | Uploading | `alignment-gui-ibl` only |

## Cluster features

Shows the autocorrelogram and template waveform of a single cluster. Click a cluster in an image
plot that shows cluster-level data to open a window for it.

`Plugins → Cluster Features` has entries to minimise or show ({kbd}`M`) and to close
({kbd}`Alt+X`) all the cluster windows.

## Range controller

Sets the range of each plot type with a pair of sliders:

- image plots and probe plots: the colour scale
- line plots: the x axis

| Action | How |
|---|---|
| Change a range | Drag the minimum and maximum sliders |
| Set an exact range | Type the value in the box beside the slider |
| Reset one plot type | **Reset** beside its sliders |
| Reset every plot | {kbd}`R` |

With more than one shank or configuration, checkboxes choose the shanks (or **All**) and the
configurations (or **both**) the ranges apply to.

## 3D features

Shows the probe trajectory in a 3D brain.

- The channels along the probe are coloured by the plot shown in the Probe plots panel, and update
  when it changes.
- When the image plot shows clusters, the clusters are shown at their locations.

| Control | Effect |
|---|---|
| Cluster marker size slider | Size of the cluster markers |
| **Show regions** | Show or hide the brain regions |
| **Show picks** | Show or hide the traced trajectory points |

## Channel prediction

Predicts the brain region of each channel from its electrophysiology features, and aligns the
probe automatically. See {ref}`ref-channel-prediction`.

## Ephys features

Compares the probe with other insertions in the IBL ephys atlas. Choose a brain region from those
along the probe, and a feature: the feature is shown along every insertion that passes through
that region, ten insertions per page, beside each insertion's brain regions. The colour map can
be changed.

## Upload dialog

Shown when uploading a session with more than one shank: tick the shanks to upload, or **All**. The
same dialog is used for **Save Progress**.

## QC dialog

Shown in `alignment-gui-ibl` for each shank before its alignment is uploaded. Cancelling it
cancels the upload.

Confidence of alignment
: High, Medium or Low.

QC for ephys recording
: Pass, Warning or Critical. For Warning or Critical, at least one problem must be ticked under
  **Describe problem with recording**. The list of problems comes from `ibllib`.

Resolve this alignment
: Whether to mark the probe's alignment as resolved with this alignment.

**Apply to all shanks**
: Shown with more than one shank. Gives the remaining shanks the same assessment instead of asking
  for each.
