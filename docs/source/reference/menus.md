(ref-menus)=
# Menus

This page lists every entry in the menu bar. The plot menus list only the plots for which data was
found, so some entries appear only with certain datasets or modes.

## Image Plots

One plot at a time, in the left panel of the Ephys figure. {kbd}`Alt+1` / {kbd}`Shift+Alt+1` cycle
through them.

| Entry | Shows | Needs |
|---|---|---|
| `Amplitude` | Every spike by time and depth, coloured by amplitude | Spike sorting |
| `Cluster Amp vs Depth vs FR` | Clusters by amplitude and depth, coloured by firing rate | Spike sorting |
| `Cluster Amp vs Depth vs Duration` | Clusters by amplitude and depth, coloured by peak-to-trough duration | Spike sorting |
| `Cluster FR vs Depth vs Amp` | Clusters by firing rate and depth, coloured by amplitude | Spike sorting |
| `Firing Rate` | Firing rate over time and depth | Spike sorting |
| `Correlation` | Correlation of activity between depths | Spike sorting |
| `rms AP` | AP band RMS over time and depth | `_iblqc_ephysTimeRmsAP` |
| `rms LF` | LF band RMS over time and depth | `_iblqc_ephysTimeRmsLF` |
| `LF spectrum` | LF power by frequency and depth | `_iblqc_ephysSpectralDensityLF` |
| `Raw ap snippet <n>`, `Raw lf snippet <n>` | Short snippets of the raw data | Raw AP or LF data |
| One entry per passive stimulus | Responses to passive stimuli | Passive stimulus data (IBL sessions) and the `ibl` extra |

Clicking a cluster in a cluster plot opens its waveform (see {ref}`ref-plugins`).

## Line Plots

| Entry | Shows | Needs |
|---|---|---|
| `Firing Rate` | Mean firing rate along depth | Spike sorting |
| `Amplitude` | Mean amplitude along depth | Spike sorting |
| `Dead Channels` | Channels detected as dead | Raw AP data |
| `Noisy Channels Coherence` | Channels detected as noisy by coherence | Raw AP data |
| `Noisy Channels PSD` | Channels detected as noisy by power spectrum | Raw AP data |
| `Outside Channels` | Channels detected as outside the brain | Raw AP data |

{kbd}`Alt+2` / {kbd}`Shift+Alt+2` cycle through them.

## Probe Plots

Drawn on the probe geometry. The channels in the Slice figure are coloured by the plot shown here.
{kbd}`Alt+3` / {kbd}`Shift+Alt+3` cycle through them.

| Entry | Shows | Needs |
|---|---|---|
| `rms AP`, `rms LF` | Mean RMS per channel | `_iblqc_ephysTimeRms*` |
| `0 - 4 Hz`, `4 - 10 Hz`, `10 - 30 Hz`, `30 - 80 Hz`, `80 - 200 Hz` | LF power per channel in each band | `_iblqc_ephysSpectralDensityLF` |
| `RF Map - on`, `RF Map - off` | Receptive field maps | Receptive field mapping data (IBL sessions) and the `ibl` extra |
| `Ephys Atlas - <feature>` | One ephys-atlas feature per channel | Channel Prediction plugin |

## Feature Plots

| Entry | Shows | Needs |
|---|---|---|
| `Ephys Atlas` | Every ephys-atlas feature side by side | Channel Prediction plugin |

## Slice Plots

The image in the Slice figure. {kbd}`Alt+4` / {kbd}`Shift+Alt+4` cycle through them.

| Entry | Shows |
|---|---|
| `CCF` | Allen atlas average template |
| `Annotation` | Allen atlas annotation |
| `Histology red`, `Histology green` | Your histology volumes, when given (see {ref}`ref-input-datasets`) |

## Region Plots

The brain regions in the Histology figure. {kbd}`Alt+5` / {kbd}`Shift+Alt+5` cycle through them.

| Entry | Shows |
|---|---|
| `Allen`, `Beryl`, `Cosmos` | Atlas regions at three levels of detail (see {ref}`explanation-region-mappings`) |
| Model entries | Predicted regions and auto-alignment, with the Channel Prediction plugin (see {ref}`ref-channel-prediction`) |

## Filter Plots

The units shown in the spike sorting plots.

| Entry | Units shown |
|---|---|
| `All` | Every unit |
| `KS good` | Kilosort good units (`ks2_label`) |
| `KS mua` | Kilosort multi-unit (`ks2_label`) |
| `IBL good` | Units passing the IBL metrics (`label`) |
| Custom filters | Added by plugins (see {ref}`howto-add-unit-filter`) |

## Fit Options

| Entry | Shortcut |
|---|---|
| `Fit` | {kbd}`Enter` |
| `Remove Line` | {kbd}`Shift+D` |
| `Next` | {kbd}`Shift+Right` |
| `Previous` | {kbd}`Shift+Left` |
| `Reset` | {kbd}`Shift+R` |
| `Upload` | {kbd}`Shift+U` |
| `Save Progress` | {kbd}`Shift+S` |

## Display Options

| Entry | Shortcut |
|---|---|
| `Toggle Image Plots ->` … `Toggle Region Plots ->` | {kbd}`Alt+1` … {kbd}`Alt+5` |
| `Toggle Image Plots <-` … `Toggle Region Plots <-` | {kbd}`Shift+Alt+1` … {kbd}`Shift+Alt+5` |
| `Reset Axis` | {kbd}`Shift+A` |
| `Hide/Show Labels` | {kbd}`Shift+L` |
| `Hide/Show Lines` | {kbd}`Shift+H` |
| `Hide/Show Channels` | {kbd}`Shift+C` |
| `Toggle layout` | {kbd}`T` |
| `Next shank` | {kbd}`Right` |
| `Previous shank` | {kbd}`Left` |
| `Reset Range` | {kbd}`R` |

## Plugins

| Entry | See |
|---|---|
| `Cluster Features` | Minimise or close the cluster windows |
| `Range Controller` | {ref}`ref-plugins` |
| `3D features` | {ref}`ref-plugins` |
| `Channel Prediction` | {ref}`ref-channel-prediction` |
| `Ephys Features` | {ref}`ref-plugins` (`alignment-gui-ibl` only) |
