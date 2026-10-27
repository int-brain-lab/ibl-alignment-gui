(ref-session-yaml)=
# Session YAML

This page lists every key of the session YAML, the file that tells the GUI where each dataset of a
session is. Open it from the data source button (`...` → **Open session YAML…**) or with
`alignment-gui -y <file>` (see {ref}`ref-command-line`).

## Structure

```yaml
path: /base/folder              # optional

defaults:                       # optional
  <dataset>:
    path: <path>

probes:                         # either probes ...
  <probe name>:
    path: <path>                # optional
    datasets:
      <dataset>:
        path: <path>

configurations:                 # ... or configurations, not both
  <configuration name>:
    path: <path>                # optional
    probes:
      <probe name>:
        ...                     # as under probes above
```

## Keys

### Top level

`path`
: Base folder that relative paths are resolved against. Optional.

`defaults`
: A path for each dataset, used for every probe that does not give its own. Optional.

`probes`
: The probes of the session, by name. The names are used for the shank tabs. Use either `probes`
  or `configurations`.

`configurations`
: Up to two configurations, by name, shown side by side and aligned together. Each holds its own
  `probes`. Every configuration must contain the same probe names. A file with `probes` and no
  `configurations` is treated as a single configuration.

### Configuration

`path`
: Base folder for the probes of this configuration. Optional.

`probes`
: The probes of this configuration, by name, as under the top-level `probes`.

### Probe

`path`
: Base folder for the datasets of this probe. Optional.

`datasets`
: A path for each dataset, by dataset name. A dataset given as `{}`, or left out, uses its path
  from `defaults`.

### Dataset

`path`
: Folder holding the dataset, absolute or relative. `~` is expanded to the home folder. For
  `features` it is the path to a file rather than a folder.

## Datasets

Every probe must give `picks` and `output`, and at least one of `spike_sorting` and
`processed_ephys`. The other datasets are optional. Datasets are not filled in from one another,
with one exception: if `processed_ephys` is not given, the RMS and power spectra are read from
`raw_ephys`.

| Dataset | Contents | If not given |
|---|---|---|
| `spike_sorting` | Spike sorting output | Required if `processed_ephys` is not given; otherwise the plots that need spike sorting are empty |
| `processed_ephys` | RMS and power spectra | Read from `raw_ephys`; required if `spike_sorting` is not given |
| `picks` | Traced probe trajectory | Required |
| `output` | Where the results are written | Required |
| `raw_ephys` | Raw SpikeGLX recordings and metadata | No raw data snippets are shown |
| `histology` | Histology volumes | Only the Allen atlas slices are shown |
| `features` | Per-channel features file | None, unless a file is chosen from the Channel Prediction menu |

The files expected in each dataset are listed in {ref}`ref-input-datasets` and
{ref}`ref-output-files`.

## Path resolution

A dataset's path is its own `path`, or the one from `defaults` if it gives none. An absolute path
is used as it is. A relative path has the base folders prepended in this order until it is
absolute:

1. the probe's `path`
2. the configuration's `path`
3. the top-level `path`

If it is still relative after all three, loading fails with an error. How this works, with worked
examples, is explained in {ref}`explanation-session-yaml-resolution`.

## Errors

Loading the file fails when:

- the file does not exist, is empty, or does not hold a mapping
- a probe does not give `picks` or `output`, or gives neither `spike_sorting` nor `processed_ephys`
- there are more than two configurations
- the configurations do not all contain the same probes
- a relative path cannot be made absolute

## Example

```yaml
path: /mnt/data/subject01/2025-01-15

defaults:
  histology:
    path: /mnt/data/subject01/histology

probes:
  probe00:
    datasets:
      spike_sorting:
        path: probe00/kilosort
      processed_ephys:
        path: probe00/kilosort
      raw_ephys:
        path: probe00/spikeglx
      picks:
        path: probe00/picks
      output:
        path: probe00/alignment
```
