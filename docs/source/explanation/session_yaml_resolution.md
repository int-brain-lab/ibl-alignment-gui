(explanation-session-yaml-resolution)=
# How session YAML paths are resolved

A session YAML can give every dataset path in full, but most of the time the datasets of a session
share a common folder, and the probes of a project share a folder layout. The YAML has two
mechanisms to avoid repeating those parts: base paths and defaults. This page explains how they
combine. The keys themselves are listed in {ref}`ref-session-yaml`.

## Base paths

A `path` can be given at three levels: the top of the file, each configuration, and each probe.
These are base folders. A dataset path that is relative is joined onto them, starting with the
closest:

1. the probe's `path`
2. the configuration's `path`
3. the top-level `path`

Joining stops as soon as the result is absolute. So a relative probe `path` is itself joined onto
the configuration and top-level paths, and an absolute path at any level cuts off the levels above
it. If the path is still relative after all three, loading fails.

## Defaults

The `defaults` section gives a path for a dataset that probes can leave out. A probe that does not
give a dataset, or gives it as `{}`, uses the path from `defaults` instead, and that path is then
resolved against the base paths like any other.

Defaults supply the dataset path, not a base folder. A probe that gives its own relative path for
a dataset does not have the default prepended; the default is simply not used.

This makes `defaults` suited to two things:

- folders shared by every probe, such as the histology, given as an absolute path
- folder names that are the same in every probe, such as `kilosort`, given as a relative path and
  resolved against each probe's own base path

## Examples

Each example shows the YAML and the folder the `spike_sorting` dataset resolves to.

**Top-level base path**

```yaml
path: /mnt/data
probes:
  probe00:
    datasets:
      spike_sorting:
        path: probe00/kilosort
```

`/mnt/data/probe00/kilosort`

**Configuration base path**

```yaml
path: /mnt/data
configurations:
  dense:
    path: dense_session
    probes:
      probe00:
        datasets:
          spike_sorting:
            path: kilosort
```

`/mnt/data/dense_session/kilosort`

**Probe base path**

```yaml
path: /mnt/data
probes:
  probe00:
    path: probe00
    datasets:
      spike_sorting:
        path: kilosort
```

`/mnt/data/probe00/kilosort`

**Absolute dataset path**

```yaml
path: /mnt/data
probes:
  probe00:
    path: probe00
    datasets:
      spike_sorting:
        path: /custom/location/kilosort
```

`/custom/location/kilosort`: the base paths are ignored.

**Default with an empty dataset**

```yaml
path: /mnt/data
defaults:
  spike_sorting:
    path: pykilosort
probes:
  probe00:
    path: probe00
    datasets: {}
```

`/mnt/data/probe00/pykilosort`: the default is resolved against the probe's base path.

## A complete example

```yaml
path: /mnt/s0/Data

defaults:
  histology:
    path: /common/histology/subject_001
  spike_sorting:
    path: pykilosort
  raw_ephys:
    path: spikeglx

configurations:
  dense:
    path: dense_session
    probes:
      probe00:
        path: probe00
        datasets:
          processed_ephys:
            path: pykilosort
          picks:
            path: /custom/path/to/picks
          output:
            path: alignment_outputs

  sparse:
    path: sparse_session
    probes:
      probe00:
        path: probe00
        datasets:
          spike_sorting:
            path: kilosort
          processed_ephys:
            path: raw_ephys_data
          raw_ephys: {}
          picks:
            path: kilosort
          output:
            path: alignment_outputs
```

| Configuration | Dataset | Resolved path | Why |
|---|---|---|---|
| dense | `spike_sorting` | `/mnt/s0/Data/dense_session/probe00/pykilosort` | Default, resolved against the base paths |
| dense | `processed_ephys` | `/mnt/s0/Data/dense_session/probe00/pykilosort` | Own path |
| dense | `raw_ephys` | `/mnt/s0/Data/dense_session/probe00/spikeglx` | Default |
| dense | `picks` | `/custom/path/to/picks` | Absolute |
| dense | `histology` | `/common/histology/subject_001` | Absolute default |
| dense | `output` | `/mnt/s0/Data/dense_session/probe00/alignment_outputs` | Own path |
| sparse | `spike_sorting` | `/mnt/s0/Data/sparse_session/probe00/kilosort` | Own path, so the default is not used |
| sparse | `processed_ephys` | `/mnt/s0/Data/sparse_session/probe00/raw_ephys_data` | Own path |
| sparse | `raw_ephys` | `/mnt/s0/Data/sparse_session/probe00/spikeglx` | `{}` uses the default |
| sparse | `picks` | `/mnt/s0/Data/sparse_session/probe00/kilosort` | Own path |
| sparse | `histology` | `/common/histology/subject_001` | Absolute default |
| sparse | `output` | `/mnt/s0/Data/sparse_session/probe00/alignment_outputs` | Own path |

## Datasets that are not given

After the paths are resolved, a dataset that is still missing is not filled in from another, with
one exception: without `processed_ephys`, the RMS and power spectra are read from `raw_ephys`.
Which datasets are required is listed in {ref}`ref-session-yaml`.
