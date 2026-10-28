(explanation-shanks-probes-configurations)=
# Shanks, probes and configurations

The GUI can show several recordings at once. This page explains the three ways it groups them,
how they relate to each other, and why the choice matters for where the results are written.

## Shanks

A shank is the unit the GUI aligns. Every shank has its own traced trajectory, its own reference
lines and its own fit, even when several shanks belong to one probe. A multi-shank probe is
therefore aligned once per shank.

Showing the shanks of a probe side by side still helps: the shanks are fixed relative to each
other, so the alignment of one is a check on its neighbours. A boundary that one shank crosses at a
given depth should appear at a similar depth on the shank next to it, if the anatomy is similar
there.

## Probes in the session YAML

A probe entry in the session YAML is a set of dataset folders: spike sorting, raw data, picks and
output. How a multi-shank probe is entered depends on how it was spike sorted.

Sorted as a whole
: One probe entry. The spike sorting holds every shank, and the GUI splits the channels into
  shanks using the probe geometry (see {ref}`ref-probe-geometry`). It expects one traced
  trajectory per shank, `xyz_picks_shank1.json`, `xyz_picks_shank2.json` and so on, and writes the
  results with the same shank suffix. All shanks can share one `output` folder.

Sorted one shank at a time
: One probe entry per shank. Each entry is a single-shank probe as far as the GUI is concerned: it
  expects `xyz_picks.json` and writes `channel_locations.json` with no suffix. Two entries writing
  to the same `output` folder would overwrite each other's results and read each other's previous
  alignments, so each needs its own `output` folder.

Either way the shanks look the same in the GUI: one tab, or one column in the grid layout, per
shank.

## Configurations

A configuration is a complete set of probes. With two configurations, the GUI shows the same
probes from two different sources side by side, for example:

- the same recording spike sorted with two sorters or with different parameters
- the same probe recorded with two channel maps, such as dense and sparse
- the same data before and after a change to the preprocessing

The two configurations are aligned together: a reference line placed on one applies to the same
shank in the other, since they are the same piece of tissue. This is why both configurations must
contain the same probe names; the names are how the GUI matches them up. At most two
configurations can be loaded.

## Choosing a layout

| Your data | Session YAML |
|---|---|
| One single-shank probe | One probe entry |
| One multi-shank probe, sorted together | One probe entry, one picks file per shank |
| One multi-shank probe, sorted per shank | One probe entry per shank, each with its own `output` |
| Two versions of the same recording | Two configurations with the same probe names |

The keys are listed in {ref}`ref-session-yaml`.
