(explanation-offline-and-online-modes)=
# Offline and online modes

The GUI can be started in two ways: `alignment-gui` works on data on your own disk, and
`alignment-gui-ibl` works on data held in the IBL Alyx database. This page explains how the two
differ, and how the GUI is organised so that the same alignment code serves both.

## The same GUI, different sources

In both modes the figures, the alignment and the plugins are the same. What changes is where each
piece of data comes from and where the results go:

| | Offline (`alignment-gui`) | Online (`alignment-gui-ibl`) |
|---|---|---|
| Choosing a session | A data folder or a session YAML | Subject, session and probe dropdowns, or `--pid` |
| Spike sorting and processed data | Files on disk | Downloaded through ONE and cached locally |
| Raw data snippets | Read from the SpikeGLX files on disk | Streamed through ONE |
| Traced trajectory | `xyz_picks.json` on disk | The insertion's trajectory in Alyx |
| Previous alignments | `prev_alignments.json` | Stored with the insertion in Alyx |
| Uploading | Writes JSON files to the output folder | Saves to Alyx, with a quality assessment |
| Channel prediction features | A features file | The ephys-atlas feature table on Alyx |
| Extra plots and plugins | | Passive stimulus responses, receptive field maps, ephys features |

Offline mode needs nothing beyond the base install. Online mode needs the `ibl` extra and an Alyx
account.

## Resolved alignments

Online, an insertion can be marked as resolved: several people have aligned it, and one alignment
has been agreed as final. The resolved alignment is the one the GUI starts from, and the QC dialog
offers to resolve the insertion with a new alignment. Offline there is no such shared state; the
GUI starts from the most recent alignment in `prev_alignments.json`.

## How the GUI is organised

Each shank is given a set of loaders, one per kind of data:

| Loader | Provides |
|---|---|
| `data` | Spike sorting and processed electrophysiology |
| `geom` | Probe geometry |
| `ephys` | Raw data snippets |
| `align` | The traced trajectory and previous alignments |
| `upload` | Saving the results |
| `features` | Ephys-atlas features |
| `hist` | Slice images |
| `plots` | The plots, computed from the loaded data |

Each loader is defined by an abstract base class with a local and a ONE implementation, for example
`DataLoaderLocal` and `DataLoaderOne`. A probe handler builds the right set for the mode:
`ProbeHandlerLocal` for a data folder, `ProbeHandlerLocalYaml` for a session YAML, and
`ProbeHandlerONE` for Alyx. Everything above the loaders, from the alignment to the plugins, works
the same whichever implementation it is given.

This is also how the GUI could be connected to another database: by writing implementations of
the loaders for it, and a probe handler that builds them.

## Related

- {ref}`ref-command-line`: the two commands
- {ref}`howto-align-ibl-insertion`: aligning in online mode
- The loaders in the {doc}`../api`: `ibl_alignment_gui.loaders`
