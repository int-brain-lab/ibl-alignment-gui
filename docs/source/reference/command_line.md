(ref-command-line)=
# Command line

This page lists the commands installed with the package. Each opens the GUI window.

## `alignment-gui`

Opens the GUI on data on your own disk.

```console
alignment-gui [-y YAML]
```

`-y`, `--yaml`
: Path to a session YAML (see {ref}`ref-session-yaml`). The session is loaded when the window
  opens. Without it the window opens empty, and a data folder or session YAML is chosen from the
  data source button (`...`).

## `alignment-gui-ibl`

Opens the GUI on IBL data from the Alyx database, through ONE. Requires the `ibl` extra (see
{ref}`ref-optional-dependencies`); without it the command exits with a message saying how to
install it.

```console
alignment-gui-ibl [-p PID | -c CSV]
```

Without options, the subject, session and probe are chosen from dropdowns in the window.

`-p`, `--pid`
: A probe insertion id. The dropdowns are set to that insertion and its data is loaded when the
  window opens.

`-c`, `--csv`
: A CSV file listing sessions recorded with two channel maps on the same probe: the dense map,
  read through ONE, and the quarter map, read from the local disk. Both are shown side by side
  as two configurations. Cannot be used together with `--pid`.

In this mode uploading an alignment also asks for a quality assessment of the alignment and the
recording (see {ref}`ref-plugins`).

## Logging

Every command logs what it is loading to the terminal, at `INFO` level.
