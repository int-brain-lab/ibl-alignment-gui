(howto-open-session)=
# How to open a session

This guide shows you how to load a session into the GUI, from inside it or straight from the command
line.

## From the GUI

1. Click the data source button (`...`) in the top right corner.
2. Choose one of:

   **Open data folder…**
   : A single folder holding every dataset of the session. The data loads as soon as the folder is
     chosen.

   **Open session YAML…**
   : A session YAML listing where each dataset is (see {ref}`howto-single-probe-yaml`).

A different session can be opened the same way at any time, without restarting the GUI.

## From the command line

Pass the session YAML with `-y`, and the session loads when the window opens:

```console
alignment-gui -y /path/to/session.yaml
```

## Related

- {ref}`ref-command-line`
