(howto-configure)=
# Configure sessions

These guides show you how to tell the GUI where a session's data is. Every key you can use is
listed in {ref}`ref-session-yaml`, and {ref}`explanation-shanks-probes-configurations` explains how
shanks, probes and configurations fit together.

There are two ways to point the GUI at your data:

A data folder
: If all the files for a single-shank probe are in one folder, you can open that folder directly,
  with no extra setup.

A session YAML
: For anything else, write a short YAML file that lists where each dataset is. You need one when:

  - the datasets of a probe are in different folders
  - the probe has more than one shank, sorted together or one shank at a time
  - you want to show two versions of the same recording side by side
  - several probes share folders, and you want to avoid repeating the paths

Start with {ref}`howto-open-session`, then pick the guide that matches your data below.

```{toctree}
:maxdepth: 1

open_session
single_probe_yaml
multi_shank_sorted_together
shanks_sorted_separately
compare_two_configurations
share_paths_with_defaults
```
