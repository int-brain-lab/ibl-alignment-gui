(howto-share-paths-with-defaults)=
# How to avoid repeating paths in a session YAML

This guide shows you how to shorten a session YAML whose probes share folders or folder names.

## Steps

1. Move the parts every path shares into base paths. A `path` at the top of the file, on a
   configuration or on a probe is prepended to the relative dataset paths below it:

   ```yaml
   path: /mnt/data/subject01/2025-01-15
   probes:
     probe00:
       path: probe00
       datasets:
         spike_sorting:
           path: kilosort       # /mnt/data/subject01/2025-01-15/probe00/kilosort
   ```

2. Move folders shared by every probe, such as the histology, into `defaults` as absolute paths.

3. Move folder names that are the same for every probe into `defaults` as relative paths. Each
   probe then resolves them against its own base path, and can leave the dataset out or give it as
   `{}`:

   ```yaml
   path: /mnt/data/subject01/2025-01-15

   defaults:
     histology:
       path: /mnt/data/subject01/histology   # absolute: the same folder for every probe
     spike_sorting:
       path: kilosort                        # relative: joined onto each probe's base path
     raw_ephys:
       path: spikeglx
     picks:
       path: picks

   probes:
     probe00:
       path: probe00
       # spike_sorting: /mnt/data/subject01/2025-01-15/probe00/kilosort
       # raw_ephys:     /mnt/data/subject01/2025-01-15/probe00/spikeglx
       # picks:         /mnt/data/subject01/2025-01-15/probe00/picks
       # histology:     /mnt/data/subject01/histology
       datasets:
         output:
           path: alignment   # /mnt/data/subject01/2025-01-15/probe00/alignment
     probe01:
       path: probe01
       # spike_sorting: /mnt/data/subject01/2025-01-15/probe01/kilosort
       # raw_ephys:     /mnt/data/subject01/2025-01-15/probe01/spikeglx
       # picks:         /mnt/data/subject01/2025-01-15/probe01/picks
       # histology:     /mnt/data/subject01/histology
       datasets:
         output:
           path: alignment   # /mnt/data/subject01/2025-01-15/probe01/alignment
   ```

4. Keep a dataset's own `path` only where a probe differs from the default. A probe's own relative
   path replaces the default; it is not joined onto it.

## Related

- {ref}`explanation-session-yaml-resolution`: the full rules, with worked examples
