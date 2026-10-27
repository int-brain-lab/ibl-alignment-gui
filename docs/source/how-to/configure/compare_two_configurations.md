(howto-compare-two-configurations)=
# How to compare two versions of a recording

This guide shows you how to display two versions of the same recording side by side and align them
together, for example two spike sortings, or recordings with dense and sparse channel maps.

## Steps

1. Put each version under `configurations`, with a name of your choice. Each configuration has its
   own `path` and `probes`:

   ```yaml
   defaults:
     histology:
       path: /path/to/histology

   configurations:
     dense:
       path: /path/to/dense_session
       probes:
         probe00:
           datasets:
             spike_sorting:
               path: alf/probe00/kilosort
             raw_ephys:
               path: raw_ephys_data/probe00
             picks:
               path: alf/probe00/kilosort
             output:
               path: alignment_outputs

     sparse:
       path: /path/to/sparse_session
       probes:
         probe00:
           datasets:
             spike_sorting:
               path: alf/probe00/kilosort
             raw_ephys:
               path: raw_ephys_data/probe00
             picks:
               path: alf/probe00/kilosort
             output:
               path: alignment_outputs
   ```

2. Use the same probe names in both configurations. The names are how the GUI pairs the probes up,
   and loading fails if the two do not list the same probes. Multi-shank probes and several probes
   work in each configuration in the same way as without configurations.

3. Open the session. The two configurations are shown side by side; a reference line placed in one
   is applied to the same shank in the other.

At most two configurations can be loaded at once.

## Related

- {ref}`explanation-shanks-probes-configurations`: how configurations are aligned together
