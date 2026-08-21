Datasets
========

An overview of the input and output datasets used by the alignment GUI. For instructions on generating these
datasets from your own recordings, see :doc:`data_preparation`.

Each dataset group is stored in its own folder. These folders are declared in the session YAML under the dataset key
described below. See :doc:`usage` for the YAML format and details on how dataset paths are resolved.

When a single data folder is opened directly instead of a YAML file, all datasets are expected to be located within that folder.


.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Datasets
     - YAML key
     - Contents
   * - ``spikes.*``, ``clusters.*``, ``channels.*``
     - ``spike_sorting``
     - Spike sorting output
   * - ``_iblqc_ephysTimeRms*``, ``_iblqc_ephysSpectralDensityLF``
     - ``processed_ephys``
     - RMS and power spectra computed from the raw data
   * - ``*.ap.meta``, ``*.ap.bin``, ``*.lf.bin``
     - ``raw_ephys``
     - Raw SpikeGLX recordings and their metadata
   * - Histology volumes
     - ``histology``
     - Registered histology images and Allen CCF template and annotations
   * - ``xyz_picks``
     - ``picks``
     - Traced probe trajectory
   * - ``channel_locations``, ``prev_alignments``
     - ``output``
     - Written by the GUI when an alignment is uploaded


Spike sorting data
------------------

Read from the ``spike_sorting`` folder.

``spikes.amps``
    :Units: V
    :Type: double
    :Dimensions: [number of spikes]
    :Format: npy
    :Description: Amplitude of each spike.

``spikes.clusters``
    :Units: index
    :Type: int
    :Dimensions: [number of spikes]
    :Format: npy
    :Description: Cluster assignments for each spike (integers counting from 0). Can be used as
        direct indexing of the ``cluster.*`` attributes.

``spikes.depths``
    :Units: µm
    :Type: double
    :Dimensions: [number of spikes]
    :Format: npy
    :Description: Depth along probe of each spike, computed from the waveform centre of mass.
        0 means probe tip, positive upwards.

``spikes.times``
    :Units: seconds
    :Type: float64
    :Dimensions: [number of spikes]
    :Format: npy
    :Description: Times of spikes, relative to experiment onset.

``clusters.channels``
    :Units: index
    :Type: int
    :Dimensions: [number of clusters]
    :Format: npy
    :Description: The channel each cluster is assigned to, used to place the clusters along the
        probe.

``clusters.metrics``
    :Units: N/A
    :Type: N/A
    :Dimensions: [number of clusters]
    :Format: csv or pqt
    :Description: Quality control metrics at the cluster level. The fields may vary and none are
        required, but two of them enable the unit filters: ``ks2_label`` enables the
        ``KS good`` and ``KS mua`` filters, and ``label`` enables the ``IBL good`` filter. If the
        field a filter needs is missing, the GUI logs a warning and shows all units instead.

``clusters.peakToTrough``
    :Units: ms
    :Type: double
    :Dimensions: [number of clusters]
    :Format: npy
    :Description: Template waveform duration for the cluster, the time elapsed from peak to
        trough (``peak_sample - trough_sample``). Can be negative if the peak happens first.

``clusters.waveforms``
    :Units: V
    :Type: single
    :Dimensions: [number of clusters, number of time samples, number of selected channels]
    :Format: npy
    :Description: Waveform from the spike sorting templates, stored as a sparse array for only a
        subset of channels closest to the peak channel.

``channels.localCoordinates``
    :Units: µm
    :Type: float
    :Dimensions: [number of channels, 2]
    :Format: npy
    :Description: Position of channels relative to the probe coordinate system. The x (first)
        dimension is across the width of the shank, the y (second) is the depth, where 0 is the
        deepest site and positive is above it.

``channels.rawInd``
    :Units: index
    :Type: int
    :Dimensions: [number of channels]
    :Format: npy
    :Description: Which index in the raw recording file of its home probe each channel
        corresponds to, counting from zero. Can be used as direct indexing for the raw binary
        files (AP and LF).


Processed electrophysiology data
--------------------------------

Read from the ``processed_ephys`` folder. If no ``processed_ephys`` path is given it falls back to
``raw_ephys``, and then to ``spike_sorting``.

``_iblqc_ephysSpectralDensityLF.power``
    :Units: V**2/Hz
    :Type: float32
    :Dimensions: [number of frequencies, number of channels]
    :Format: npy
    :Description: Power spectral density for all channels of the LF band of the raw data.

``_iblqc_ephysSpectralDensityLF.freqs``
    :Units: Hz
    :Type: float32
    :Dimensions: [number of frequencies]
    :Format: npy
    :Description: Frequencies used to compute the power spectral density.

``_iblqc_ephysTimeRmsAP.rms``
    :Units: V
    :Type: float32
    :Dimensions: [number of time windows, number of channels]
    :Format: npy
    :Description: RMS amplitude of the AP band of the raw data as a function of time.

``_iblqc_ephysTimeRmsAP.timestamps``
    :Units: s
    :Type: float32
    :Dimensions: [number of time windows]
    :Format: npy
    :Description: Time scale for the AP RMS amplitude, relative to the raw binary ephys file.

``_iblqc_ephysTimeRmsLF.rms``
    :Units: V
    :Type: float32
    :Dimensions: [number of time windows, number of channels]
    :Format: npy
    :Description: RMS amplitude of the LF band of the raw data as a function of time.

``_iblqc_ephysTimeRmsLF.timestamps``
    :Units: s
    :Type: float32
    :Dimensions: [number of time windows]
    :Format: npy
    :Description: Time scale for the LF RMS amplitude, relative to the raw binary ephys file.


Raw electrophysiology data
--------------------------

Read from the ``raw_ephys`` folder. These are optional, but without them the raw data snippets
cannot be shown.

``*.ap.meta``
    :Format: SpikeGLX meta
    :Description: SpikeGLX metadata for the AP band. Used for the probe geometry, and as the
        source of the shank count when no ``channels`` object is present, so it is the only way to
        open a session that has no spike sorting.

``*.ap.bin`` and ``*.lf.bin``
    :Format: SpikeGLX binary
    :Description: Raw AP and LF recordings, read in short snippets to display the raw data plots.


Histology data
--------------

Read from the ``histology`` folder. Optional histology images to show the coronal slices in the GUI.

The gui looks for ``.nrrd`` files whose names contain ``RD`` or ``GR``.

``histology_image_RD`` (optional)
    :Dimensions: [voxels in AP, voxels in DV, voxels in ML] for nrrd,
        [voxels in ML, voxels in DV, voxels in AP] for tif
    :Format: nrrd or tif
    :Description: Red channel of the histology stack registered to the 25 µm Allen atlas. Any
        ``.nrrd`` file with ``RD`` in its name is picked up.

``histology_image_GR`` (optional)
    :Dimensions: [voxels in AP, voxels in DV, voxels in ML] for nrrd,
        [voxels in ML, voxels in DV, voxels in AP] for tif
    :Format: nrrd or tif
    :Description: Green channel of the histology stack registered to the 25 µm Allen atlas. Any
        ``.nrrd`` file with ``GR`` in its name is picked up.

.. note::
   If no histology images are provided the alignment can still be done against the Allen atlas
   template and annotations, which are downloaded automatically.


Probe trajectory
----------------

Read from the ``picks`` folder. If no ``picks`` path is given, the ``spike_sorting`` folder is
searched instead.

``xyz_picks``
    :Units: µm
    :Type: double
    :Dimensions: [number of selected points, 3]
    :Format: json
    :Description: Coordinates of the selected points along the traced probe track, with respect
        to bregma. The file is matched as ``*xyz_picks.json``, so a prefix may be added.
    :Example: ``{"xyz_picks": [[-2863, -3999, -643], [-2813, -3975, -842], [-2788, -4025, -1193],
        [-2763, -3949, -1543], [-2688, -4025, -1718], ...]}``


Output data
-----------

Written to the ``output`` folder. If no ``output`` path is given the results are written alongside
the spike sorting, and then alongside the picks.

``channel_locations``
    :Format: json
    :Dimensions: [number of channels]
    :Description: Location of each channel, including its x, y and z coordinate with respect to
        bregma, its position on the probe, and the brain region it falls in.
    :Example:
        .. code-block:: json

           {
             "channel_0": {
               "x": -2453.8710675786924,
               "y": -4189.7375279160615,
               "z": -3415.687639580306,
               "axial": 20.0,
               "lateral": 43.0,
               "brain_region_id": 0,
               "brain_region": "void"
             },
             "channel_1": {
               "x": -2453.8710675786924,
               "y": -4189.7375279160615,
               "z": -3415.687639580306,
               "axial": 20.0,
               "lateral": 11.0,
               "brain_region_id": 0,
               "brain_region": "void"
             }
           }

``prev_alignments``
    :Format: json
    :Dimensions: [number of previous alignments]
    :Description: The reference lines used to align the electrophysiology and histology features,
        keyed by the date and user of each alignment, so that previous alignments can be reloaded.

``alignment_progress``
    :Format: json
    :Description: An alignment saved with **Save Progress** that has not been uploaded yet. It is
        offered in the alignment dropdown under a ``recovered`` key the next time the session is
        loaded, and is deleted once the alignment has been uploaded.


File naming
-----------

Multi-shank probes
~~~~~~~~~~~~~~~~~~

When a probe has more than one shank, the trajectory and the outputs are per shank and carry a
shank suffix, counting from 1 and with no separator before the number:

.. code-block:: text

    xyz_picks_shank1.json
    channel_locations_shank1.json
    prev_alignments_shank1.json
    alignment_progress_shank1.json

