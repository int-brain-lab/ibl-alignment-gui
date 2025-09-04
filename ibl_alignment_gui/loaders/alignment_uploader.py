import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

import ibllib.qc.critical_reasons as critical_note
from iblatlas import atlas
from iblatlas.atlas import AllenAtlas
from ibllib.pipes import histology
from ibllib.qc.alignment_qc import AlignmentQC
from iblutil.util import Bunch
from one import params
from one.api import ONE


class AlignmentUploader(ABC):
    """
    Abstract base class for saving alignment results.

    Subclasses must implement the abstract `upload_data` method.

    Parameters
    ----------
    brain_atlas : AllenAtlas
        An AllenAtlas instance
    """

    def __init__(self, brain_atlas: atlas.AllenAtlas) -> None:

        self.brain_atlas = brain_atlas

    @abstractmethod
    def upload_data(self, *args) -> str:
        """Upload alignment data."""


class AlignmentUploaderOne(AlignmentUploader):
    """
    Alignment uploader using ONE. xyz channels and alignments are saved to Alyx database.

    Parameters
    ----------
    insertion : dict
        Probe insertion information.
    one : ONE
        An ONE instance used to upload results to Alyx
    brain_atlas : AllenAtlas
        An AllenAtlas object.
    """

    def __init__(self, insertion: dict[str, Any], one: ONE, brain_atlas: atlas.AllenAtlas):

        self.one: ONE = one
        self.pid: str = insertion['id']
        self.pname: str = insertion['name']
        self.resolved: bool = insertion['json'].get('extended_qc', {}).get(
            'alignment_resolved', False)
        self.qc_str: str | None = None
        self.confidence_str: str | None = None
        self.user: str = params.get().ALYX_LOGIN
        self.force_resolve: bool = False
        self.align_key: str | None = None

        super().__init__(brain_atlas)

    def upload_data(self, data: dict[str, Any]) -> str:
        """
        Upload channels, alignments, and QC to Alyx.

        Parameters
        ----------
        data : dict
            Alignment and channel data.

        Returns
        -------
        str
            Message containing information about upload result.
        """
        # Upload channels
        is_channels = self.upload_channels(data)
        # Upload alignments
        updated_alignments = self.upload_alignments(data)
        # Update alignment qc
        is_resolved = self.upload_qc(data, updated_alignments)

        return self.get_upload_info(is_channels, is_resolved)

    def get_upload_info(self, channels: bool, resolved: bool) -> str:
        """
        Return an info message based on upload result.

        Parameters
        ----------
        channels : bool
            Whether channels were uploaded.
        resolved : bool
            Where the alignment is resolved.

        Returns
        -------
        str
            Status message.
        """
        if channels and not resolved:
            # Channels saved alignment not resolved
            return (f'Channels locations for {self.pname} saved to Alyx.'
                    '\nAlignment not resolved')
        if channels and resolved:
            # channels saved alignment resolved, writen to flatiron
            return  (f'Channel locations for {self.pname} saved to Alyx.'
                     '\nAlignment resolved and channels datasets written to flatiron')
        if not channels and resolved:
            # alignment already resolved, save alignment but channels not written
            return (f'Channel locations for {self.pname} not saved to Alyx as alignment '
                    f'has already been resolved. \nNew user reference lines have been saved')

        return 'No changes made'

    def upload_channels(self, data: dict[str, Any]) -> bool:
        """
        Upload channel locations to Alyx if not resolved.

        Parameters
        ----------
        data : dict
            A dict containing data for upload.

        Returns
        -------
        bool
            True if channels uploaded, False otherwise.
        """
        if self.resolved:
            return False

        # Create new trajectory and overwrite previous one
        histology.register_aligned_track(
            self.pid, data['xyz_channels'], chn_coords=data['chn_coords'], one=self.one,
            overwrite=True, brain_atlas=self.brain_atlas)

        return True

    def upload_alignments(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Upload alignment data to Alyx.

        Parameters
        ----------
        data : dict
            A dict containing data for upload.

        Returns
        -------
        alignments: dict
            Updated alignments dictionary.
        """
        align_time = datetime.now().replace(second=0, microsecond=0).isoformat()
        self.align_key = f'{align_time}_{self.user}'
        extra_alignment = {self.align_key: [data['feature'], data['track'],
                                            self.qc_str, self.confidence_str]}
        alignments = self._remove_duplicate_users(data['alignments'])
        alignments.update(extra_alignment)
        self.save_alignments(alignments)

        return alignments

    def _remove_duplicate_users(self, alignments: dict[str, Any]) -> dict[str, Any]:
        """
        Remove duplicate alignments for the same user if unresolved.

        Parameters
        ----------
        alignments : dict
            Existing alignments.

        Returns
        -------
        alignments: dict
            Alignments with duplicated user keys removed.
        """
        old_user = [key for key in alignments if self.user in key]
        # Only delete duplicated if trajectory is not resolved
        if len(old_user) > 0 and not self.resolved:
            for old in old_user:
                alignments.pop(old)

        return alignments


    def save_alignments(self, alignments: dict[str, Any]) -> None:
        """
        Save updated alignments to Alyx.

        Parameters
        ----------
        alignments : dict
            Updated alignments.
        """
        # Get the new trajectory and update
        traj = self.one.alyx.rest('trajectories', 'list', probe_insertion=self.pid,
                                  provenance='Ephys aligned histology track', no_cache=True)

        self.one.alyx.rest('trajectories', 'partial_update', id=traj[0]['id'],
                           data={'probe_insertion': self.pid, 'json': alignments})

    def get_user_qc(
            self,
            align_qc: str,
            ephys_qc: str,
            ephys_desc: list[str],
            force_resolve: bool
    ) -> None:
        """
        Set QC and confidence strings, optionally launching critical reasons GUI.

        Parameters
        ----------
        align_qc : str
            Alignment confidence.
        ephys_qc : str
            Ephys QC.
        ephys_desc : list of str
            Description of QC issues.
        force_resolve : bool
            Whether to force the alignment to be resolved.
        """
        ephys_desc_str = 'None' if len(ephys_desc) == 0 else ", ".join(ephys_desc)
        self.qc_str = ephys_qc.upper() + ': ' + ephys_desc_str
        self.confidence_str = f'Confidence: {align_qc}'
        self.force_resolve = force_resolve

        if ephys_qc.upper() == 'CRITICAL':
            critical_note.main_gui(self.pid, reasons_selected=ephys_desc, alyx=self.one.alyx)

    def upload_qc(self, data: dict[str, Any], alignments: dict[str, Any]) -> bool:
        """
        Compute alignment qc and upload evaluation to Alyx.

        Parameters
        ----------
        data : dict
            Data required to run alignment qc.
        alignments : dict
            Dictionary of alignments on which to compute the qc.

        Returns
        -------
        self.resolved: bool
            Alignment resolved bool
        """
        align_qc = AlignmentQC(self.pid, one=self.one, brain_atlas=self.brain_atlas,
                               collection=data['probe_collection'])

        align_qc.load_data(prev_alignments=alignments, xyz_picks=data['xyz_picks'],
                           depths=data['chn_depths'], cluster_chns=data['cluster_chns'],
                           chn_coords=data['chn_coords'])

        if self.force_resolve:
            align_qc.resolve_manual(self.align_key, force=True, upload_flatiron=False)
            self.resolved = True
        else:
            results = align_qc.run(upload_flatiron=False)
            self.resolved = results['alignment_resolved']

        align_qc.update_experimenter_evaluation(prev_alignments=alignments)

        return self.resolved


class AlignmentUploaderLocal(AlignmentUploader):
    """
    Alignment uploader using local file system.

    xyz channels and alignments are saved to json files.

    For single-shank data, save filenames:
        - channel_locations.json
        - prev_alignments.json

    For multi-shank data, expected filenames:
        - channel_locations_shank<N>.json
        - prev_alignments_shank<N>.json

    Parameters
    ----------
    data_path: Path
        The path to the local data folder.
    shank_idx : int
        Index of the shank (0-based).
    n_shanks : int
        Total number of shanks.
    brain_atlas: AllenAtlas
        An AllenAtlas instance
    user: str or None
        Username for tagging alignments.
    """

    def __init__(
            self,
            data_path: Path,
            shank_idx: int,
            n_shanks: int,
            brain_atlas: AllenAtlas,
            user: str | None = None):

        self.data_path: Path = data_path
        self.shank_idx: int = shank_idx
        self.n_shanks: int = n_shanks
        self.user: str | None = user
        self.orig_idx: np.ndarray | None = None
        super().__init__(brain_atlas)

    def upload_data(self, data: dict[str, Any], shank_sites: Bunch[str, Any]) -> str:
        """
        Save channels and alignments to local files.

        Parameters
        ----------
        data : dict
            Alignment and channel data.
        shank_sites : Bunch
            A Bunch object containing the channels that correspond to the shank

        Returns
        -------
        str
            Message containing information about upload result.

        Notes
        -----
        This method sets the following attributes:

        self.orig_idx : np.ndarray
            The original index of the channel in the raw data
        """
        self.orig_idx = shank_sites['orig_idx']
        self.upload_channels(data)
        self.upload_alignments(data)

        return 'Channels locations saved'

    def get_brain_regions(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Get brain regions for each channel based on xyz coordinates.

        Parameters
        ----------
        data : dict
            Alignment and channel data.

        Returns
        -------
        brain_regions : dict
            Information about location of electrode channels in brain atlas
        """
        brain_regions = self.brain_atlas.regions.get(
            self.brain_atlas.get_labels(data['xyz_channels']))
        brain_regions['xyz'] = data['xyz_channels']
        brain_regions['lateral'] = data['chn_coords'][:, 0]
        brain_regions['axial'] = data['chn_coords'][:, 1]
        assert np.unique([len(brain_regions[k]) for k in brain_regions]).size == 1
        return brain_regions

    def get_channels(self, brain_regions: dict[str, Any]) -> dict[str, dict]:
        """
        Create channel dictionary in form to write to json file.

        Parameters
        ----------
        brain_regions: dict
            Information about location of electrode channels in brain atlas

        Returns
        -------
        channels : dict[str, dict]
            Dictionary of dictionaries containing data for each channel

        """
        channel_dict = dict()
        for i in np.arange(brain_regions.id.size):
            channel = {
                'x': np.float64(brain_regions.xyz[i, 0] * 1e6),
                'y': np.float64(brain_regions.xyz[i, 1] * 1e6),
                'z': np.float64(brain_regions.xyz[i, 2] * 1e6),
                'axial': np.float64(brain_regions.axial[i]),
                'lateral': np.float64(brain_regions.lateral[i]),
                'brain_region_id': int(brain_regions.id[i]),
                'brain_region': brain_regions.acronym[i]
            }
            if self.orig_idx is not None:
                channel['original_channel_idx'] = int(self.orig_idx[i])

            data = {'channel_' + str(i): channel}
            channel_dict.update(data)

        bregma = atlas.ALLEN_CCF_LANDMARKS_MLAPDV_UM['bregma'].tolist()
        origin = {'origin': {'bregma': bregma}}
        channel_dict.update(origin)

        return channel_dict

    def upload_alignments(self, data: dict[str, Any]) -> None:
        """
        Update and save alignments to local json file.

        Parameters
        ----------
        data: dict
            Alignment and channel data.
        """
        align_time = datetime.now().replace(second=0, microsecond=0).isoformat()
        align_key = f'{align_time}_{self.user}' if self.user else align_time
        extra_alignment = {align_key: [data['feature'], data['track']]}

        alignments = data['alignments']
        if alignments:
            alignments.update(extra_alignment)
        else:
            alignments = extra_alignment
        # Save the new alignment
        self.save_alignments(alignments)

    def upload_channels(self, data: dict[str, Any]) -> None:
        """
        Get channel locations and save to local json file.

        Parameters
        ----------
        data : dict
            Alignment and channel data.
        """
        brain_regions = self.get_brain_regions(data)
        channels = self.get_channels(brain_regions)
        self.save_channels(channels)


    def save_alignments(self, alignments: dict[str, Any]) -> None:
        """
        Save alignments to local json file.

        Parameters
        ----------
        alignments : dict[str, Any]
            Dictionary of alignment data.
        """
        prev_align_filename = 'prev_alignments.json' if self.n_shanks == 1 else \
            f'prev_alignments_shank{self.shank_idx + 1}.json'

        self._save_json_file(prev_align_filename, alignments)

    def save_channels(self, channels: dict[str, dict]) -> None:
        """
        Save channel locations to local json file.

        Parameters
        ----------
        channels: dict[str, dict]
            Dictionary of dictionaries containing data for each channel
        """
        chan_loc_filename = 'channel_locations.json' if self.n_shanks == 1 else \
            f'channel_locations_shank{self.shank_idx + 1}.json'

        self._save_json_file(chan_loc_filename, channels)

    def _save_json_file(self, file_path: str, json_data: dict[str, Any]) -> None:
        """
        Save data to a json file.

        Parameters
        ----------
        file_path: str
            The name of the json file to save to.
        json_data:
            The data to save to the JSON file. Must be JSON serializable
        """
        with open(self.data_path.joinpath(file_path), "w") as f:
            json.dump(json_data, f, indent=2, separators=(',', ': '))
