from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np

from iblatlas import atlas
from one import params

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pathlib import Path

    from ibl_alignment_gui.backends.allen.docdb_api import DocDB
    from ibl_alignment_gui.loaders.transform_loader import TransformLoader
    from iblatlas.atlas import BrainAtlas
    from iblutil.util import Bunch
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

    def __init__(self, brain_atlas: BrainAtlas) -> None:
        self.brain_atlas = brain_atlas

    @abstractmethod
    def upload_data(self, *args, **kwargs) -> str:
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
        self.resolved: bool = (
            insertion['json'].get('extended_qc', {}).get('alignment_resolved', False)
        )
        self.qc_str: str | None = None
        self.confidence_str: str | None = None
        self.user: str = params.get().ALYX_LOGIN
        self.force_resolve: bool = False
        self.align_key: str | None = None

        super().__init__(brain_atlas)

    def upload_data(self, data: dict[str, Any], **kwargs) -> str:
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
            return f'Channels locations for {self.pname} saved to Alyx.\nAlignment not resolved'
        if channels and resolved:
            # channels saved alignment resolved, writen to flatiron
            return (
                f'Channel locations for {self.pname} saved to Alyx.'
                '\nAlignment resolved and channels datasets written to flatiron'
            )
        if not channels and resolved:
            # alignment already resolved, save alignment but channels not written
            return (
                f'Channel locations for {self.pname} not saved to Alyx as alignment '
                f'has already been resolved. \nNew user reference lines have been saved'
            )

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
        if self.resolved and not self.force_resolve:
            return False

        from ibllib.pipes import histology  # noqa: PLC0415

        # Create new trajectory and overwrite previous one
        histology.register_aligned_track(
            self.pid,
            data['xyz_channels'],
            chn_coords=data['chn_coords'],
            one=self.one,
            overwrite=True,
            brain_atlas=self.brain_atlas,
        )

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
        extra_alignment = {
            self.align_key: [data['feature'], data['track'], self.qc_str, self.confidence_str]
        }
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
        traj = self.one.alyx.rest(
            'trajectories',
            'list',
            probe_insertion=self.pid,
            provenance='Ephys aligned histology track',
            no_cache=True,
        )

        self.one.alyx.rest(
            'trajectories',
            'partial_update',
            id=traj[0]['id'],
            data={'probe_insertion': self.pid, 'json': alignments},
        )

    def set_user_qc(
        self, align_qc: str, ephys_qc: str, ephys_desc: list[str], force_resolve: bool
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
        ephys_desc_str = 'None' if len(ephys_desc) == 0 else ', '.join(ephys_desc)
        self.qc_str = ephys_qc.upper() + ': ' + ephys_desc_str
        self.confidence_str = f'Confidence: {align_qc}'
        self.force_resolve = force_resolve

        if ephys_qc.upper() == 'CRITICAL':
            from ibllib.qc import critical_reasons  # noqa: PLC0415

            critical_reasons.main_gui(self.pid, reasons_selected=ephys_desc, alyx=self.one.alyx)

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
        from ibllib.qc import alignment_qc  # noqa: PLC0415

        align_qc = alignment_qc.AlignmentQC(
            self.pid,
            one=self.one,
            brain_atlas=self.brain_atlas,
            collection=data['probe_collection'],
        )

        align_qc.load_data(
            prev_alignments=alignments,
            xyz_picks=data['xyz_picks'],
            depths=data['chn_depths'],
            cluster_chns=data['cluster_chns'],
            chn_coords=data['chn_coords'],
        )

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
    brain_atlas: BrainAtlas
        A BrainAtlas instance (AllenAtlas or BrainAtlasAnatomical)
    user: str or None
        Username for tagging alignments.
    transform_loader: TransformLoader or None
        A TransformLoader used to additionally save channel locations in the Allen CCF
        (used in the anatomical workflow). If None, only the atlas-space channel locations
        are saved.
    """

    def __init__(
        self,
        data_path: Path,
        shank_idx: int,
        n_shanks: int,
        brain_atlas: BrainAtlas,
        user: str | None = None,
        transform_loader: TransformLoader | None = None,
    ):
        self.data_path: Path = data_path
        self.shank_idx: int = shank_idx
        self.n_shanks: int = n_shanks
        self.user: str | None = user
        self.transform_loader: TransformLoader | None = transform_loader
        self.orig_idx: np.ndarray | None = None
        super().__init__(brain_atlas)

    def upload_data(self, data: dict[str, Any], shank_sites: Bunch[str, Any] | None = None) -> str:
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
            self.brain_atlas.get_labels(data['xyz_channels'])
        )
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
                'brain_region': brain_regions.acronym[i],
            }
            if self.orig_idx is not None:
                channel['original_channel_idx'] = int(self.orig_idx[i])

            data = {'channel_' + str(i): channel}
            channel_dict.update(data)

        bregma = atlas.ALLEN_CCF_LANDMARKS_MLAPDV_UM['bregma'].tolist()
        origin = {'origin': {'bregma': bregma}}
        channel_dict.update(origin)

        return channel_dict

    def upload_alignments(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Update and save alignments to local json file.

        Parameters
        ----------
        data: dict
            Alignment and channel data.

        Returns
        -------
        alignments : dict[str, Any]
            The alignments dictionary with the newly added alignment merged in.
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

        return alignments

    def upload_channels(self, data: dict[str, Any]) -> tuple[dict[str, dict], dict[str, dict]]:
        """
        Get channel locations and save to local json file.

        When a TransformLoader is available, the channel locations are additionally warped
        into the Allen CCF and saved to a separate ``channel_locations_ccf`` json file.

        Parameters
        ----------
        data : dict
            Alignment and channel data.

        Returns
        -------
        channels : dict[str, dict]
            The atlas-space channel locations.
        ccf_channels : dict[str, dict]
            The channel locations warped into the Allen CCF, or an empty dict when no transform
            loader is available.
        """
        brain_regions = self.get_brain_regions(data)
        channels = self.get_channels(brain_regions)
        self.save_channels(channels)

        if self.transform_loader is not None and self.transform_loader.exists:
            ccf_channels = self.get_ccf_channels(brain_regions, data['xyz_channels'])
            self.save_channels(ccf_channels, suffix='_ccf')

            return channels, ccf_channels

        return channels, {}

    def get_ccf_channels(
        self, brain_regions: dict[str, Any], xyz_channels: np.ndarray
    ) -> dict[str, dict]:
        """
        Create a channel dictionary with channel locations warped into the Allen CCF.

        Mirrors :meth:`get_channels` but replaces the atlas-space x/y/z coordinates with the
        CCF coordinates returned by the transform loader. The CCF coordinates are stored in
        the native units of the registration output (not scaled to microns), and the bregma
        origin is omitted, as the registration target defines its own coordinate system.

        Parameters
        ----------
        brain_regions: dict
            Information about location of electrode channels in brain atlas.
        xyz_channels: np.ndarray
            An (N, 3) array of channel locations in the atlas physical space (RAS, metres).

        Returns
        -------
        channels : dict[str, dict]
            Dictionary of dictionaries containing CCF data for each channel.
        """
        ccf_xyz = self.transform_loader.transform_to_ccf(xyz_channels, self.brain_atlas)

        channel_dict = dict()
        for i in np.arange(brain_regions.id.size):
            channel = {
                'x': np.float64(ccf_xyz[i, 0]),
                'y': np.float64(ccf_xyz[i, 1]),
                'z': np.float64(ccf_xyz[i, 2]),
                'axial': np.float64(brain_regions.axial[i]),
                'lateral': np.float64(brain_regions.lateral[i]),
                'brain_region_id': int(brain_regions.id[i]),
                'brain_region': brain_regions.acronym[i],
            }
            if self.orig_idx is not None:
                channel['original_channel_idx'] = int(self.orig_idx[i])

            channel_dict.update({'channel_' + str(i): channel})

        return channel_dict

    def save_alignments(self, alignments: dict[str, Any]) -> None:
        """
        Save alignments to local json file.

        Parameters
        ----------
        alignments : dict[str, Any]
            Dictionary of alignment data.
        """
        prev_align_filename = (
            'prev_alignments.json'
            if self.n_shanks == 1
            else f'prev_alignments_shank{self.shank_idx + 1}.json'
        )

        self._save_json_file(prev_align_filename, alignments)

    def save_channels(self, channels: dict[str, dict], suffix: str = '') -> None:
        """
        Save channel locations to local json file.

        Parameters
        ----------
        channels: dict[str, dict]
            Dictionary of dictionaries containing data for each channel
        suffix: str
            Suffix appended to the ``channel_locations`` filename stem (e.g. ``'_ccf'`` for
            channel locations in the Allen CCF). Empty by default.
        """
        chan_loc_filename = (
            f'channel_locations{suffix}.json'
            if self.n_shanks == 1
            else f'channel_locations{suffix}_shank{self.shank_idx + 1}.json'
        )

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
        with open(self.data_path.joinpath(file_path), 'w') as f:
            json.dump(json_data, f, indent=2, separators=(',', ': '))


class AlignmentUploaderDocDB(AlignmentUploaderLocal):
    """
    Alignment uploader for the Allen/Code Ocean (anatomical) workflow with DocDB support.

    Extends :class:`AlignmentUploaderLocal`: the local json files (channel locations, previous
    alignments and, when a transform loader is available, the CCF channel locations) are always
    written. When ``use_db`` is True a QC evaluation holding the channel results, previous
    alignments and CCF channel results is additionally posted to DocDB via the injected
    :class:`~ibl_alignment_gui.backends.allen.docdb_api.DocDB` client; when False only the local
    files are written and the uploader behaves like :class:`AlignmentUploaderLocal`.

    Parameters
    ----------
    data_path : Path
        The path to the local data folder.
    shank_idx : int
        Index of the shank (0-based).
    n_shanks : int
        Total number of shanks.
    brain_atlas : BrainAtlas
        A BrainAtlas instance (AllenAtlas or BrainAtlasAnatomical).
    docdb : DocDB
        The DocDB client used to post the QC evaluation (injected, analogous to ``one``).
    user : str or None
        Username for tagging alignments and recorded as the DocDB curator.
    transform_loader : TransformLoader or None
        A TransformLoader used to warp channel locations into the Allen CCF. When available, the
        CCF channel results are included in the DocDB record.
    use_db : bool
        Whether to post results to DocDB (True) in addition to writing the local files, or write
        only the local files (False).
    """

    def __init__(
        self,
        data_path: Path,
        shank_idx: int,
        n_shanks: int,
        brain_atlas: BrainAtlas,
        docdb: DocDB,
        user: str | None = None,
        transform_loader: TransformLoader | None = None,
        use_db: bool = True,
    ):
        self.docdb: DocDB = docdb
        self.use_db: bool = use_db
        super().__init__(
            data_path,
            shank_idx,
            n_shanks,
            brain_atlas,
            user=user,
            transform_loader=transform_loader,
        )

    def upload_data(self, data: dict[str, Any], shank_sites: Bunch[str, Any] | None = None) -> str:
        """
        Save channels and alignments locally, then post to DocDB when ``use_db`` is set.

        Parameters
        ----------
        data : dict
            Alignment and channel data.
        shank_sites : Bunch
            A Bunch object containing the channels that correspond to the shank.

        Returns
        -------
        str
            Message describing the upload result.
        """
        self.orig_idx = shank_sites['orig_idx']
        channels, ccf_channels = self.upload_channels(data)
        alignments = self.upload_alignments(data)

        session_name = self.data_path.parent.stem
        probe = f'{self.data_path.stem}_{self.shank_idx}'

        if self.use_db:
            self.docdb.write_output(
                session_name,
                probe,
                channels,
                alignments,
                ccf_channels,
                curator=self.user,
            )

        return 'Channels locations saved'
