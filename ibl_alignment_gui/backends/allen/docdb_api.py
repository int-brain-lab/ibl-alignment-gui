"""An injectable client for the Allen Neural Dynamics DocDB.

:class:`DocDB` plays the same role for the Allen/Code Ocean (anatomical) workflow that
``ONE`` plays for the IBL/Alyx workflow: a single instance wraps the DocDB connection and is
injected into the DocDB alignment loader and uploader
(:class:`~ibl_alignment_gui.loaders.alignment_loader.AlignmentLoaderDocDB` and
:class:`~ibl_alignment_gui.loaders.alignment_uploader.AlignmentUploaderDocDB`).

It reads previous alignments from, and writes alignment QC evaluations to, a session's derived
ecephys record. The ``MetadataDbClient`` is built lazily and cached so the connection is reused
across calls, and a pre-built client can be injected to use a fake backend in tests.

This module imports the heavy ``aind``/``boto3``/``aws_requests_auth`` dependencies (the ``allen``
extra: ``pip install ibl_alignment_gui[allen]``) at the top level, so importing it requires them.
To keep the base install lean, importers load it lazily (e.g.
:class:`~ibl_alignment_gui.handlers.probe_handler.ProbeHandlerAllenYaml` imports :class:`DocDB`
inside its ``__init__``). The ``alignment-gui-allen`` launcher checks the extra is installed up
front (see :func:`ibl_alignment_gui.utils.optional.has_allen`), so the import succeeds whenever the
DocDB backend is actually used.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any

import boto3
import requests
from aind_data_access_api.document_db import MetadataDbClient
from aind_data_access_api.helpers.data_schema import get_quality_control_by_id
from aind_data_schema.core.quality_control import (
    QCEvaluation,
    QCMetric,
    QCStatus,
    Stage,
    Status,
)
from aind_data_schema_models.modalities import Modality
from aind_qcportal_schema.metric_value import CurationHistory, CurationMetric
from aws_requests_auth.aws_auth import AWSRequestsAuth

logger = logging.getLogger(__name__)

# The Allen Neural Dynamics metadata API.
API_GATEWAY_HOST = 'api.allenneuraldynamics.org'
DATABASE = 'metadata_index'
COLLECTION = 'data_assets'
ADD_QC_EVALUATION_URL = f'https://{API_GATEWAY_HOST}/v1/add_qc_evaluation'
AWS_REGION = 'us-west-2'
AWS_SERVICE = 'execute-api'


class DocDB:
    """
    Client for reading/writing alignment data to the Allen Neural Dynamics DocDB.

    Parameters
    ----------
    docdb_api_client : MetadataDbClient or None
        A pre-built DocDB client. If None, a default client is created lazily on first use.
    """

    def __init__(self, docdb_api_client: Any | None = None) -> None:
        self._client: Any | None = docdb_api_client

    @property
    def client(self) -> Any:
        """The underlying DocDB client, built lazily and cached on first access."""
        if self._client is None:
            self._client = MetadataDbClient(
                host=API_GATEWAY_HOST,
                database=DATABASE,
                collection=COLLECTION,
            )
        return self._client

    def query_docdb_id(self, session_name: str) -> tuple[str, dict]:
        """
        Return the DocDB id and record for a session's derived ecephys asset.

        Parameters
        ----------
        session_name : str
            The name of the session to query (matched as a regex against
            ``data_description.name``).

        Returns
        -------
        tuple of (str, dict)
            The DocDB ``_id`` and the full record for the most recent matching asset.

        Raises
        ------
        ValueError
            If no derived ecephys record is found for the session.
        """
        response = self.client.retrieve_docdb_records(
            filter_query={
                'data_description.data_level': 'derived',
                'data_description.name': {'$regex': session_name},
                'data_description.modality.abbreviation': 'ecephys',
            }
        )
        if len(response) == 0:
            raise ValueError(f'No ephys sorted record found in docdb for {session_name}')

        # Pull the most recent ephys sorting record.
        latest_record = max(response, key=lambda x: x['created'])

        return latest_record['_id'], latest_record

    def load_alignments(
        self, session_name: str, probe: str, shank_idx: int = 0
    ) -> dict[str, Any] | None:
        """
        Load previously stored alignments for a probe/shank from DocDB.

        Reads the QC evaluation named ``Probe Alignment for {session}_{probe}_{shank_idx}`` from
        the session's derived ecephys record and returns its stored ``previous_alignments``.

        Parameters
        ----------
        session_name : str
            The name of the session for the probe.
        probe : str
            The probe for which the alignment was done.
        shank_idx : int
            Index of the shank (0-based).

        Returns
        -------
        dict or None
            The stored alignments dictionary, or None if no matching evaluation is found.

        Raises
        ------
        ValueError
            If no derived ecephys record is found for the session.
        """
        docdb_id = self.query_docdb_id(session_name)[0]
        quality_control = get_quality_control_by_id(self.client, docdb_id)
        if quality_control is None:
            return None

        evaluation_name = f'Probe Alignment for {session_name}_{probe}_{shank_idx}'
        alignment_evaluations = [
            evaluation
            for evaluation in quality_control.evaluations
            if evaluation.name == evaluation_name
        ]
        if len(alignment_evaluations) == 0:
            logger.info(f'No alignment found in docdb for {session_name}_{probe}_{shank_idx}')
            return None

        logger.info(
            f'Found docdb record for {session_name}_{probe}_{shank_idx}, loading alignment'
        )
        # Pull the latest alignment evaluation.
        latest_alignment_evaluation = max(alignment_evaluations, key=lambda x: x.created)
        curation_metric = latest_alignment_evaluation.metrics[0].value['curations']

        return json.loads(curation_metric[0])['previous_alignments']

    def write_output(
        self,
        session_name: str,
        probe: str,
        channel_results: dict[str, Any],
        previous_alignments: dict[str, Any],
        ccf_channel_results: dict[str, Any],
        curator: str | None = None,
    ) -> None:
        """
        Append an alignment QC evaluation to a session's derived ecephys DocDB record.

        Pulls the latest ephys sorted record and posts a QC evaluation holding the current channel
        results, previous alignments and CCF channel results.

        Parameters
        ----------
        session_name : str
            The name of the session for the probe.
        probe : str
            The probe for which the alignment is being done.
        channel_results : dict
            The current channel results (atlas space) from the GUI.
        previous_alignments : dict
            The stored alignment information.
        ccf_channel_results : dict
            The channel results aligned to the Allen common coordinate framework (CCF).
        curator : str or None
            Name of the curator recorded in the evaluation. Falls back to the ``USERNAME``/``USER``
            environment variables when None.
        """
        docdb_id = self.query_docdb_id(session_name)[0]

        curator = curator or os.getenv('USERNAME') or os.getenv('USER')
        curation_history = CurationHistory(curator=curator, timestamp=datetime.now())
        curations = {
            'channel_results': channel_results,
            'previous_alignments': previous_alignments,
            'ccf_channel_results': ccf_channel_results,
        }
        curation_metric = CurationMetric(
            curations=[json.dumps(curations)],
            curation_history=[curation_history],
        )

        evaluation_name = f'Probe Alignment for {session_name}_{probe}'
        description = 'Probe Alignment of Ephys with Histology'
        qc_metric = QCMetric(
            name=evaluation_name,
            description=description,
            value=curation_metric,
            status_history=[
                QCStatus(
                    evaluator=curation_history.curator,
                    status=Status.PASS,
                    timestamp=datetime.now(),
                )
            ],
        )
        evaluation = QCEvaluation(
            modality=Modality.ECEPHYS,
            stage=Stage.PROCESSING,
            name=evaluation_name,
            description=description,
            metrics=[qc_metric],
        )

        post_request_content = {
            'data_asset_id': docdb_id,
            'qc_evaluation': evaluation.model_dump(mode='json'),
        }
        response = requests.post(
            url=ADD_QC_EVALUATION_URL, auth=self._aws_auth(), json=post_request_content
        )
        if response.status_code != 200:
            logger.error(f'Failed to write {session_name} with {probe} to docdb')
            logger.error(f'HTTP Status Code: {response.status_code}')
            logger.error(f'Response: {response.text}')

    @staticmethod
    def _aws_auth() -> Any:
        """
        Build AWS SigV4 auth for the metadata API gateway from the ambient credentials.

        Built fresh per request (rather than cached) so session tokens do not go stale over a
        long GUI session.

        Returns
        -------
        AWSRequestsAuth
            Auth object for signing requests to the API gateway.
        """
        credentials = boto3.Session().get_credentials()
        return AWSRequestsAuth(
            aws_access_key=credentials.access_key,
            aws_secret_access_key=credentials.secret_key,
            aws_token=credentials.token,
            aws_host=API_GATEWAY_HOST,
            aws_region=AWS_REGION,
            aws_service=AWS_SERVICE,
        )
