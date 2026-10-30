"""
Automatic probe alignment with the Ephys Atlas channel-level model (channel-prediction plugin).

The model and the alignment method live in ``ephysatlas.alignment`` (ibleatools); this module is
only their GUI. The released model is loaded with ``ephysatlas.load_pretrained`` (through
``ephysatlas.alignment.ChannelModel``) from the Hugging Face Hub or a local release, and run with
its ``predict``. The shank's recorded channel features are then warped along its histology track
(the GUI's ``xyz_samples``, top to bottom of the brain) onto the model's predictions along it,
behind a progress dialog. The result figure opens in a pop-up window, and the prediction panel
shows either the atlas region of each channel where the alignment placed it, or the alignment
confidence along the probe.

Each shank is aligned once: both prediction-panel views -- the predicted regions and the alignment
confidence, two prediction options -- are built from the alignment straight away and cached by
the plugin, so switching between them, or selecting one again, does not re-align. The result
window opens once and is kept in the plugin state; the 'Show auto-alignment results' menu item
shows it again.

``ephysatlas`` (and torch) are imported only when the model is loaded or a shank is aligned, so
this module can be imported without them, like :mod:`inference`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from qtpy import QtWidgets

from ibl_alignment_gui.app.load_worker import TaskCancelledError
from ibl_alignment_gui.plugins.ephys_atlas._common import (
    AUTO_ALIGNMENT_CONFIDENCE_KEY,
    AUTO_ALIGNMENT_KEY,
    NO_FEATURES_MESSAGE,
    ModelSpec,
    PredictionError,
    alignment_windows,
    clear_predictions,
    features_for_vintage,
    features_vintage,
    missing_features_error,
    model_vintage,
    update_results_action,
)
from ibl_alignment_gui.utils.helpers import shank_loop
from iblutil.util import Bunch

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd
    from ephysatlas.alignment import ChannelModel

    from ibl_alignment_gui.app.controllers.app_controller import AlignmentGUIController
    from ibl_alignment_gui.app.controllers.shank_controller import ShankController

logger = logging.getLogger(__name__)

MODEL_NAME = 'Encoding'  # plugin-state key the loaded model is cached under
# Confidence colouring: p(good) is rounded to this step so neighbouring channels of similar
# confidence merge into one segment; channels without a value are drawn grey.
CONFIDENCE_STEP = 0.05
NO_VALUE_RGB = (170, 170, 170)
# Explanation shown in the alignment model's load dialog.
ENCODER_NOTE = (
    'The model is loaded with ephysatlas.load_pretrained, downloaded once and cached '
    '(~/.cache/ephysatlas/models). It ships its molecular-context volumes and training '
    'neighbour bank, so no other data is needed.'
)


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------
def _default_source() -> tuple[str, str]:
    """Return the alignment model's default repository and release tag, from ephysatlas."""
    from ephysatlas.alignment import CHANNEL_MODEL_REPO_ID, VINTAGE  # noqa: PLC0415 - heavy, lazy

    return CHANNEL_MODEL_REPO_ID, VINTAGE


def _load_encoder(source: tuple[str, str], progress) -> ChannelModel:
    """Load an alignment model release, with its probe-confidence model.

    Parameters
    ----------
    source : tuple of (str, str)
        ``(repo_id_or_directory, revision)``: a Hugging Face repository and release tag, or a
        local release directory.
    progress : Callable
        ``progress(fraction, message)``, called as loading proceeds.

    Returns
    -------
    ChannelModel
        The loaded model.
    """
    from ephysatlas.alignment import ChannelModel  # noqa: PLC0415 - heavy, lazy

    progress(0.1, f'Loading {source[0]}@{source[1]} (downloaded on first use)…')
    model = ChannelModel(*source)  # ephysatlas.load_pretrained(repo_id, revision=...)
    progress(0.8, 'Loading the probe-confidence model…')
    _ = model.conf_model
    progress(1.0, 'Model ready')
    return model


@shank_loop
def invalidate_predictions(
    controller: AlignmentGUIController, items: ShankController, **kwargs
) -> None:
    """Forget a shank's auto-alignment views so the next selection re-aligns it.

    Used after a new model or features file; the shank's result window is closed.
    """
    clear_predictions(items, AUTO_ALIGNMENT_KEY, AUTO_ALIGNMENT_CONFIDENCE_KEY)
    window = alignment_windows(controller).pop((items.name, items.config), None)
    if window is not None:
        window.close()
    update_results_action(controller)


# -----------------------------------------------------------------------------
# Result windows
# -----------------------------------------------------------------------------
class AlignmentResultWindow(QtWidgets.QDialog):
    """A non-modal window with the alignment result figure, a toolbar and a save button.

    Closing only hides it, so the prediction option can show it again.
    """

    def __init__(self, parent, figure, title: str):
        super().__init__(parent)
        from matplotlib.backends.backend_qtagg import (  # noqa: PLC0415 - Qt backend on demand
            FigureCanvasQTAgg,
            NavigationToolbar2QT,
        )

        self.setWindowTitle(title)
        self.resize(1500, 800)
        self.figure = figure
        canvas = FigureCanvasQTAgg(figure)
        save = QtWidgets.QPushButton('Save figure…')
        save.clicked.connect(self._save)
        close = QtWidgets.QPushButton('Close')
        close.clicked.connect(self.close)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(save)
        buttons.addWidget(close)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(NavigationToolbar2QT(canvas, self))
        layout.addWidget(canvas, stretch=1)
        layout.addLayout(buttons)
        canvas.draw()

    def _save(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save alignment figure', 'alignment.png', 'Images (*.png *.pdf *.svg)'
        )
        if path:
            self.figure.savefig(path, dpi=200)


def _open_result_window(
    controller: AlignmentGUIController, items: ShankController, result, probe: dict
) -> None:
    """Draw the shank's alignment result in a new window, keep it and show it once.

    The window is kept in :func:`alignment_windows` under ``(shank, config)``; afterwards it is
    shown again only from the 'Show auto-alignment results' menu item, which this enables.
    """
    from ephysatlas.alignment.plotting import plot_alignment_result  # noqa: PLC0415
    from matplotlib.figure import Figure  # noqa: PLC0415

    # Only what a new recording has: the histology trace, the planned trajectory and the
    # inferred alignment (no reference alignment, no region accuracy).
    fig = plot_alignment_result(
        result,
        controller.model.brain_atlas,
        planned_xyz=probe.get('planned_xyz'),
        fig=Figure(figsize=(17, 8.5)),
    )
    window = AlignmentResultWindow(
        controller.view, fig, f'{result.label} -- {result.pid or "probe"}'
    )
    alignment_windows(controller)[items.name, items.config] = window
    update_results_action(controller)
    window.show()


# -----------------------------------------------------------------------------
# Probe inputs and the alignment
# -----------------------------------------------------------------------------
def _current_pid(controller, items) -> str:
    for obj in (items.model, controller.model):
        for attr in ('pid', 'probe_id'):
            val = getattr(obj, attr, None)
            if val is not None:
                return str(val)
    return ''


def alignment_inputs(model: ChannelModel, df: pd.DataFrame) -> dict:
    """Turn a features table into the alignment's inputs, without the GUI or a shank.

    Parameters
    ----------
    model : ChannelModel
        The loaded alignment model, for its features and vintage.
    df : pandas.DataFrame
        The probe's features, with an ``axial_um`` column.

    Returns
    -------
    dict
        ``features`` (the model's features, rows top of the probe first, non-finite values set to
        0), ``planned_xyz`` (the planned trajectory, or None if the table has none) and
        ``axial_um``.

    Raises
    ------
    PredictionError
        If the table lacks a feature the model needs.
    """
    missing = [c for c in model.features if c not in df.columns]
    if missing:
        raise missing_features_error(
            missing, model_vintage(getattr(model.encoder, 'index', None)), features_vintage(df)
        )
    df = df.sort_values(
        ['axial_um', 'channel'] if 'channel' in df else ['axial_um'], ascending=False
    ).reset_index(drop=True)
    features = df[model.features].to_numpy(dtype=np.float32, copy=True)
    features[~np.isfinite(features)] = 0.0
    planned = (
        df[['x_target', 'y_target', 'z_target']].to_numpy(dtype=np.float32)
        if {'x_target', 'y_target', 'z_target'} <= set(df.columns)
        else None
    )
    return dict(
        features=features, planned_xyz=planned, axial_um=df['axial_um'].to_numpy(dtype=float)
    )


def _align(
    controller: AlignmentGUIController,
    items: ShankController,
    model: ChannelModel,
    progress: Callable[[float | None, str], None],
) -> tuple[object, dict]:
    """Align the shank with a loaded model, without Qt.

    The features come from the table of the model's vintage, as for the region decoders
    (:func:`features_for_vintage`).

    Returns
    -------
    AlignmentResult
        The alignment, from ``ephysatlas.alignment.align_histology``.
    dict
        Its inputs (:func:`alignment_inputs`) and the insertion's ``pid``.

    Raises
    ------
    PredictionError
        If the model's release names no feature vintage, the probe has no usable features or the
        alignment fails.
    TaskCancelledError
        If Cancel was pressed, at the alignment's next progress report.
    """
    vintage = model_vintage(getattr(model.encoder, 'index', None))
    df = features_for_vintage(controller, items, vintage, progress)
    if df is None:
        raise PredictionError(NO_FEATURES_MESSAGE)
    probe = alignment_inputs(model, df)
    # The model leaves this insertion out of the neighbours it predicts from.
    probe['pid'] = _current_pid(controller, items)
    # GUI track: bottom -> top along the histology picks, 10 µm apart, top to bottom of the brain.
    track_gui = np.asarray(items.model.align_handle.xyz_samples, dtype=np.float32)
    from ephysatlas.alignment import align_histology  # noqa: PLC0415 - heavy, lazy

    try:
        result = align_histology(
            model, probe['features'], track_gui[::-1].copy(), pid=probe['pid'], progress=progress
        )
    except TaskCancelledError:
        raise
    except Exception as exc:  # e.g. too few usable channels: reported to the user
        raise PredictionError(str(exc)) from exc

    return result, probe


def align_shank(
    controller: AlignmentGUIController,
    items: ShankController,
    model: ChannelModel,
    progress: Callable[[float | None, str], None],
) -> tuple[dict[str, Bunch], tuple]:
    """Align the shank and return both auto-alignment views, and what its result window needs.

    The alignment model's ``compute`` (see :class:`ModelSpec`): no Qt is used, so it can run on a
    background thread; the result window is opened afterwards by :func:`_show_result`. Nothing is
    kept here: the Channel Prediction plugin caches the views, so the shank is aligned again only
    after they are cleared (:func:`invalidate_predictions`).

    Returns
    -------
    dict of {str: Bunch}
        The regions view (:func:`predict_regions`) under ``'Auto-alignment'`` and the confidence
        view (:func:`predict_confidence`) under ``'Auto-alignment (confidence)'``.
    tuple
        The alignment result and its inputs, for the result window.

    Raises
    ------
    PredictionError
        If the probe has no usable features or the alignment fails.
    TaskCancelledError
        If Cancel was pressed.
    """
    result, probe = _align(controller, items, model, progress)
    views = {
        AUTO_ALIGNMENT_KEY: predict_regions(
            controller.model.brain_atlas, result.channel_xyz, probe['axial_um']
        ),
        AUTO_ALIGNMENT_CONFIDENCE_KEY: predict_confidence(
            result.p_good, result.valid, probe['axial_um']
        ),
    }

    return views, (result, probe)


def _show_result(controller: AlignmentGUIController, items: ShankController, extra) -> None:
    """Open the shank's result window, on the main thread (the alignment model's on_computed)."""
    result, probe = extra
    _open_result_window(controller, items, result, probe)


# -----------------------------------------------------------------------------
# The prediction-panel views: predicted regions and confidence
# -----------------------------------------------------------------------------
def _depth_edges(levels: np.ndarray) -> np.ndarray:
    """Return the edges (µm) of probe depths: halfway between them, 10 µm past the ends."""
    return np.r_[levels[0] - 10.0, 0.5 * (levels[1:] + levels[:-1]), levels[-1] + 10.0]


def _first_most_common(ids: np.ndarray):
    """Return the most common of ``ids``; on a tie, the one that comes first."""
    values, first, counts = np.unique(ids, return_index=True, return_counts=True)
    tied = counts == counts.max()
    return values[tied][np.argmin(first[tied])]


def predict_regions(brain_atlas, channel_xyz: np.ndarray, axial_um: np.ndarray) -> Bunch:
    """Return the regions view: the atlas region of each channel where the alignment placed it.

    The view covers the probe only. A Neuropixels probe has two sites per depth, which can land
    either side of a region boundary, so each depth takes the region most of its sites are in (on a
    tie, the region of its first site in probe order).

    Parameters
    ----------
    brain_atlas : AllenAtlas
        The GUI's brain atlas.
    channel_xyz : np.ndarray
        ``[C, 3]`` position (m) of each channel on the histology track, from the alignment.
    axial_um : np.ndarray
        ``[C]`` probe depth (µm) of each channel.

    Returns
    -------
    Bunch
        The region boundaries, labels and colours along the probe, for ``plot_prediction``.
    """
    depth = np.asarray(axial_um, dtype=float)
    # Allen region ids, as ``regions.get`` expects (not region-table indices).
    ids = np.asarray(brain_atlas.get_labels(np.asarray(channel_xyz), mapping='Allen'))
    ok = np.isfinite(depth)
    depth, ids = depth[ok], ids[ok]
    levels = np.unique(depth)  # one per probe depth, tip first
    level_ids = np.array([_first_most_common(ids[depth == d]) for d in levels])

    # One segment per run of the same region, its edges halfway between neighbouring depths
    edges = _depth_edges(levels)
    starts = np.r_[0, np.flatnonzero(np.diff(level_ids)) + 1]
    ends = np.r_[starts[1:], len(levels)]
    regions = brain_atlas.regions.get(level_ids[starts])
    region = np.c_[edges[starts], edges[ends]]
    axis_label = np.empty((len(starts), 2), dtype=object)
    axis_label[:, 0] = region.mean(axis=1)
    axis_label[:, 1] = regions.acronym

    return Bunch(region=region, axis_label=axis_label, colour=np.asarray(regions.rgb, dtype=int))


def predict_confidence(p_good: np.ndarray, valid: np.ndarray, axial_um: np.ndarray) -> Bunch:
    """Return the confidence view: the alignment confidence p(good) along the probe.

    Green is high and red low confidence (``RdYlGn``); channels without a value are grey. Depths
    are the channels' probe depths (µm), and the view covers the probe only, like the regions view.

    Parameters
    ----------
    p_good : np.ndarray
        ``[C]`` alignment confidence of each channel.
    valid : np.ndarray
        ``[C]`` whether each channel was aligned; channels that were not have no value.
    axial_um : np.ndarray
        ``[C]`` probe depth (µm) of each channel.

    Returns
    -------
    Bunch
        The segments (``region``), their ``colour`` and ``axis_label``, without boundary lines,
        for ``plot_prediction``.
    """
    import matplotlib as mpl  # noqa: PLC0415 - only needed here

    depth = np.asarray(axial_um, dtype=float)
    p_good = np.where(valid, p_good, np.nan)
    # One value per probe depth (Neuropixels has two sites per depth), tip first.
    levels = np.unique(depth[np.isfinite(depth)])
    p = np.array(
        [
            np.nanmean(p_good[depth == d]) if np.isfinite(p_good[depth == d]).any() else np.nan
            for d in levels
        ]
    )
    edges = _depth_edges(levels)

    cmap = mpl.colormaps['RdYlGn']
    q = np.where(np.isfinite(p), np.round(p / CONFIDENCE_STEP) * CONFIDENCE_STEP, np.nan)
    region, colour, labels = [], [], []
    start = 0
    for i in range(1, len(levels) + 1):
        if i < len(levels) and (q[i] == q[start] or (np.isnan(q[i]) and np.isnan(q[start]))):
            continue
        value = q[start]
        region.append((edges[start], edges[i]))
        colour.append(
            NO_VALUE_RGB
            if np.isnan(value)
            else tuple(int(255 * c) for c in cmap(float(value))[:3])
        )
        start = i
    # Axis labels: the mean confidence every ~500 µm along the probe.
    for lo in np.arange(levels[0], levels[-1], 500.0):
        window = (levels >= lo) & (levels < lo + 500.0)
        if np.isfinite(p[window]).any():
            labels.append((lo + 250.0, f'p {np.nanmean(p[window]):.2f}'))
    return Bunch(
        region=np.asarray(region, dtype=float),
        axis_label=np.asarray(labels, dtype=object).reshape(-1, 2),
        colour=np.asarray(colour, dtype=int),
        boundaries=False,
    )


# -----------------------------------------------------------------------------
# Model spec
# -----------------------------------------------------------------------------
# The alignment model, as the shared load dialog and model cache need it.
ENCODER_SPEC = ModelSpec(
    key=MODEL_NAME,
    title='Load alignment model',
    default_source=_default_source,
    note=ENCODER_NOTE,
    load=_load_encoder,
    invalidate=invalidate_predictions,
    options=(AUTO_ALIGNMENT_KEY, AUTO_ALIGNMENT_CONFIDENCE_KEY),
    compute=align_shank,
    needs_torch=True,
    model_class='NeighborInpaintingModel',
    on_computed=_show_result,
)
