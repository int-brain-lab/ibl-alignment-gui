"""
Automatic probe alignment with the Ephys Atlas channel-level model (channel-prediction plugin).

The model and the alignment method live in ``ephysatlas.alignment`` (ibleatools); this module is
only their GUI. The released model is loaded with ``ephysatlas.load_pretrained`` (through
``ephysatlas.alignment.ChannelModel``) from the Hugging Face Hub or a local release, and run with
its ``predict``. The shank's recorded channel features are then warped along its histology track
(the GUI's ``xyz_samples``, top to bottom of the brain) onto the model's predictions along it,
behind a progress dialog. The result figure opens in a pop-up window, and the region panel shows
either the predicted regions or the alignment confidence along the probe.

Each shank's alignment is computed once and kept, so selecting the region option again re-opens its
window (when closed) and switching the colouring redraws the column without re-aligning.

The heavy ``torch``/``ephysatlas`` imports load with this module, so the plugin imports it lazily.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from ephysatlas.alignment import CHANNEL_MODEL_REPO_ID, VINTAGE, ChannelModel, align_histology
from ephysatlas.alignment.data import load_probe
from ephysatlas.spatial_encoder.utils import FEATURE_LIST
from qtpy import QtCore, QtWidgets

from ibl_alignment_gui.plugins.ephys_atlas._common import (
    AUTO_ALIGNMENT_KEY,
    clear_predictions,
    features_for_vintage,
    has_features,
    has_local_features,
    plugin_state,
)
from ibl_alignment_gui.utils.helpers import shank_loop
from iblutil.util import Bunch

if TYPE_CHECKING:
    from collections.abc import Callable

    from ibl_alignment_gui.app.controllers.app_controller import AlignmentGUIController
    from ibl_alignment_gui.app.controllers.shank_controller import ShankController

logger = logging.getLogger(__name__)

MODEL_NAME = 'Encoding'  # plugin-state key the loaded model is cached under
PREDICTION_KEY = AUTO_ALIGNMENT_KEY
# Confidence colouring: p(good) is rounded to this step so neighbouring channels of similar
# confidence merge into one segment; channels without a value are drawn grey.
CONFIDENCE_STEP = 0.05
NO_VALUE_RGB = (170, 170, 170)


class AlignmentCancelledError(Exception):
    """Raised from the progress callback when the user presses Cancel."""


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------
class _ModelDialog(QtWidgets.QDialog):
    """Choose the released channel-level model: a Hugging Face repo + tag, or a local release."""

    def __init__(self, parent, repo_id: str, revision: str, local_dir: Path | None):
        super().__init__(parent)
        self.setWindowTitle('Load alignment model')
        self.setMinimumWidth(560)
        form = QtWidgets.QFormLayout()
        self._repo = QtWidgets.QLineEdit(repo_id)
        self._revision = QtWidgets.QLineEdit(revision)
        form.addRow('Hugging Face repository', self._repo)
        form.addRow('Release tag', self._revision)

        local_row = QtWidgets.QHBoxLayout()
        self._local = QtWidgets.QLineEdit('' if local_dir is None else str(local_dir))
        self._local.setPlaceholderText('optional: a local release directory instead of the Hub')
        browse = QtWidgets.QPushButton('Browse…')
        browse.clicked.connect(self._browse)
        local_row.addWidget(self._local)
        local_row.addWidget(browse)
        form.addRow('Local release', local_row)

        note = QtWidgets.QLabel(
            'The model is loaded with ephysatlas.load_pretrained, downloaded once and cached '
            '(~/.cache/ephysatlas/models). It ships its molecular-context volumes and training '
            'neighbour bank, so no other data is needed.'
        )
        note.setWordWrap(True)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(note)
        layout.addWidget(buttons)

    def _browse(self) -> None:
        chosen = QtWidgets.QFileDialog.getExistingDirectory(
            self, 'Select a local release directory'
        )
        if chosen:
            self._local.setText(chosen)

    def source(self) -> tuple[str, str]:
        """``(repo_id_or_directory, revision)`` to load."""
        local = self._local.text().strip()
        if local:
            return local, self._revision.text().strip() or VINTAGE
        return (
            self._repo.text().strip() or CHANNEL_MODEL_REPO_ID,
            self._revision.text().strip() or VINTAGE,
        )


def load_model_dialog(controller: AlignmentGUIController) -> bool:
    """Ask which release to use and load it (only when the source changed).

    Returns
    -------
    bool
        True when a model is loaded, False when the user cancelled or loading failed.
    """
    plugin = plugin_state(controller)
    state = plugin.get(MODEL_NAME) or dict(model=None, source=None)
    plugin[MODEL_NAME] = state
    if not has_features(controller):
        QtWidgets.QMessageBox.warning(
            controller.view,
            'Channel Prediction',
            'No features found for this probe. '
            'Set via Plugins -> Channel Prediction -> Load features file...',
        )
        return False
    repo, revision = state['source'] or (CHANNEL_MODEL_REPO_ID, VINTAGE)
    local = Path(repo) if Path(str(repo)).is_dir() else None
    dialog = _ModelDialog(
        controller.view, CHANNEL_MODEL_REPO_ID if local else repo, revision, local
    )
    if dialog.exec() != QtWidgets.QDialog.Accepted:
        return state['model'] is not None
    source = dialog.source()
    if state['model'] is not None and state['source'] == source:
        return True
    try:
        model = run_with_progress(
            controller,
            'Loading alignment model',
            lambda progress: _load(source, progress),
            cancellable=False,
        )
    except Exception as exc:  # noqa: BLE001 - show any loading problem to the user
        QtWidgets.QMessageBox.critical(
            controller.view,
            'Load alignment model',
            f'Could not load {source[0]}@{source[1]}:\n{exc}',
        )
        return False
    state.update(model=model, source=source)
    invalidate_predictions(controller)
    return True


def _load(source: tuple[str, str], progress) -> ChannelModel:
    progress(0.1, f'Loading {source[0]}@{source[1]} (downloaded on first use)…')
    model = ChannelModel(*source)  # ephysatlas.load_pretrained(repo_id, revision=...)
    progress(0.8, 'Loading the probe-confidence model…')
    _ = model.conf_model
    progress(1.0, 'Model ready')
    return model


def get_model(controller: AlignmentGUIController) -> ChannelModel | None:
    """Return the loaded model, loading it first when needed (None if the user cancelled).

    A ``source`` pre-set on the plugin state (``(repo_id_or_dir, revision)``, e.g. by a launcher
    script) loads without the dialog; otherwise the load dialog opens.
    """
    state = plugin_state(controller).get(MODEL_NAME)
    if state is not None and state.get('model') is None and state.get('source'):
        try:
            state['model'] = run_with_progress(
                controller,
                'Loading alignment model',
                lambda progress: _load(tuple(state['source']), progress),
                cancellable=False,
            )
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(controller.view, 'Load alignment model', str(exc))
            return None
    if state is None or state.get('model') is None:
        if not load_model_dialog(controller):
            return None
        state = plugin_state(controller).get(MODEL_NAME)
    return state['model']


@shank_loop
def invalidate_predictions(
    controller: AlignmentGUIController, items: ShankController, **kwargs
) -> None:
    """Forget a shank's alignment so the next selection recomputes it.

    Used after a new model or features file; the shank's result window is closed.
    """
    clear_predictions(items, PREDICTION_KEY)
    state = getattr(items.model, 'auto_alignment', None)
    if state and state.get('window') is not None:
        state['window'].close()
    items.model.auto_alignment = None


@shank_loop
def clear_display(controller: AlignmentGUIController, items: ShankController, **kwargs) -> None:
    """Drop a shank's drawn auto-alignment column (not its alignment), e.g. to recolour it."""
    clear_predictions(items, PREDICTION_KEY)


# -----------------------------------------------------------------------------
# Progress and result windows
# -----------------------------------------------------------------------------
def run_with_progress(
    controller: AlignmentGUIController, title: str, work: Callable, *, cancellable: bool = True
):
    """Run ``work(progress)`` behind a modal progress dialog showing its fraction and message.

    ``progress(fraction, message)`` updates the bar and its text and keeps the GUI responsive;
    Cancel raises :class:`AlignmentCancelledError` inside ``work`` at its next progress report.
    """
    dialog = QtWidgets.QProgressDialog(
        title, 'Cancel' if cancellable else None, 0, 1000, controller.view
    )
    dialog.setWindowTitle(title)
    dialog.setWindowModality(QtCore.Qt.WindowModal)
    dialog.setMinimumDuration(0)
    dialog.setMinimumWidth(420)
    dialog.setAutoClose(False)
    dialog.setValue(0)
    QtWidgets.QApplication.processEvents()

    def progress(fraction: float, message: str) -> None:
        if cancellable and dialog.wasCanceled():
            raise AlignmentCancelledError()
        dialog.setLabelText(f'{title}\n\n{message}')
        dialog.setValue(int(round(1000 * float(fraction))))
        QtWidgets.QApplication.processEvents()

    QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
    try:
        return work(progress)
    finally:
        QtWidgets.QApplication.restoreOverrideCursor()
        dialog.close()


class AlignmentResultWindow(QtWidgets.QDialog):
    """A non-modal window with the alignment result figure, a toolbar and a save button.

    Closing only hides it, so the region-panel option can show it again.
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


def show_result_window(controller: AlignmentGUIController, items: ShankController) -> None:
    """Show the shank's alignment window, drawing it the first time.

    A window that is already open is brought to the front.
    """
    state = getattr(items.model, 'auto_alignment', None)
    if not state:
        return
    window = state.get('window')
    if window is None:
        from ephysatlas.alignment.plotting import plot_alignment_result  # noqa: PLC0415
        from matplotlib.figure import Figure  # noqa: PLC0415

        result, probe = state['result'], state['probe']
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
        state['window'] = window
    if not window.isVisible():
        window.show()
    window.raise_()
    window.activateWindow()


@shank_loop
def show_result_windows(
    controller: AlignmentGUIController, items: ShankController, **kwargs
) -> None:
    """Re-open the alignment window of every shank that has been aligned and is not shown."""
    show_result_window(controller, items)


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


def model_vintage(model: ChannelModel) -> str:
    """Label of the feature table the release was trained on (its manifest ``vintage``).

    Falls back to the release tag, which names the vintage for the published tags but not for a
    branch such as ``main``.
    """
    index = getattr(model.encoder, 'index', None) or {}
    return index.get('vintage') or model.revision


def _probe_inputs(controller, items, model: ChannelModel) -> dict:
    """Return the shank's recorded features, in row order (top first), and planned trajectory.

    A features file chosen with ``Load features file…`` is always used. Otherwise prefers the
    model's own feature release for an IBL insertion (the features it was trained on), then the
    features the GUI loaded for this shank, from the table of the model's vintage when they come
    from ONE. ``planned_xyz`` is the planned trajectory from the feature
    table when it has one.

    Raises
    ------
    RuntimeError
        If there are no features for the probe, or they lack a feature the model needs.
    """
    pid = _current_pid(controller, items)
    vintage = model_vintage(model)
    # load_probe reads the release table with the code's FEATURE_LIST, so it only fits a release
    # trained on that feature set.
    use_release = (
        pid and not has_local_features(controller) and model.features == list(FEATURE_LIST)
    )
    probe = load_probe(pid, vintage) if use_release else None
    if probe is None:
        df = features_for_vintage(controller, items, vintage)
        if df is None:
            raise RuntimeError('No features found for this probe.')
        missing = [c for c in model.features if c not in df.columns]
        if missing:
            raise RuntimeError(
                f'Features file is missing required column(s): {missing}\n\nThe alignment '
                f'model ({vintage}) needs features computed with the {vintage} feature set.'
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
        probe = dict(
            pid=pid,
            features=features,
            planned_xyz=planned,
            axial_um=df['axial_um'].to_numpy(dtype=float),
        )
    return dict(
        pid=probe['pid'],
        features=probe['features'],
        planned_xyz=probe.get('planned_xyz'),
        axial_um=np.asarray(probe['axial_um'], dtype=float),
    )


def _warped_track_depths(n_track: int, channel_to_trace_work, channel_depth_um) -> np.ndarray:
    """``[n_track]`` probe depth (µm, tip ~0) of every GUI track sample.

    The GUI track runs bottom -> top and the alignment works top -> bottom. An affine fit of trace
    index -> channel depth through the aligned channels maps every track sample to a probe depth,
    so regions beyond the probe stay visible above and below it.
    """
    j_gui = (n_track - 1) - np.asarray(channel_to_trace_work, dtype=int)
    depth = np.asarray(channel_depth_um, dtype=float)
    ok = np.isfinite(depth)
    if ok.sum() >= 2 and np.ptp(j_gui[ok]) > 0:
        scale, offset = np.polyfit(j_gui[ok].astype(float), depth[ok], deg=1)
    else:
        scale, offset = 10.0, 0.0
    return scale * np.arange(n_track, dtype=float) + offset


def align_shank(controller: AlignmentGUIController, items: ShankController) -> dict | None:
    """Return the shank's alignment, computing it (and opening its window) on first use.

    None when the user cancelled or the alignment failed.

    Returns
    -------
    dict
        ``result`` (``AlignmentResult``), ``probe`` inputs, the GUI ``track`` (bottom -> top) and
        ``track_depth_um`` (probe depth of each track sample), plus the result ``window``.
    """
    state = getattr(items.model, 'auto_alignment', None)
    if state:
        return state
    model = get_model(controller)
    if model is None:
        return None
    try:
        probe = _probe_inputs(controller, items, model)
    except Exception as exc:  # noqa: BLE001 - show any feature-loading problem to the user
        QtWidgets.QMessageBox.critical(controller.view, 'Auto-alignment', str(exc))
        return None
    # GUI track: bottom -> top along the histology picks, 10 µm apart, top to bottom of the brain.
    track_gui = np.asarray(items.model.align_handle.xyz_samples, dtype=np.float32)
    try:
        result = run_with_progress(
            controller,
            'Auto-alignment',
            lambda progress: align_histology(
                model,
                probe['features'],
                track_gui[::-1].copy(),
                pid=probe['pid'],
                progress=progress,
            ),
        )
    except AlignmentCancelledError:
        return None
    except Exception as exc:  # noqa: BLE001
        QtWidgets.QMessageBox.critical(controller.view, 'Auto-alignment', str(exc))
        return None
    state = dict(
        result=result,
        probe=probe,
        track=track_gui,
        track_depth_um=_warped_track_depths(
            len(track_gui), result.diagnostics['channel_to_trace'], probe['axial_um']
        ),
        window=None,
    )
    items.model.auto_alignment = state
    show_result_window(controller, items)
    return state


# -----------------------------------------------------------------------------
# The region-panel column: predicted regions or confidence
# -----------------------------------------------------------------------------
def predict_regions(controller, items):
    """Regions along the shank's whole histology track: ``(region_ids, depths_m)``."""
    state = align_shank(controller, items)
    if state is None:
        return None
    # Allen region ids, as the region panel's ``regions.get`` expects (not region-table indices).
    ids = controller.model.brain_atlas.get_labels(state['track'], mapping='Allen')
    return ids, state['track_depth_um'] / 1e6


def predict_confidence(controller, items) -> Bunch | None:
    """Return the alignment confidence p(good) along the probe, for ``plot_histology``.

    Green is high and red low confidence (``RdYlGn``); the track beyond the probe and channels
    without a value are grey. Depths are the channels' probe depths (µm), like the region column.
    """
    import matplotlib as mpl  # noqa: PLC0415 - only needed here

    state = align_shank(controller, items)
    if state is None:
        return None
    result, probe = state['result'], state['probe']
    depth = np.asarray(probe['axial_um'], dtype=float)
    p_good = np.where(result.valid, result.p_good, np.nan)
    # One value per probe depth (Neuropixels has two sites per depth), tip first.
    levels = np.unique(depth[np.isfinite(depth)])
    p = np.array(
        [
            np.nanmean(p_good[depth == d]) if np.isfinite(p_good[depth == d]).any() else np.nan
            for d in levels
        ]
    )
    edges = np.r_[levels[0] - 10.0, 0.5 * (levels[1:] + levels[:-1]), levels[-1] + 10.0]

    cmap = mpl.colormaps['RdYlGn']
    q = np.where(np.isfinite(p), np.round(p / CONFIDENCE_STEP) * CONFIDENCE_STEP, np.nan)
    region, colour, labels = [], [], []
    track = np.sort(state['track_depth_um'][[0, -1]])
    if track[0] < edges[0]:
        region.append((track[0], edges[0]))
        colour.append(NO_VALUE_RGB)
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
    if track[1] > edges[-1]:
        region.append((edges[-1], track[1]))
        colour.append(NO_VALUE_RGB)
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
