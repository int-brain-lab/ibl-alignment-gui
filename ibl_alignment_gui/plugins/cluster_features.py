from typing import TYPE_CHECKING, Any

import numpy as np
import pyqtgraph as pg
from qtpy import QtCore, QtGui, QtWidgets

from brainbox.population.decode import xcorr
from ibl_alignment_gui.utils.qt.custom_widgets import PopupWindow, set_axis

if TYPE_CHECKING:
    from ibl_alignment_gui.app.app_controller import AlignmentGUIController, AlignmentGUIView
    from ibl_alignment_gui.app.shank_controller import ShankController
    from ibl_alignment_gui.loaders.plot_loader import PlotLoader

PLUGIN_NAME = 'Cluster Features'

AUTOCORR_BIN_SIZE = 0.5 / 1000
AUTOCORR_WIN_SIZE = 20 / 1000
FS = 30000


def setup(controller: 'AlignmentGUIController') -> None:
    """
    Set up the Cluster Features plugin.

    Adds a submenu to the main GUI for managing the cluster popups.

    Parameters
    ----------
    controller: AlignmentGUIController
        The main application controller.
    """
    controller.plugins[PLUGIN_NAME] = dict()
    controller.plugins[PLUGIN_NAME]['loader'] = ClusterPopupManager(controller)
    controller.plugins[PLUGIN_NAME]['callback'] = callback
    controller.plugins[PLUGIN_NAME]['activate'] = True

    # Add a submenu to the main menu
    plugin_menu = QtWidgets.QMenu(PLUGIN_NAME, controller.view)
    controller.plugin_options.addMenu(plugin_menu)
    # Add the actions to the submenu
    # Minimise cluster popups
    action = QtWidgets.QAction(f'Minimise/Show {PLUGIN_NAME}', controller.view)
    action.setShortcut('M')
    action.setShortcutContext(QtCore.Qt.ApplicationShortcut)
    action.triggered.connect(controller.plugins[PLUGIN_NAME]['loader'].minimise_popups)
    plugin_menu.addAction(action)
    # Close cluster popups
    action = QtWidgets.QAction(f'Close {PLUGIN_NAME}', controller.view)
    action.setShortcut('Alt+X')
    action.setShortcutContext(QtCore.Qt.ApplicationShortcut)
    action.triggered.connect(controller.plugins[PLUGIN_NAME]['loader'].close_popups)
    plugin_menu.addAction(action)


def callback(
    controller: 'AlignmentGUIController', items: 'ShankController', _, point: pg.ScatterPlotItem
) -> None:
    """
    Triggered when a cluster in a scatter plot is clicked.

    Computes autocorrelation and template waveform for the selected cluster
    and opens a popup showing the plots.

    Parameters
    ----------
    controller: AlignmentGUIController
        The main application controller.
    items: ShankController
        The shank controller containing cluster data.
    point: pg.ScatterPlotItem
        The clicked scatter item in the scatter plot.
    """
    point_pos = point[0].pos()
    clust_idx = np.argwhere(items.cluster_data == point_pos.x())[0][0]

    plot_loader = items.model.loaders['plots']

    t_autocorr, t_template = compute_timescales(plot_loader)

    data = {}
    data['t_autocorr'] = t_autocorr
    data['autocorr'], clust_no = get_autocorr(plot_loader, clust_idx)
    data['t_template'] = t_template
    data['template_wf'] = get_template_wf(plot_loader, clust_idx)

    controller.plugins[PLUGIN_NAME]['loader'].add_popup(items.name, clust_no, items.config, data)


class ClusterPopup(PopupWindow):
    """
    A popup qt window per cluster.

    Shows plots of the cluster autocorrelogram and template waveform.

    Parameters
    ----------
    title: str
        The title of the popup window.
    data: dict or None
        A dictionary containing data to be plotted in the popup.
    parent: QtWidgets.QMainWindow or None
        The parent window of the popup.
    """

    def __init__(self, title: str, view: 'AlignmentGUIView', data: dict | None = None):
        self.data = data
        super().__init__(title, parent=view, size=(300, 300), graphics=True)

    def setup(self) -> None:
        """Configure the plots inside the popup window."""
        autocorr_plot = pg.PlotItem()
        autocorr_plot.setXRange(
            min=np.min(self.data['t_autocorr']), max=np.max(self.data['t_autocorr'])
        )
        autocorr_plot.setYRange(min=0, max=1.05 * np.max(self.data['autocorr']))
        set_axis(autocorr_plot, 'bottom', label='T (ms)')
        set_axis(autocorr_plot, 'left', label='Number of spikes')
        plot = pg.BarGraphItem(
            x=self.data['t_autocorr'],
            height=self.data['autocorr'],
            width=0.24,
            brush=QtGui.QColor(160, 160, 160),
        )
        autocorr_plot.addItem(plot)

        template_plot = pg.PlotItem()
        plot = pg.PlotCurveItem()
        template_plot.setXRange(
            min=np.min(self.data['t_template']), max=np.max(self.data['t_template'])
        )
        set_axis(template_plot, 'bottom', label='T (ms)')
        set_axis(template_plot, 'left', label='Amplitude (a.u.)')
        plot.setData(
            x=self.data['t_template'],
            y=self.data['template_wf'],
            pen=pg.mkPen(color='k', style=QtCore.Qt.SolidLine, width=2),
        )
        template_plot.addItem(plot)

        self.popup_widget.addItem(autocorr_plot, 0, 0)
        self.popup_widget.addItem(template_plot, 1, 0)


class ClusterPopupManager:
    """
    Manager for multiple cluster popups in the GUI.

    Attributes
    ----------
    parent_view : QtWidgets.QMainWindow
        The main window of the application.
    cluster_popups : list
        A list of currently open cluster popups.
    popup_status : bool
        Status indicating whether popups are minimised or shown.
    """

    def __init__(self, controller: 'AlignmentGUIController'):
        self.view = controller.view
        self.cluster_popups = []
        self.popup_status = True

    def add_popup(self, shank: str, clust_no: int, config: str, data: dict[str, Any]) -> None:
        """
        Add a new cluster popup to the manager and set up its signals.

        Parameters
        ----------
        shank: str
            The name of shank
        clust_no: int
            The cluster number.
        config: str
            The config of the shank
        data: dict
            A dict containing data to be plotted in the popup.
        """
        name = f'{shank}_{config}' if config else shank
        clust_popup = ClusterPopup._get_or_create(
            f'{name}: cluster {clust_no}', self.view, data=data
        )
        clust_popup.closed.connect(self.popup_closed)
        clust_popup.leave.connect(self.popup_left)
        clust_popup.enter.connect(self.popup_entered)
        self.cluster_popups.append(clust_popup)

    def minimise_popups(self) -> None:
        """Toggle between minimizing and restoring all cluster popups."""
        self.popup_status = not self.popup_status
        if self.popup_status:
            for pop in self.cluster_popups:
                pop.showNormal()
        else:
            for pop in self.cluster_popups:
                pop.showMinimized()

    def close_popups(self) -> None:
        """Close all cluster popups and reset the list."""
        for pop in self.cluster_popups:
            pop.blockSignals(True)
            pop.close()
        self.cluster_popups = []

    def popup_closed(self, popup: ClusterPopup) -> None:
        """
        Triggered when a popup is closed by the user.

        Parameters
        ----------
        popup: ClusterPopup
            The cluster popup that was closed.
        """
        if len(self.cluster_popups) > 0:
            popup_idx = [iP for iP, pop in enumerate(self.cluster_popups) if pop == popup][0]
            self.cluster_popups.pop(popup_idx)

    def popup_left(self) -> None:
        """Triggered when the mouse leaves a popup."""
        self.view.raise_()
        self.view.activateWindow()

    def popup_entered(self, popup: ClusterPopup) -> None:
        """Triggered when the mouse enters a popup."""
        popup.raise_()
        popup.activateWindow()

    def reset(self) -> None:
        """Triggered when the main GUI is closed. Closes all popups and resets the manager."""
        self.close_popups()


def compute_timescales(plot_loader: 'PlotLoader') -> tuple[np.ndarray, np.ndarray]:
    """
    Compute time vectors for autocorrelogram and template waveform plots.

    Parameters
    ----------
    plot_loader: PlotLoader
        The plot loader object containing spike and cluster data.

    Returns
    -------
    t_autocorr: np.ndarray
        The time vector for autocorrelogram (ms).
    t_template: np.ndarray
        The time vector for template waveform (ms).
    """
    t_autocorr = 1e3 * np.arange(
        (AUTOCORR_WIN_SIZE / 2) - AUTOCORR_WIN_SIZE,
        (AUTOCORR_WIN_SIZE / 2) + AUTOCORR_BIN_SIZE,
        AUTOCORR_BIN_SIZE,
    )
    n_template = plot_loader.data['clusters']['waveforms'][0, :, 0].size
    t_template = 1e3 * (np.arange(n_template)) / FS

    return t_autocorr, t_template


def get_autocorr(plot_loader: 'PlotLoader', clust_idx: int) -> tuple[np.ndarray, int]:
    """
    Compute the autocorrelogram for a specific cluster.

    Parameters
    ----------
    plot_loader: PlotLoader
        The plot loader object containing spike and cluster data.
    clust_idx: int
        Index of the cluster

    Returns
    -------
    autocorr: np.ndarray
        The autocorrelogram of the selected cluster
    clust_id: int
        The cluster id of the selected cluster
    """
    idx = plot_loader.spike_clusters == plot_loader.cluster_idx[clust_idx]
    autocorr = xcorr(
        plot_loader.spike_times[idx],
        plot_loader.spike_clusters[idx],
        AUTOCORR_BIN_SIZE,
        AUTOCORR_WIN_SIZE,
    )
    if plot_loader.data['clusters'].get('metrics', {}).get('cluster_id', None) is None:
        clust_id = plot_loader.cluster_idx[clust_idx]
    else:
        clust_id = plot_loader.data['clusters'].metrics.cluster_id[
            plot_loader.cluster_idx[clust_idx]
        ]

    return autocorr[0, 0, :], clust_id


def get_template_wf(plot_loader: 'PlotLoader', clust_idx: int) -> np.ndarray:
    """
    Retrieve the template waveform for a specific cluster.

    Parameters
    ----------
    plot_loader: PlotLoader
        The plot loader object containing spike and cluster data.
    clust_idx: int
        Index of the cluster

    Returns
    -------
    template_wf: np.ndarray
        The template waveform of the selected cluster
    """
    template_wf = plot_loader.data['clusters']['waveforms'][
        plot_loader.cluster_idx[clust_idx], :, 0
    ]
    return template_wf * 1e6
