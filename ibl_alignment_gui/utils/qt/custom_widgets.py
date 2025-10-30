import random
from abc import abstractmethod
from collections import defaultdict
from typing import Callable

import matplotlib as mpl
import numpy as np
import pyqtgraph as pg
from pyqtgraph.functions import makeARGB
from qtpy import QtCore, QtGui, QtWidgets

from iblutil.util import Bunch


def set_axis(
        fig: pg.PlotItem | pg.PlotWidget,
        ax: str,
        show: bool = True,
        label: str | None = None,
        pen: str = 'k',
        ticks: bool = True
) -> pg.AxisItem:
    """
    Show, hide, and configure an axis on a pyqtgraph figure.

    Parameters
    ----------
    fig : pg.PlotWidget or pg.PlotItem
        The figure containing the axis to modify.
    ax : str
        The orientation of the axis. Must be one of {'left', 'right', 'top', 'bottom'}.
    show : bool, optional
        Whether to show the axis (default is True).
    label : str or None, optional
        The label text for the axis (default is None).
    pen : str, optional
        The color for the axis line and text (default is 'k' for black).
    ticks : bool, optional
        Whether to show axis ticks (default is True).

    Returns
    -------
    axis : pg.AxisItem
        The configured axis object.
    """
    if ax not in {'left', 'right', 'top', 'bottom'}:
        raise ValueError(f"Invalid axis '{ax}'. Must be one of 'left', 'right', 'top', 'bottom'.")

    label = label or ''
    axis = fig.getAxis(ax) if isinstance(fig, pg.PlotItem) else fig.plotItem.getAxis(ax)

    if show:
        axis.show()
        axis.setPen(pen)
        axis.setTextPen(pen)
        axis.setLabel(label)
        if not ticks:
            axis.setTicks([[(0, ''), (0.5, ''), (1, '')]])
    else:
        axis.hide()

    return axis


def set_font(
        fig: pg.PlotItem | pg.PlotWidget,
        ax: str,
        ptsize: int = 8,
        width: int | None = None,
        height: int | None = None
) -> None:
    """
    Set the font size and optionally the axis width/height for a given axis in a pyqtgraph figure.

    Parameters
    ----------
    fig : pg.PlotItem or pg.PlotWidget
        The figure containing the axis to modify.
    ax : str
        The orientation of the axis. Must be one of {'left', 'right', 'top', 'bottom'}.
    ptsize : int, optional
        Point size for the axis font (default is 8).
    width : int, optional
        Width to set for the axis in pixels. Only applicable for vertical axes.
    height : int, optional
        Height to set for the axis in pixels. Only applicable for horizontal axes.
    """
    if ax not in {'left', 'right', 'top', 'bottom'}:
        raise ValueError(f"Invalid axis '{ax}'. Must be one of 'left', 'right', 'top', 'bottom'.")

    axis = fig.getAxis(ax) if isinstance(fig, pg.PlotItem) else fig.plotItem.getAxis(ax)

    font = QtGui.QFont()
    font.setPointSize(ptsize)
    axis.setStyle(tickFont=font)
    axis.setLabel(**{'font-size': f'{ptsize}pt'})

    if width is not None:
        axis.setWidth(width)
    if height is not None:
        axis.setHeight(height)


class PopupWindow(QtWidgets.QWidget):
    """
    A reusable popup window with optional graphics layout support.

    This class serves as a base for creating popup windows that:
    - Are top-level, floating windows.
    - Can contain either a pyqtgraph GraphicsLayoutWidget or a standard QWidget with a layout.
    - Emit signals when the mouse enters, leaves, or when the window is closed.
    - Can be reused or retrieved by title using the `_get_or_create` class method.

    Subclasses must implement the abstract `setup()` method to populate the popup content.

    Signals
    -------
    closed : QtCore.Signal(QtWidgets.QWidget)
        Emitted when the popup is closed.
    leave: QtCore.Signal(QtWidgets.QWidget)
        Emitted when the mouse leaves the widget area.
    enter: QtCore.Signal(QtWidgets.QWidget)
        Emitted when the mouse enters the widget area.

    Parameters
    ----------
    title : str
        The window title.
    parent : QWidget, optional
        Parent widget. If provided, the popup closes automatically when the parent is destroyed.
    size : tuple or list, default=(300, 300)
        Initial window size (width, height).
    graphics : bool, default=True
        If True, use a pg.GraphicsLayoutWidget as the content area, otherwise, use a standard
        QWidget with a QGridLayout.
    """

    closed = QtCore.Signal(QtWidgets.QWidget)
    leave = QtCore.Signal(QtWidgets.QWidget)
    enter = QtCore.Signal(QtWidgets.QWidget)

    @classmethod
    def _instances(cls) -> list[QtWidgets.QWidget]:
        """
        Return a list of currently active instances of this PopupWindow subclass.

        Returns
        -------
        list[PopupWindow]
            All visible PopupWindow instances of this class.
        """
        app = QtWidgets.QApplication.instance()
        return [w for w in app.topLevelWidgets() if isinstance(w, cls)]

    @classmethod
    def _get_or_create(cls, title: str, **kwargs):
        """
        Retrieve an existing PopupWindow with the given title or create a new one.

        If a visible window with the same title exists, it is activated and returned.
        Otherwise, a new instance is created and returned.

        Parameters
        ----------
        title : str
            The title of the popup window to retrieve or create.
        **kwargs
            Additional keyword arguments passed to the constructor.

        Returns
        -------
        PopupWindow
            The existing or newly created popup window.
        """
        window = next((w for w in cls._instances() if w.isVisible() and
                       w.windowTitle() == title), None)
        if window is None:
            window = cls(title, **kwargs)
        else:
            window.showNormal()
            window.activateWindow()
        return window

    def __init__(
            self,
            title: str,
            parent: QtWidgets.QMainWindow=None,
            size: tuple | list = (300, 300),
            graphics: bool = True
    ):
        super().__init__(parent)

        self.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.WindowStaysOnTopHint)
        self.resize(*size)
        self.move(random.randrange(30) + 1000, random.randrange(30) + 200)

        # Create content widget
        if graphics:
            self.popup_widget = pg.GraphicsLayoutWidget()
        else:
            self.popup_widget = QtWidgets.QWidget()
            self.layout = QtWidgets.QGridLayout()
            self.popup_widget.setLayout(self.layout)

        # Top-level layout
        self.main_layout = QtWidgets.QVBoxLayout()
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.addWidget(self.popup_widget)
        self.setLayout(self.main_layout)

        # Close popup when parent is closed
        if parent is not None:
            parent.destroyed.connect(self.close)

        self.setWindowTitle(title)
        self.setup()
        self.show()

    @abstractmethod
    def setup(self):
        """Abstract method to be implemented by subclasses."""

    def close_event(self, event: QtCore.QEvent, *args, **kwargs) -> None:
        """
        Handle the window close event.

        Emits the `closed` signal and calls the parent closeEvent.

        Parameters
        ----------
        event : QCloseEvent
            The Qt close event.
        """
        self.closed.emit(self)
        super().closeEvent(event, *args, **kwargs)

    def leave_event(self, event: QtCore.QEvent, *args, **kwargs) -> None:
        """
        Handle the mouse leave event.

        Emits the `leave` signal and calls the parent leaveEvent.

        Parameters
        ----------
        event : QEvent
            The Qt leave event.
        """
        self.leave.emit(self)
        super().leaveEvent(event, *args, **kwargs)

    def enter_event(self, event: QtCore.QEvent, *args, **kwargs) -> None:
        """
        Handle the mouse enter event.

        Emits the `enter` signal and calls the parent enterEvent.

        Parameters
        ----------
        event : QtCore.QEvent
            The Qt enter event.
        """
        self.enter.emit(self)
        super().enterEvent(event, *args, **kwargs)


class ColorBar(pg.GraphicsWidget):
    """
    A custom color bar widget for visualizing scalar data ranges as a gradient.

    This widget:
    - Creates a color gradient based on a Matplotlib colormap.
    - Displays it as a horizontal or vertical bar in a pyqtgraph scene.
    - Provides ticks and labels to indicate data levels.
    - Can map raw data values into corresponding QColor brushes.

    Parameters
    ----------
    cmap_name : str
        Name of the Matplotlib colormap to use.
    width : int, default=20
        The width of the color bar in scene units.
    height : int, default=5
        The height of the color bar in scene units.
    plot_item : pg.PlotItem, optional
        A plot item to which this color bar will be added. If provided,
        the widget is automatically inserted and its axes are prepared.
    cbin : int, default=256
        Number of discrete color bins for the LUT.
    orientation : {'horizontal', 'vertical'}, default='horizontal'
        Orientation of the color bar.
    """

    def __init__(
            self,
            cmap_name: str,
            width: int = 20,
            height: int = 5,
            plot_item: pg.PlotItem | None = None,
            cbin: int = 256,
            orientation: str = 'horizontal'
    ):
        pg.GraphicsWidget.__init__(self)

        # Set dimensions
        self.width: int = width
        self.width: int = width
        self.height: int = height

        # Set orientation
        self.orientation: str = orientation

        # Create colour map from matplotlib colourmap name
        self.cmap_name: str = cmap_name
        self.cmap, self.lut, self.grad = self.get_color(self.cmap_name, cbin=cbin)

        # Create plot item to place the colorbar
        self.plot: pg.PlotItem = plot_item
        if self.plot:
            self.plot.setXRange(0, self.width)
            self.plot.setYRange(0, self.height)
            self.plot.addItem(self)

        QtGui.QPainter()

        self.ticks = None

    @staticmethod
    def get_color(
            cmap_name: str,
            cbin: int=256
    ) -> tuple[pg.ColorMap, np.ndarray, QtGui.QLinearGradient]:
        """
        Generate a pyqtgraph-compatible color map, LUT, and gradient from a given colormap.

        Parameters
        ----------
        cmap_name : str
            Name of the Matplotlib colormap.
        cbin : int, default=256
            Number of discrete bins for the LUT.

        Returns
        -------
        map : pg.ColorMap
            A pyqtgraph ColorMap object.
        lut : np.ndarray
            Lookup table for color mapping.
        grad : QtGui.QLinearGradient
            Gradient object for rendering the bar.
        """
        mpl_cmap = mpl.cm.get_cmap(cmap_name)
        if isinstance(mpl_cmap, mpl.colors.LinearSegmentedColormap):
            cbins = np.linspace(0.0, 1.0, cbin)
            colors = (mpl_cmap(cbins)[np.newaxis, :, :3][0]).tolist()
        else:
            colors = mpl_cmap.colors
        colors = [(np.array(c) * 255).astype(int).tolist() + [255.] for c in colors]
        positions = np.linspace(0, 1, len(colors))
        cmap = pg.ColorMap(positions, colors)
        lut = cmap.getLookupTable()
        grad = cmap.getGradient()

        return cmap, lut, grad

    def paint(self, p: QtGui.QPainter, *args) -> None:
        """
        Render the color bar gradient.

        Parameters
        ----------
        p : QtGui.QPainter
            The painter used to draw the widget.
        """
        p.setPen(QtCore.Qt.NoPen)
        self.grad.setStart(0, self.height / 2)
        self.grad.setFinalStop(self.width, self.height / 2)
        p.setBrush(pg.QtGui.QBrush(self.grad))
        p.drawRect(QtCore.QRectF(0, 0, self.width, self.height))

    def get_brush(
            self,
            data: np.ndarray,
            levels: list | tuple | np.ndarray | None = None
    ) -> list[QtGui.QColor]:
        """
        Convert numeric data values into QColor brushes based on the color bar's LUT.

        Parameters
        ----------
        data : ndarray
            Array of data values to map.
        levels : tuple[float, float], optional
            Min/max values to normalize the data. Defaults to data range.

        Returns
        -------
        list[QtGui.QColor]
            List of QColor objects corresponding to the data values.
        """
        if levels is None:
            levels = [np.min(data), np.max(data)]
        brush_rgb, _ = makeARGB(data[:, np.newaxis], levels=levels, lut=self.lut, useRGBA=True)
        brush = [QtGui.QColor(*col) for col in np.squeeze(brush_rgb)]
        return brush

    def get_colour_map(self) -> np.ndarray:
        """
        Return the underlying LUT for this color bar.

        Returns
        -------
        ndarray
            Lookup table array.
        """
        return self.lut

    def set_levels(
            self,
            levels: tuple | list | np.ndarray,
            label: str | None = None,
            n_ticks: int = 2
    ) -> None:
        """
        Set the levels represented by the color bar and configure ticks and optional label.

        Parameters
        ----------
        levels : tuple or list or np.ndarray
            The (min, max) data values for the color mapping.
        label : str, optional
            Axis label text.
        n_ticks : int, default=2
            Number of ticks to display on the axis.
        """
        self.levels = levels
        self.ticks = self.get_ticks(n_ticks)
        self.label = label
        self.set_axis(ticks=self.ticks, label=label)

    def set_axis(
            self,
            ticks: list[tuple[float, str]] | None = None,
            label: str | None = None,
            loc: str | None = None,
            extent: int = 30
    ) -> None:
        """
        Configure the axis associated with this color bar.

        Parameters
        ----------
        ticks : list[tuple[float, str]], optional
            Tick positions and labels.
        label : str, optional
            Axis label text.
        loc : {'top', 'bottom', 'left', 'right'}, optional
            Which axis to configure. Defaults based on orientation.
        extent : int, default=30
            Height or width allocated for the axis (in scene units).
        """
        if loc is None:
            loc = 'top' if self.orientation == 'horizontal' else 'left'
        ax = self.plot.getAxis(loc)
        ax.show()
        ax.setStyle(stopAxisAtTick=(True, True), autoExpandTextSpace=True)
        if self.orientation == 'horizontal':
            ax.setHeight(extent)
        else:
            ax.setWidth(extent)
        if ticks:
            ax.setPen('k')
            ax.setTextPen('k')
            ax.setTicks([ticks])
        else:
            ax.setTextPen('w')
            ax.setPen('w')
        # Note this has to come after the setPen above otherwise overwritten
        ax.setLabel(label, color='k')

    def get_ticks(self, n: int = 3) -> list[tuple[float, str]]:
        """
        Generate evenly spaced tick positions and labels based on the current color bar levels.

        Parameters
        ----------
        n : int, default=3
            Number of ticks to generate.

        Returns
        -------
        list[tuple[float, str]]
            A list of (position, label) pairs for axis ticks.
        """
        extent = self.width if self.orientation == 'horizontal' else self.height
        offset = (0.005 * extent)

        ticks = []
        for i in range(n):
            frac = i / (n - 1)
            pos = frac * extent
            val = self.levels[0] + frac * (self.levels[1] - self.levels[0])
            val = int(val) if np.abs(val) > 1 else np.round(val, 1)
            if i == 0:
                pos += offset
            elif i == n-1:
                pos -= offset
            ticks.append((pos, str(val)))

        return ticks


class GridTabSwitcher(QtWidgets.QWidget):
    """
    A container widget for displaying multiple panels in either a grid layout or a tabbed layout.

    Attributes
    ----------
    custom_signal : QtCore.Signal(str)
        A signal emitted when the layout is toggled
    layout : QtWidgets.QVBoxLayout
        The main vertical layout containing either the grid or tab widget.
    panels : list[QtWidgets.QWidget]
        The panel widgets to add to the grid or tab widget.
    panel_names : list[str]
        The names of the panels (used for tab labels in tabbed mode).
    headers : tuple[QtWidgets.QLabel], optional
        Header labels to be shown on top of the panels
    tab_widget : QtWidgets.QTabWidget
        The tab widget used in tabbed layout mode.
    grid_widget : QtWidgets.QSplitter
        The grid widget for grid layout mode.
    top_grid, bottom_grid : QtWidgets.QSplitter
        Horizontal and vertical splitters used to create the grid layout.
    grid_layout : bool
        Whether the current layout is grid-based (True) or tabbed (False).
    """

    custom_signal = QtCore.Signal(str)

    def __init__(self):
        super().__init__()

        self.setFocusPolicy(QtCore.Qt.StrongFocus)

        # Create a layout for the widget
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        # Lists to keep track of panels and headers
        self.panels = []
        self.panel_names = []
        self.headers = []

        # Tab widget
        self.tab_widget = QtWidgets.QTabWidget()
        self.tab_widget.setTabPosition(QtWidgets.QTabWidget.South)
        self.tab_widget.hide()
        self.tab_widget.setStyleSheet("""
        QTabBar::tab:selected {
            background-color: #c92d0e;
            color: white;
            font-weight: bold;
        }
        """)

        # Grid widget
        self.grid_widget = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.top_grid = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.bottom_grid = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        # Track whether we are in grid or tab layout
        self.grid_layout = True

    def initialise(self, panels: list[QtWidgets.QWidget], names: list[str],
                   headers: list[QtWidgets.QLabel] | None = None) -> None:
        """
        Initialize the widget with a set of panels and their names.

        Parameters
        ----------
        panels : list[QtWidgets.QWidget]
            The panel widgets to be displayed.
        names : list[str]
            The names corresponding to each panel (used for tab labels).
        headers : list[QtWidgets.QLabel], optional
            Header labels associated with each panel.
        """
        self.headers = headers
        self.panels = panels
        self.panel_names = list(names)

        if len(self.panels) > 1:
            self.grid_widget.addWidget(self.top_grid)
        if len(self.panels) > 2:
            self.grid_widget.addWidget(self.bottom_grid)

        self.add_grid_layout()

    def delete_widgets(self) -> None:
        """Remove all panels from the current layout and delete the widgets."""
        if self.grid_layout:
            self.remove_grid_layout(delete=True)
        else:
            self.remove_tab_layout(delete=True)
        self.panels = []

    def add_header(self) -> None:
        """Add any associated headers to the panels."""
        if self.headers:
            for panel, header in zip(self.panels, self.headers, strict=False):
                panel.layout().insertWidget(0, header)

    def remove_header(self) -> None:
        """Remove any associated headers from the panels."""
        if self.headers:
            for panel, header in zip(self.panels, self.headers, strict=False):
                panel.layout().removeWidget(header)

    def add_grid_layout(self) -> None:
        """Add the panels to a grid layout and show the grid widget. Supports 1-4 panels."""
        if len(self.panels) == 1:
            self.grid_widget.addWidget(self.panels[0])
        elif len(self.panels) == 2:
            self.top_grid.addWidget(self.panels[0])
            self.top_grid.addWidget(self.panels[1])
            self.top_grid.setSizes([1] * self.top_grid.count())
        elif len(self.panels) == 3:
            self.top_grid.addWidget(self.panels[0])
            self.top_grid.addWidget(self.panels[1])
            self.bottom_grid.addWidget(self.panels[2])
            self.top_grid.setSizes([1] * self.top_grid.count())
            self.bottom_grid.setSizes([1] * 2)
        elif len(self.panels) == 4:
            self.top_grid.addWidget(self.panels[0])
            self.top_grid.addWidget(self.panels[1])
            self.bottom_grid.addWidget(self.panels[2])
            self.bottom_grid.addWidget(self.panels[3])
            self.top_grid.setSizes([1] * self.top_grid.count())
            self.bottom_grid.setSizes([1] * self.bottom_grid.count())
        else:
            return

        self.grid_widget.show()
        for panel in self.panels:
            panel.show()

        self.layout.addWidget(self.grid_widget)

    def remove_grid_layout(self, delete: bool = False) -> None:
        """
        Remove all panels from the grid layout and hide the grid widget.

        Parameters
        ----------
        delete : bool, default=False
            If True, deletes the widgets after removal.
        """
        if len(self.panels) == 1:
            splitters = [self.grid_widget]
        elif len(self.panels) == 2:
            splitters = [self.top_grid]
        else:
            splitters = [self.top_grid, self.bottom_grid]

        for splitter in splitters:
            for i in reversed(range(splitter.count())):
                widget = splitter.widget(i)
                widget.setParent(None)
                if delete:
                    del widget

        self.layout.removeWidget(self.grid_widget)
        self.grid_widget.hide()

    def add_tab_layout(self) -> None:
        """Add the panels to a tabbed layout and show the tab widget."""
        for i, w in enumerate(self.panels):
            self.tab_widget.addTab(w, self.panel_names[i])
        self.layout.addWidget(self.tab_widget)
        self.tab_widget.show()

    def remove_tab_layout(self, delete: bool = False) -> None:
        """
        Remove all panels from the tabbed layout and hide the tab widget.

        Parameters
        ----------
        delete : bool, default=False
            If True, deletes the widgets after removal.
        """
        for i in reversed(range(self.tab_widget.count())):
            widget = self.tab_widget.widget(i)
            widget.setParent(None)
            if delete:
                del widget

        self.layout.removeWidget(self.tab_widget)
        self.tab_widget.hide()
        if delete:
            self.grid_layout = not self.grid_layout

    def toggle_layout(self) -> None:
        """Toggle between grid and tab layout."""
        self.tab_widget.blockSignals(True)
        if self.grid_layout:
            # Switch to tab layout
            self.remove_grid_layout()
            self.remove_header()
            self.add_tab_layout()
        else:
            # Switch to grid layout
            self.remove_tab_layout()
            self.add_header()
            self.add_grid_layout()
            # Emit signal so we can respond to change
            self.custom_signal.emit("layout_switched")

        self.grid_layout = not self.grid_layout
        self.tab_widget.blockSignals(False)


class ButtonWidget(QtWidgets.QWidget):
    """
    Widget containing buttons and labels for fitting and navigating through moves.

    Parameters
    ----------
    parent: QtWidgets.QMainWindow
        The parent window

    Attributes
    ----------
    buttons: Bunch
        A Bunch object containing the added buttons. Each button is a QPushButton object.
    labels: Bunch
        A Bunch object containing the added labels. Each label is a QLabel object.

    """

    def __init__(self, parent: QtWidgets.QMainWindow | None = None):
        super().__init__(parent)

        self.buttons: Bunch = Bunch()
        self.labels: Bunch = Bunch()

        self.create_widgets()
        self.layout_widgets()

    def create_widgets(self) -> None:
        """Create the buttons and labels."""
        # Button to apply interpolation
        self.buttons['fit'] = QtWidgets.QPushButton('Fit')
        # Button to apply offset
        self.buttons['offset'] = QtWidgets.QPushButton('Offset')
        # String to display current move index
        self.labels['current'] = QtWidgets.QLabel()
        # String to display total number of moves
        self.labels['total'] = QtWidgets.QLabel()
        # Button to reset GUI to initial state
        self.buttons['reset']= QtWidgets.QPushButton('Reset')
        # Button to upload final state to Alyx/ to local file
        self.buttons['upload'] = QtWidgets.QPushButton('Upload')
        # Button to go to next move
        self.buttons['next'] = QtWidgets.QPushButton('Next')
        # Button to go to previous move
        self.buttons['previous'] = QtWidgets.QPushButton('Previous')

    def layout_widgets(self) -> None:
        """Layout the buttons and labels."""
        # Layout rows
        hlayout1 = QtWidgets.QHBoxLayout()
        hlayout1.addWidget(self.buttons['fit'], stretch=1)
        hlayout1.addWidget(self.buttons['offset'], stretch=1)
        hlayout1.addWidget(QtWidgets.QLabel(), stretch=2) # Placeholder to push buttons to the left
        hlayout2 = QtWidgets.QHBoxLayout()
        hlayout2.addWidget(self.buttons['previous'], stretch=1)
        hlayout2.addWidget(self.buttons['next'], stretch=1)
        hlayout2.addWidget(self.labels['current'], stretch=2)
        hlayout3 = QtWidgets.QHBoxLayout()
        hlayout3.addWidget(self.buttons['reset'], stretch=1)
        hlayout3.addWidget(self.buttons['upload'], stretch=1)
        hlayout3.addWidget(self.labels['total'], stretch=2)

        # Main layout
        button_layout = QtWidgets.QVBoxLayout()
        button_layout.addLayout(hlayout1)
        button_layout.addLayout(hlayout2)
        button_layout.addLayout(hlayout3)
        self.setLayout(button_layout)


class SelectionWidget(QtWidgets.QWidget):
    """
    Widget containing various dropdowns and buttons to select the data to load.

    The added items depend on how the gui is run. For example, in offline mode, a dialog to select
    the local folder is provided in place of some dropdowns.

    Parameters
    ----------
    offline: bool
        Whether to run in offline mode (local file system) or online mode (connection to Alyx)
    config: bool
        Whether a config dropdown should be added to allow selection of different probe
        configurations
    parent: QtWidgets.QMainWindow
        The parent window

    Attributes
    ----------
    offline: bool
        Offline or online mode
    config: bool
        Whether a config dropdown is added
    dropdowns: dict
        A dictionary containing the added dropdowns as Bunch objects.
        Each Bunch has keys 'list' (list of options), 'combobox' (the combobox widget),
        and 'line' (the line edit widget, if applicable)
    buttons: dict
        A dictionary containing the added buttons as Bunch objects.
        Each Bunch has keys 'button' (the button widget)
        and line (the line edit widget, if applicable)
    button_style: dict
        A dictionary containing the stylesheets for activated and deactivated buttons
    """

    def __init__(
            self,
            offline: bool = False,
            config: bool = False,
            parent: QtWidgets.QMainWindow | None = None
    ):
        super().__init__(parent)

        self.offline: bool = offline
        self.config: bool = config
        self.dropdowns: dict[str, Bunch] = defaultdict(Bunch)
        self.buttons: dict[str, Bunch] = defaultdict(Bunch)
        self.button_style: dict = {
            'activated': """
            QPushButton {
                background-color: grey;
                border: 1px solid lightgrey;
                color: white;
                border-radius: 5px;  /* Rounded corners */
                padding: 2px;
            }
        """,
            'deactivated': """
            QPushButton {
                background-color: white;
                border: 1px solid transparent;
                color: grey;
                border-radius: 5px;  /* Rounded corners */
                padding: 2px;
            }
        """
        }

        self.create_widgets()
        self.layout_widgets()

    def create_widgets(self) -> None:
        """Create the dropdowns and buttons."""
        if not self.offline:
            # If offline mode is False, read in Subject and Session options from Alyx
            # Drop down list to choose subject
            subject_list, subject_combobox, subject_line, _ = self.create_combobox(editable=True)
            self.dropdowns['subject']['list'] = subject_list
            self.dropdowns['subject']['combobox'] = subject_combobox
            self.dropdowns['subject']['line'] = subject_line
            # Drop down list to choose session
            session_list, session_combobox, *_ = self.create_combobox()
            self.dropdowns['session']['list'] = session_list
            self.dropdowns['session']['combobox'] = session_combobox
        else:
            # If offline mode is True, provide dialog to select local folder that holds data
            self.buttons['folder']['line'] = QtWidgets.QLineEdit()
            self.buttons['folder']['button'] = QtWidgets.QToolButton()
            self.buttons['folder']['button'].setText('...')

        # Drop down list to choose previous alignments
        align_list, align_combobox, *_ = self.create_combobox()
        self.dropdowns['align']['list'] = align_list
        self.dropdowns['align']['combobox'] = align_combobox

        # Drop down list to select shank
        shank_list, shank_combobox, *_ = self.create_combobox()
        self.dropdowns['shank']['list'] = shank_list
        self.dropdowns['shank']['combobox'] = shank_combobox

        # Drop down list to select config
        config_list, config_combobox, *_ = self.create_combobox()
        self.dropdowns['config']['list'] = config_list
        self.dropdowns['config']['combobox'] = config_combobox

        # Load data button
        self.buttons['data']['button'] = QtWidgets.QPushButton('Load')
        self.buttons['data']['button'].setFixedWidth(70)
        self.buttons['data']['button'].setStyleSheet(self.button_style['deactivated'])

    def layout_widgets(self) -> None:
        """Layout the dropdowns and buttons."""
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        if not self.offline:
            layout.addWidget(self.dropdowns['subject']['combobox'])
            layout.addWidget(self.dropdowns['session']['combobox'])
            layout.addWidget(self.dropdowns['shank']['combobox'])
            layout.addWidget(self.dropdowns['align']['combobox'])
            layout.addWidget(self.buttons['data']['button'])
            if self.config:
                layout.addWidget(self.dropdowns['config']['combobox'])
        else:
            layout.addWidget(self.buttons['folder']['line'])
            layout.addWidget(self.buttons['folder']['button'])
            layout.addWidget(self.dropdowns['shank']['combobox'])
            layout.addWidget(self.dropdowns['align']['combobox'])
            layout.addWidget(self.buttons['data']['button'])

        self.setLayout(layout)

    def activate_data_button(self) -> None:
        """Change the style of the load data button to the activated style."""
        self.buttons['data']['button'].setStyleSheet(self.button_style['activated'])

    def deactivate_data_button(self) -> None:
        """Change the style of the load data button to the deactivated style."""
        self.buttons['data']['button'].setStyleSheet(self.button_style['deactivated'])

    @staticmethod
    def create_combobox(
            editable: bool = False
    ) -> tuple[QtGui.QStandardItemModel,
               QtWidgets.QComboBox, QtWidgets.QLineEdit | None,
               QtWidgets.QCompleter | None]:
        """
        Create a combobox with an optional editable line edit.

        Parameters
        ----------
        editable: bool
            Whether to add a line edit widget to the combobox

        Returns
        -------
        model: QtGui.QStandardItemModel
            The data model associated with the combobox. Items should be added to this model to
            populate the combobox.
        combobox: QtGui.QComboBox
            The combobox widget.
        line_edit: QtWidgets.QLineEdit or None
            The QLineEdit associated with the combobox if `editable=True`; otherwise None.
        completer: QtWidgets.QCompleter or None
        """
        model = QtGui.QStandardItemModel()
        combobox = QtWidgets.QComboBox()
        combobox.setModel(model)
        line_edit = None
        completer = None
        if editable:
            line_edit = QtWidgets.QLineEdit()
            combobox.setLineEdit(line_edit)
            completer = QtWidgets.QCompleter()
            completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
            combobox.setCompleter(completer)
            combobox.completer().setModel(model)

        return model, combobox, line_edit, completer

    @staticmethod
    def populate_combobox(
            data: list[str] | np.ndarray[str] | dict,
            list_name: QtGui.QStandardItemModel,
            combobox: QtWidgets.QComboBox,
            init=True
    ) -> None:
        """
        Populate a combobox and its associated model with a list or array of string options.

        Parameters
        ----------
        data : list, np.ndarray, or dict of strings
            A list of strings to add to the widget.
        list_name : QtGui.QStandardItemModel
            The model object to which items will be added.
        combobox : QtWidgets.QComboBox
            The combo box widget to be populated and configured.
        init: bool
            If init set the selected item to the first option in the list
        """
        list_name.clear()
        for dat in data:
            item = QtGui.QStandardItem(dat)
            item.setEditable(False)
            list_name.appendRow(item)

        # Ensure the drop-down menu is wide enough to display the longest string
        min_width = combobox.fontMetrics().width(max(data, key=len))
        min_width += combobox.view().autoScrollMargin()
        min_width += combobox.style().pixelMetric(QtWidgets.QStyle.PM_ScrollBarExtent)
        combobox.view().setMinimumWidth(min_width)

        # Set the default selected item to the first option, if available
        if init:
            combobox.setCurrentIndex(0)


class FitWidget(QtWidgets.QWidget):
    """
    A widget for displaying a fit plot with a checkbox in the top left corner of the display.

    Parameters
    ----------
    parent : QWidgets.QMainWindow, optional
        The parent window

    Attributes
    ----------
    fig_fit : pg.PlotWidget
        The plot widget displaying the fit.
    lin_fit_option : QtWidgets.QCheckBox
        A checkbox to toggle linear fitting.
    """

    def __init__(self, parent: QtWidgets.QMainWindow | None = None):
        super().__init__(parent)

        # Figure to show fit
        self.fig_fit = pg.PlotWidget(background='w')
        self.fig_fit.setMouseEnabled(x=False, y=False)
        self.fig_fit.sigDeviceRangeChanged.connect(self.on_fig_size_changed)
        axis_range = (-2000, 6000)
        self.fig_fit.setXRange(*axis_range)
        self.fig_fit.setYRange(*axis_range)
        set_axis(self.fig_fit, 'bottom', label='Original coordinates (um)')
        set_axis(self.fig_fit, 'left', label='New coordinates (um)')

        # Unity line
        plot = pg.PlotCurveItem()
        plot.setData(x=[1], y=[1], pen=pg.mkPen(color='k', style=QtCore.Qt.DotLine, width=2))
        self.fig_fit.addItem(plot)

        # Linear fit option checkbox
        self.lin_fit_option = QtWidgets.QCheckBox('Linear fit', self.fig_fit)
        self.lin_fit_option.setChecked(True)

        # Layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.fig_fit)
        self.setLayout(layout)

    def on_fig_size_changed(self) -> None:
        """Move the location of the checkbox when the figure size changes."""
        self.lin_fit_option.move(70, 10)




class LutWidget(pg.GraphicsLayoutWidget):
    """
    A widget that creates and manages a Histogram-based Lookup Table (LUT).

    The LUT is used to synchronize image intensity levels across multiple displayed images.

    Attributes
    ----------
    slice_lut : pg.HistogramLUTItem
        The histogram LUT item controlling intensity scaling.
    lut_layout : pg.GraphicsLayout
        Internal layout container for the LUT item.
    images : list
        A list of pyqtgraph ImageItem instances to which LUT levels are applied.
    lut_status : bool
        Whether the LUT is currently visible (True) or removed (False).
    lut_levels : tuple or None
        The current intensity level range applied to all images.
    """

    def __init__(self):
        super().__init__()

        # Create LUT histogram item
        self.slice_lut: pg.HistogramLUTItem = pg.HistogramLUTItem()
        self.slice_lut.axis.hide()
        self.slice_lut.sigLevelsChanged.connect(self.update_lut_levels)

        # Layout to hold LUT
        self.lut_layout: pg.GraphicsLayout = pg.GraphicsLayout()
        self.lut_layout.addItem(self.slice_lut)
        self.addItem(self.lut_layout)

        self.images: list = []
        self.lut_status: bool = True
        self.lut_levels: tuple | None = None

    def set_lut(self, imgs, cbar):
        """
        Associate a list of images with this LUT and apply a color map.

        Parameters
        ----------
        imgs : list
            A list of pyqtgraph ImageItem instances to be linked to this LUT.
        cbar : ColorBar
            A ColorBar object
        """
        if not imgs:
            return

        self.images = imgs

        self.slice_lut.blockSignals(True)
        self.slice_lut.setImageItem(imgs[0])  # Attach LUT to the first image
        self.slice_lut.gradient.setColorMap(cbar.cmap)
        self.slice_lut.autoHistogramRange()

        hist_levels = self.slice_lut.getLevels()
        hist_vals, hist_counts = imgs[0].getHistogram()

        # Attempt to estimate an upper level cutoff based on data frequency
        upper_idx_candidates = np.where(hist_counts > 10)[0]
        if len(upper_idx_candidates) > 0:
            upper_idx = upper_idx_candidates[-1]
            upper_val = hist_vals[upper_idx]
        else:
            upper_val = hist_levels[1]

        # If lower level is non-zero, adjust upper bound accordingly
        if hist_levels[0] != 0:
            self.set_lut_levels([hist_levels[0], upper_val])
        else:
            self.set_lut_levels()

        self.slice_lut.blockSignals(False)

    def add_lut(self) -> None:
        """Add the LUT item back into the layout (if previously removed)."""
        if not self.lut_status:
            self.lut_layout.addItem(self.slice_lut)
            self.lut_status = True

    def remove_lut(self) -> None:
        """Remove the LUT item from the layout (if not already removed)."""
        if self.lut_status:
            self.lut_layout.removeItem(self.slice_lut)
            self.lut_status = False

    def set_lut_levels(self, levels: list | tuple | None = None) -> None:
        """
        Apply the specified intensity levels to all linked images and update the LUT.

        Parameters
        ----------
        levels : tuple or list or None
            The (min, max) levels to apply. If None, uses the last known levels.
        """
        levels = levels or self.lut_levels
        if levels is None:
            return

        self.lut_levels = levels
        # Update all linked images
        for img in self.images:
            img.setLevels(levels)
        # Update the histogram LUT display
        self.slice_lut.setLevels(min=levels[0], max=levels[1])

    def update_lut_levels(self) -> None:
        """Update stored LUT levels from the HistogramLUTItem and apply to all images."""
        self.lut_levels = self.slice_lut.getLevels()
        for img in self.images:
            img.setLevels(self.lut_levels)


class MenuWidget(QtWidgets.QMenuBar):
    """
    A custom menu bar with tabs for different plot types and fit and display options.

    Parameters
    ----------
    parent: QtWidgets.QMainWindow
        The parent window

    Attributes
    ----------
    parent: QtWidgets.QMainWindow
        The parent window
    tabs: dict[str, Bunch]
        A dictionary containing the created tabs as Bunch objects.
        Each Bunch has keys 'menu' (the menu object)
    """

    def __init__(self, parent: QtWidgets.QMainWindow | None = None):
        super().__init__(parent)

        self.parent: QtWidgets.QMainWindow | None = parent
        self.tabs: dict[str, Bunch] = defaultdict(Bunch)

        self.setNativeMenuBar(False)
        self.setStyleSheet("""QMenuBar {padding-bottom: 10px;}""")

        self.create_tabs()

    def create_tabs(self):
        """Create tabs on the menu bar."""
        # Add tabs for following plot options
        # (these are exclusive, i.e. only one can be selected at a time)
        for group in ['image', 'line', 'probe', 'slice', 'filter']:
            self.tabs[group]['menu'] = self.addMenu(f'{group.capitalize()} Plots')
            self.tabs[group]['group'] = QtWidgets.QActionGroup(self.tabs[group]['menu'])
            self.tabs[group]['group'].setExclusive(True)

        # Add tab for fit and displat options (thse are non-exclusive)
        self.tabs['fit']['menu'] = self.addMenu("Fit Options")
        self.tabs['display']['menu'] = self.addMenu('Display Options')

    def populate_exclusive_tab(
            self,
            name: str,
            callback: Callable,
            options: list[str] =None,
            set_checked: bool = True
    ) -> str | None:
        """
        Populate an exclusive tab with actions.

        Parameters
        ----------
        name : str
            The name of the tab to populate.
        callback : Callable
            Function to call when an action is triggered.
        options : list of str, optional
            The options to add to the tab
        set_checked : bool, optional
            If True, the first action will be pre-checked.

        Returns
        -------
        str or None
            The name of the initially checked option, or None if no keys provided.
        """
        if set_checked:
            self.tabs[name]['menu'].clear()
            self.remove_actions(self.tabs[name]['group'])
        if options is not None:
            option_init = self.add_actions(
                options,
                callback,
                self.tabs[name]['menu'],
                self.tabs[name]['group'],
                data_only=True,
                set_checked=set_checked)
        else:
            option_init = None

        return option_init

    def populate_non_exclusive_tab(self, name: str, options: dict) -> None:
        """
        Populate a non-exclusive tab (regular menu) with actions and set shortcuts.

        Parameters
        ----------
        name : str
            The name of the tab to populate.
        options : dict
            Each item should be a tuple of:
                (label, {'callback': Callable, 'shortcut': str (optional)})
        """
        for key, val in options.items():
            option = QtWidgets.QAction(key, self.parent)
            shortcut = val.get('shortcut', None)
            if shortcut:
                option.setShortcut(shortcut)
            option.triggered.connect(val['callback'])
            self.tabs[name]['menu'].addAction(option)

    @staticmethod
    def add_actions(
            options: list[str],
            function: Callable,
            menu: QtWidgets.QMenu,
            group: QtWidgets.QActionGroup,
            set_checked: bool = True,
            **kwargs
    ) -> str:
        """
        Add a list of actions to a menu and its corresponding QActionGroup.

        Parameters
        ----------
        options : list of str
            Labels for the menu actions.
        function : Callable
            Function to connect to each action's triggered signal.
        menu : QtWidgets.QMenu
            The menu to which actions will be added.
        group : QtWidgets.QActionGroup
            The group for handling exclusive behavior.
        set_checked : bool, optional
            If True, the first item will be checked by default.
        **kwargs :
            data_only : bool, optional
                If True, pass this as an extra argument to the callback

        Returns
        -------
        action_init: str
            The name of the first action added (used as initial selection).
        """
        data_only = kwargs.get('data_only', False)
        action_init = None

        for i, option in enumerate(options):
            checked = set_checked and i == 0

            action = QtWidgets.QAction(option, checkable=True, checked=checked)
            if data_only:
                action.triggered.connect(lambda _, o=option: function(o, data_only=data_only))
            else:
                action.triggered.connect(lambda _, o=option: function(o))
            menu.addAction(action)
            group.addAction(action)
            if i == 0:
                action_init = option

        return action_init

    @staticmethod
    def remove_actions(action_group: QtWidgets.QActionGroup) -> None:
        """
        Remove and delete all actions from the provided QActionGroup.

        Parameters
        ----------
        action_group : QtWidgets.QActionGroup
            The group from which to remove all actions.
        """
        for action in list(action_group.actions()):
            action_group.removeAction(action)
            del action

    @staticmethod
    def find_actions(text: str, action_group: QtWidgets.QActionGroup) -> QtWidgets.QAction | None:
        """
        Find an action by its label text in a QActionGroup.

        Parameters
        ----------
        text : str
            The text label of the action to find.
        action_group : QtWidgets.QActionGroup
            The group in which to search for the action.

        Returns
        -------
        QtWidgets.QAction or None
            The matching QAction if found, else None.
        """
        for action in action_group.actions():
            if action.text() == text:
                return action
        return None

    @staticmethod
    def toggle_action(action_group: QtWidgets.QActionGroup) -> None:
        """
        Toggle through the actions in a QActionGroup, activating the next action in sequence.

        Parameters
        ----------
        action_group : QtWidgets.QActionGroup
            The group of QAction items representing plots to toggle through
        """
        current_act = action_group.checkedAction()
        actions = action_group.actions()
        current_idx = next(i for i, act in enumerate(actions) if act == current_act)
        next_idx = np.mod(current_idx + 1, len(actions))
        actions[next_idx].setChecked(True)
        actions[next_idx].trigger()


class ConfigWidget(QtWidgets.QWidget):
    """
    Abstract base widget for displaying electrophysiology and histogram figures.

    Creates two pg.GraphicsLayoutWidget, one with the histology figures and one with
    electrophysiology figures for a given shank and configuration.

    Subclasses must implement `get_layout` and `create_ephys_figure_layout` to provide the
    specific figure arrangements.

    Parameters
    ----------
    parent : QWidgets.QMainWindow, optional
        The parent window

    Attributes
    ----------
    ephys_area : pg.GraphicsLayoutWidget
        The widget area for electrophysiology figures.
    hist_area : pg.GraphicsLayoutWidget
        The widget area for histology figures.
    """

    def __init__(self, parent: QtWidgets.QMainWindow | None = None):
        super().__init__(parent)

        # Get layouts from subclass
        ephys_layout, hist_layout = self.get_layout()

        # Create figure areas
        self.ephys_area = self.create_figure_area(ephys_layout)
        self.hist_area = self.create_figure_area(hist_layout, tracking=True)

        # Combine figure areas into a single horizontal container
        fig_area = QtWidgets.QWidget()
        fig_area_layout = QtWidgets.QHBoxLayout()
        fig_area_layout.setContentsMargins(0, 0, 0, 0)
        fig_area_layout.setSpacing(0)
        fig_area_layout.addWidget(self.ephys_area)
        fig_area_layout.addWidget(self.hist_area)
        fig_area_layout.setStretch(0, 3)
        fig_area_layout.setStretch(1, 1)
        fig_area.setLayout(fig_area_layout)

        # Main layout: header (from subclass) + figure area
        self.setContentsMargins(0, 0, 0, 0)
        shank_layout = QtWidgets.QVBoxLayout()
        shank_layout.setContentsMargins(0, 0, 0, 0)
        shank_layout.setSpacing(0)
        shank_layout.addWidget(self.header)
        shank_layout.addWidget(fig_area)
        self.setLayout(shank_layout)

    @abstractmethod
    def get_layout(self):
        """Return the electrophysiology and histogram layouts."""

    @abstractmethod
    def create_ephys_figure_layout(self, *args):
        """Create and return the electrophysiology figure layout."""

    @staticmethod
    def create_hist_figure_layout(items) -> pg.GraphicsLayout:
        """
        Build a histology figure layout for a single configuration.

        Parameters
        ----------
        items: ShankView
            A ShankView object containing all the figure items for this configuration and shank.

        Returns
        -------
        fig_hist_layout: pg.GraphicsLayout
            The created histology figure layout.
        """
        fig_hist_layout = pg.GraphicsLayout()
        fig_hist_layout.setSpacing(0)
        # Add items to layout with positions and spans
        fig_hist_layout.addItem(items.fig_scale_cb, 0, 0, 1, 4)
        fig_hist_layout.addItem(items.fig_hist_extra_yaxis, 1, 0)
        fig_hist_layout.addItem(items.fig_hist, 1, 1)
        fig_hist_layout.addItem(items.fig_scale, 1, 2)
        fig_hist_layout.addItem(items.fig_hist_ref, 1, 3)
        # Set column and row stretch factors
        fig_hist_layout.layout.setColumnStretchFactor(0, 1)
        fig_hist_layout.layout.setColumnStretchFactor(1, 4)
        fig_hist_layout.layout.setColumnStretchFactor(2, 1)
        fig_hist_layout.layout.setColumnStretchFactor(3, 4)
        fig_hist_layout.layout.setRowStretchFactor(0, 1)
        fig_hist_layout.layout.setRowStretchFactor(1, 10)
        fig_hist_layout.layout.setHorizontalSpacing(0)

        return fig_hist_layout

    @staticmethod
    def create_figure_area(
            layout: pg.GraphicsLayout,
            tracking: bool = False
    ) -> pg.GraphicsLayoutWidget:
        """
        Wrap a GraphicsLayout in a GraphicsLayoutWidget for display.

        Parameters
        ----------
        layout : pg.GraphicsLayout
            The layout to display.
        tracking : bool, optional
            Whether to enable mouse tracking on the widget.

        Returns
        -------
        pg.GraphicsLayoutWidget
            The created figure area widget.
        """
        area = pg.GraphicsLayoutWidget(border=None)
        area.setContentsMargins(0, 0, 0, 0)
        area.ci.setContentsMargins(0, 0, 0, 0)
        if tracking:
            area.setMouseTracking(True)
        area.addItem(layout)

        return area

    def setup_double_click(self, func_click: Callable) -> None:
        """
        Connect double-click events for both figure areas.

        Parameters
        ----------
        func_click : callable
            The callback to connect to
        """
        self.ephys_area.scene().sigMouseClicked.connect(
            lambda event, i=self.idx: func_click(event, i))
        self.hist_area.scene().sigMouseClicked.connect(
            lambda event, i=self.idx: func_click(event, i))

    def setup_mouse_hover(self, func_hover: Callable) -> None:
        """
        Connect mouse-hover events for both figure areas.

        Parameters
        ----------
        func_hover : callable
            The callback to connect to
        """
        self.ephys_area.scene().sigMouseHover.connect(
            lambda hover_items, n=self.name, i=self.idx, c=self.config:
            func_hover(hover_items, n, i, c))
        self.hist_area.scene().sigMouseHover.connect(
            lambda hover_items, n=self.name, i=self.idx, c=self.config:
            func_hover(hover_items, n, i, c))

class SingleConfigWidget(ConfigWidget):
    """
    Widget for displaying ephys and histology for a single shank and configuration.

    Parameters
    ----------
    items: ShankView
        A ShankView object containing all the figure items for this configuration and shank.
    parent : QWidgets.QMainWindow, optional
        The parent window

    Attributes
    ----------
    items: ShankView
        A ShankView object containing all the figure items for this configuration and shank.
    header : QtWidgets.QLabel
        A label widget for the header, provided by the subclass.
    config : str
        The probe configuration name, provided by the subclass.
    idx : int
        The index of the shank, provided by the subclass.
    name : str
        The name of the shank, provided by the subclass.
    """

    def __init__(self, items, parent: QtWidgets.QMainWindow | None = None):
        self.items = items
        self.config: str = items.config
        self.idx: int = items.index
        self.name: str = items.name
        self.header: QtWidgets.QLabel = items.header

        super().__init__(parent)

    def create_ephys_figure_layout(self, items) -> pg.GraphicsLayout:
        """
        Build an ephys figure layout for a single configuration.

        Parameters
        ----------
        items: ShankView
            A ShankView object containing all the figure items for this configuration and shank.

        Returns
        -------
        fig_ephys_layout: pg.GraphicsLayout
            The created ephys figure layout.
        """
        items.fig_data_ax = set_axis(items.fig_img, 'left', label='Distance from probe tip (uV)')
        set_axis(items.fig_scale_cb, 'bottom', show=False)

        fig_ephys_layout = pg.GraphicsLayout()
        fig_ephys_layout.setSpacing(0)

        # Add items to layout with positions and spans
        fig_ephys_layout.addItem(items.fig_img_cb, 0, 0)
        fig_ephys_layout.addItem(items.fig_probe_cb, 0, 1, 1, 2)
        fig_ephys_layout.addItem(items.fig_img, 1, 0)
        fig_ephys_layout.addItem(items.fig_line, 1, 1)
        fig_ephys_layout.addItem(items.fig_probe, 1, 2)

        # Set column and row stretch factors
        fig_ephys_layout.layout.setColumnStretchFactor(0, 6)
        fig_ephys_layout.layout.setColumnStretchFactor(1, 2)
        fig_ephys_layout.layout.setColumnStretchFactor(2, 1)
        fig_ephys_layout.layout.setRowStretchFactor(0, 1)
        fig_ephys_layout.layout.setRowStretchFactor(1, 10)

        return fig_ephys_layout

    def get_layout(self) -> tuple[pg.GraphicsLayout, pg.GraphicsLayout]:
        """
        Create the electrophysiology and histogram layouts.

        Returns
        -------
        ephys_layout: pg.GraphicsLayout
            The created ephys figure layout.
        hist_layout: pg.GraphicsLayout
            The created histology figure layout.
        """
        ephys_layout = self.create_ephys_figure_layout(self.items)
        hist_layout = self.create_hist_figure_layout(self.items)

        return ephys_layout, hist_layout


class DualConfigWidget(ConfigWidget):
    """
    Widget for displaying ephys and histology for a single shank and two different configurations.

    The histology figure is built from the figure items of the default configuration.
    The ephys figure shows both the figure items from the default and non-default configurations
    side by side in one panel.

    Parameters
    ----------
    items_default: ShankView
        A ShankView object containing all the figure items for the default configuration
        and shank.
    items_non_default: ShankView
        A ShankView object containing all the figure items for the non-default configuration
        and shank
    parent : QWidgets.QMainWindow, optional
        The parent window

    Attributes
    ----------
    items_default: ShankView
        A ShankView object containing all the figure items for the default configuration
        and shank.
    items_non_default: ShankView
        A ShankView object containing all the figure items for the non-default configuration
        and shank
    header : QtWidgets.QLabel
        A label widget for the header, provided by the subclass.
    config : str
        The probe configuration name, provided by the subclass.
    idx : int
        The index of the shank, provided by the subclass.
    name : str
        The name of the shank, provided by the subclass.
    """

    def __init__(self,
                 items_default,
                 items_non_default,
                 parent: QtWidgets.QMainWindow | None = None):

        self.items_default = items_default
        self.items_non_default = items_non_default
        self.config: str = items_default.config
        self.idx: int = items_default.index
        self.name: str = items_default.name
        self.header: QtWidgets.QLabel = items_default.header

        super().__init__(parent)

    def create_ephys_figure_layout(
            self,
            items_default,
            items_non_default
    ) -> pg.GraphicsLayout:
        """
        Build an ephys figure layout showing two configurations alongside each other.

        Parameters
        ----------
        items_default: ShankView
            A ShankView object containing all the figure items for the default configuration
            and shank.
        items_non_default: ShankView
            A ShankView object containing all the figure items for the non-default configuration
            and shank

        Returns
        -------
        fig_ephys_layout: pg.GraphicsLayout
            The created ephys figure layout.
        """
        # Configure axes for both sets
        items_non_default.fig_data_ax = set_axis(items_non_default.fig_img, 'left', show=False)
        items_default.fig_data_ax = set_axis(
            items_default.fig_img, 'left', label='Distance from probe tip (uV)')
        set_axis(items_default.fig_scale_cb, 'bottom')

        # Link the y-axis
        items_default.fig_img.setYLink(items_non_default.fig_img)

        # Shared colorbar axes
        fig_dual_img_cb = pg.PlotItem()
        fig_dual_img_cb.setMouseEnabled(x=False, y=False)
        fig_dual_img_cb.setMaximumHeight(70)
        set_axis(fig_dual_img_cb, 'left', pen='w')
        set_axis(fig_dual_img_cb, 'top', pen='w')
        items_non_default.fig_dual_img_cb = fig_dual_img_cb
        items_default.fig_dual_img_cb = fig_dual_img_cb

        fig_dual_probe_cb = pg.PlotItem()
        fig_dual_probe_cb.setMouseEnabled(x=False, y=False)
        fig_dual_probe_cb.setMaximumHeight(70)
        set_axis(fig_dual_probe_cb, 'left', pen='w')
        set_axis(fig_dual_probe_cb, 'top', pen='w')
        items_non_default.fig_dual_probe_cb = fig_dual_probe_cb
        items_default.fig_dual_probe_cb = fig_dual_probe_cb

        # Layout arrangement
        fig_ephys_layout = pg.GraphicsLayout()
        fig_ephys_layout.setSpacing(0)

        # Add items to layout with positions and spans
        fig_ephys_layout.addItem(fig_dual_img_cb, 0, 0, 1, 2)
        fig_ephys_layout.addItem(fig_dual_probe_cb, 0, 3, 1, 3)
        fig_ephys_layout.addItem(items_default.fig_img, 1, 0)
        fig_ephys_layout.addItem(items_non_default.fig_img, 1, 1)
        fig_ephys_layout.addItem(items_default.fig_line, 1, 2)
        fig_ephys_layout.addItem(items_non_default.fig_line, 1, 3)
        fig_ephys_layout.addItem(items_default.fig_probe, 1, 4)
        fig_ephys_layout.addItem(items_non_default.fig_probe, 1, 5)

        # Set column and row stretch factors
        fig_ephys_layout.layout.setColumnStretchFactor(0, 5)
        fig_ephys_layout.layout.setColumnStretchFactor(1, 5)
        fig_ephys_layout.layout.setColumnStretchFactor(2, 1)
        fig_ephys_layout.layout.setColumnStretchFactor(3, 1)
        fig_ephys_layout.layout.setColumnStretchFactor(4, 1)
        fig_ephys_layout.layout.setColumnStretchFactor(5, 1)
        fig_ephys_layout.layout.setRowStretchFactor(0, 1)
        fig_ephys_layout.layout.setRowStretchFactor(1, 10)

        return fig_ephys_layout

    def get_layout(self) -> tuple[pg.GraphicsLayout, pg.GraphicsLayout]:
        """
        Create the electrophysiology and histogram layouts.

        Returns
        -------
        ephys_layout: pg.GraphicsLayout
            The created ephys figure layout.
        hist_layout: pg.GraphicsLayout
            The created histology figure layout.
        """
        ephys_layout = self.create_ephys_figure_layout(self.items_default, self.items_non_default)
        hist_layout = self.create_hist_figure_layout(self.items_default)

        return ephys_layout, hist_layout
