from collections.abc import Callable
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from qtpy import QtCore, QtWidgets

from ibl_alignment_gui.utils.qt import custom_widgets
from iblutil.util import Bunch

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')


class AlignmentGUIView(QtWidgets.QMainWindow):
    """
    The main GUI window for the alignment application.

    Parameters
    ----------
    offline: bool
        Whether to run in offline mode (local files) or online mode (ONE/Alyx)
    config: bool
        Whether multiple configs are to be used
    """

    def __init__(self, offline: bool = False, config: bool = False):
        super().__init__()
        self.config = config

        self.resize(1600, 800)
        self.setWindowTitle('IBL alignment GUI')
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        # Create custom widgets that will be added to the main window
        self.button_widgets = custom_widgets.ButtonWidget(parent=self)
        self.selection_widgets = custom_widgets.SelectionWidget(
            offline=offline, config=self.config, parent=self
        )
        self.menu_widgets = custom_widgets.MenuWidget(self)
        self.setMenuBar(self.menu_widgets)
        self.menu_widgets.setCornerWidget(self.selection_widgets, corner=QtCore.Qt.TopRightCorner)
        self.tab_widgets = Bunch()
        self.tab_widgets['shank'] = custom_widgets.GridTabSwitcher()
        self.tab_widgets['slice'] = custom_widgets.GridTabSwitcher()
        self.lut_widget = custom_widgets.LutWidget()
        self.fit_widget = custom_widgets.FitWidget()

        # Layout the widgets
        # Group together the slice tabs and the lut widget
        slice_area = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.tab_widgets['slice'], stretch=4)
        layout.addWidget(self.lut_widget, stretch=1)
        slice_area.setLayout(layout)

        # Add the slice area and fit plot into a splitter
        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        splitter.addWidget(slice_area)
        splitter.addWidget(self.fit_widget)

        # Add this splitter to a layout with the button widgets below
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(splitter, stretch=6)
        layout.addWidget(self.button_widgets, stretch=1)
        layout.setContentsMargins(0, 0, 0, 0)
        widget.setLayout(layout)

        # Add these onto a main splitter that contains all the individual components
        self.main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.main_splitter.addWidget(self.tab_widgets['shank'])
        self.main_splitter.addWidget(widget)
        self.main_splitter.setStretchFactor(0, 2)
        self.main_splitter.setStretchFactor(1, 1)

    def init_view(self) -> None:
        """Initialize the main view by setting the central widget and the splitter sizes."""
        self.setCentralWidget(self.main_splitter)
        total_width = self.main_splitter.width()
        self.main_splitter.setSizes([int(total_width * 0.7), int(total_width * 0.3)])

    def reset_view(self) -> None:
        """Reset the main view by clearing the fit plot and all the tab widgets."""
        self.fit_widget.fig_fit.clear()
        self.clear_tabs()

    def focus(self) -> None:
        """Ensure the main widget is set as the main focus."""
        if 'subject' in self.selection_widgets.dropdowns:
            self.selection_widgets.dropdowns['subject']['line'].clearFocus()
            self.selection_widgets.dropdowns['subject']['combobox'].clearFocus()
        self.setFocus()
        self.raise_()
        self.activateWindow()

    # --------------------------------------------------------------------------------------------
    # GridTab widget
    # --------------------------------------------------------------------------------------------
    def clear_tabs(self) -> None:
        """Clear the shank and slice tab widgets of all their contents."""
        self.tab_widgets['shank'].tab_widget.blockSignals(True)
        self.tab_widgets['slice'].tab_widget.blockSignals(True)
        self.tab_widgets['shank'].delete_widgets()
        self.tab_widgets['slice'].delete_widgets()
        self.tab_widgets['shank'].tab_widget.blockSignals(False)
        self.tab_widgets['slice'].tab_widget.blockSignals(False)

    def init_tabs(
            self,
            shank_items: dict | Bunch,
            selected_config: str,
            default_config: str,
            non_default_config: str,
            feature_view: bool = False
    ) -> list[QtWidgets.QWidget]:
        """
        Initialize the shank and slice tab widgets with the given shank items.

        Parameters
        ----------
        shank_items: dict | Bunch
            A dictionary of ShankController instances for each shank
        selected_config:
            The configuration to display. Can be 'default', 'non-default' or 'both'
        default_config:
            The default configuration.
        non_default_config
            The non-default configuration.
        feature_view: bool
            Whether to display the ephys plot as a feature view or not.

        Returns
        -------
        list[QtWidgets.QWidget]:
            A list of the shank panels created
        """
        shank_panels = []
        slice_panels = []
        headers = []

        config = default_config if selected_config == 'both' else selected_config

        for i, shank in enumerate(shank_items):

            # Create the shank panel depending on the configuration selected
            if selected_config == 'both':
                if feature_view:
                    fig_area = custom_widgets.DualConfigFeatureWidget(
                        shank_items[shank][default_config].view,
                        shank_items[shank][non_default_config].view)
                else:
                    fig_area = custom_widgets.DualConfigWidget(
                        shank_items[shank][default_config].view,
                        shank_items[shank][non_default_config].view)
            elif feature_view:
                fig_area = custom_widgets.SingleConfigFeatureWidget(
                    shank_items[shank][selected_config].view)
            else:
                fig_area = custom_widgets.SingleConfigWidget(
                    shank_items[shank][selected_config].view)

            # Add the fit items from each shank to the fit plot
            self.fit_widget.fig_fit.addItem(shank_items[shank][config].view.fit_plot)
            self.fit_widget.fig_fit.addItem(shank_items[shank][config].view.fit_scatter)
            self.fit_widget.fig_fit.addItem(shank_items[shank][config].view.fit_plot_lin)

            # Link the histology views so they pan and zoom together
            if i == 0:
                slice_link = shank_items[shank][config].view.fig_slice
            if i > 0:
                shank_items[shank][config].view.fig_slice.setYLink(slice_link)
                shank_items[shank][config].view.fig_slice.setXLink(slice_link)

            headers.append(shank_items[shank][config].view.header)
            slice_panels.append(shank_items[shank][config].view.fig_slice_area)
            shank_panels.append(fig_area)

        # Add the panels to the tab widgets
        self.tab_widgets['shank'].initialise(shank_panels, shank_items.keys(), headers)
        self.tab_widgets['slice'].initialise(slice_panels, shank_items.keys())

        return shank_panels

    def toggle_tabs(self, idx: int) -> None:
        """
        Toggle the shank and slice tabs between grid and tab view.

        After toggling ensure the correct index is selected.

        Parameters
        ----------
        idx: int
            The index of the selected panel
        """
        self.tab_widgets['slice'].toggle_layout()
        self.tab_widgets['slice'].tab_widget.setCurrentIndex(idx)
        # Change the display of the shank displays
        self.tab_widgets['shank'].toggle_layout()
        self.tab_widgets['shank'].tab_widget.setCurrentIndex(idx)

    @property
    def is_grid(self) -> bool:
        """
        Check if the shank tab view is in grid layout.

        Returns
        -------
        bool:
            True if the shank tab view is in grid layout, False if in tab layout
        """
        return self.tab_widgets['shank'].grid_layout

    def set_tabs(self, idx: int) -> None:
        """
        Set the index of the selected tab in the shank and slice tab views.

        Parameters
        ----------
        idx:
            The index to set the tabs to
        """
        if not self.tab_widgets['shank'].grid_layout:
            self.tab_widgets['shank'].tab_widget.blockSignals(True)
            self.set_slice_tab(idx)
            self.set_shank_tab(idx)
            self.tab_widgets['slice'].tab_widget.setCurrentIndex(idx)
            self.tab_widgets['shank'].tab_widget.blockSignals(False)

    def set_slice_tab(self, idx: int):
        """
        Set the tab of the slice tabs to the given index.

        Parameters
        ----------
        idx:
            The index to set the tab to
        """
        self.tab_widgets['slice'].tab_widget.setCurrentIndex(idx)

    def set_shank_tab(self, idx: int) -> None:
        """
        Set the tab of the shank tabs to the given index.

        Parameters
        ----------
        idx:
            The index to set the tab to
        """
        self.tab_widgets['shank'].tab_widget.setCurrentIndex(idx)

    def connect_tabs(
            self,
            name: str,
            callback: Callable,
            layout_callback: Callable | None = None
    ) -> None:
        """
        Connect the tab change signal to a callback.

        Parameters
        ----------
        name: str
            The name of the tab widget
        callback:  Callable
            The tab change callback to connect to
        layout_callback: Callable, optional
            The layout chance callback to connect to
        """
        self.tab_widgets[name].tab_widget.currentChanged.connect(callback)
        if layout_callback:
            self.tab_widgets[name].custom_signal.connect(layout_callback)

    # --------------------------------------------------------------------------------------------
    # Menu widget
    # --------------------------------------------------------------------------------------------
    def populate_menu_tab(self, tab: str, callback: Callable, options: list[str],
                          set_checked: bool = True) -> str | None:
        """"See :meth:`MenuWidget.populate_exclusive_tab` for details."""
        return self.menu_widgets.populate_exclusive_tab(
            tab, callback, options, set_checked=set_checked)

    def add_shortcuts_to_menu(self, tab: str, options: dict) -> None:
        """"See :meth:`MenuWidget.populate_non_exclusive_tab` for details."""
        return self.menu_widgets.populate_non_exclusive_tab(tab, options)

    def trigger_menu_option(self, tab: str, option: str) -> None:
        """
        Trigger the selection of an action in an action group stored in the menubar.

        Parameters
        ----------
        tab: str
            The name of the tab
        option: str
            The name of the option in the tab to trigger
        """
        if option:
            self.menu_widgets.find_actions(option, self.menu_widgets.tabs[tab]['group']).trigger()

    def toggle_menu_option(self, tab: str, direction: int) -> None:
        """
        Toggle through the options in an action group stored in the menubar.

        Parameters
        ----------
        tab: str
            The name of the tab
        direction: int
            The direction to toggle in the action group (1 for next, -1 for previous)
        """
        self.menu_widgets.toggle_action(self.menu_widgets.tabs[tab]['group'], direction)

    # --------------------------------------------------------------------------------------------
    # Button widget
    # --------------------------------------------------------------------------------------------
    def connect_button(self, name, callback: Callable) -> None:
        """
        Connect a button to a callback.

        Parameters
        ----------
        name: str
            The name of the button
        callback: Callable
            The callback function to connect to
        """
        self.button_widgets.buttons[name].clicked.connect(callback)

    def set_labels(self, current_idx: int, total_idx: int) -> None:
        """
        Set the strings to indicate the number of fits applied by the user.

        Parameters
        ----------
        current_idx: int
            The current index of the fit in the alignment buffer
        total_idx: int
            The total number of fits stores in the alignment buffer
        """
        self.button_widgets.labels['current'].setText(f"Current Index = {current_idx}")
        self.button_widgets.labels['total'].setText(f"Total Index = {total_idx}")

    def add_all_button(self) -> None:
        """See :meth:`ButtonWidget.add_all_button` for details."""
        self.button_widgets.add_all_button()

    # --------------------------------------------------------------------------------------------
    # Selection widget
    # --------------------------------------------------------------------------------------------
    def set_selection_dropdown(self, name: str, idx: int) -> None:
        """
        Set the dropdown to a given index.

        Parameters
        ----------
        name: str
            The name of the dropdown
        idx
            The index to set
        """
        self.selection_widgets.dropdowns[name]['combobox'].setCurrentIndex(idx)

    def connect_selection_dropdown(self, name: str, callback: Callable) -> None:
        """
        Connect a dropdown to a callback.

        Parameters
        ----------
        name: str
            The name of the dropdown
        callback: Callable
            The callback function to connect to
        """
        self.selection_widgets.dropdowns[name]['combobox'].activated.connect(callback)

    def populate_selection_dropdown(self, name: str, values: list | dict | np.ndarray) -> None:
        """
        Populate a dropdown with values.

        Parameters
        ----------
        name: str
            The name of the dropdown to populate
        values: list or dict or np.ndarray
            The values to add to the list
        """
        self.selection_widgets.populate_combobox(
            values, self.selection_widgets.dropdowns[name]['list'],
            self.selection_widgets.dropdowns[name]['combobox'])

    def clear_selection_dropdown(self, name: str | list) -> None:
        """
        Clear values from a list or a set of lists.

        Parameters
        ----------
        name: str or list
            The name of the dropdown or dropdowns to clear
        """
        if isinstance(name, str):
            self.selection_widgets.dropdowns[name]['list'].clear()
        elif isinstance(name, list | tuple):
            for n in name:
                self.selection_widgets.dropdowns[n]['list'].clear()

    def connect_selection_button(self, name: str, callback: Callable) -> None:
        """
        Connect the data button to a callback.

        Parameters
        ----------
        name: str
            The name of the button
        callback: Callable
            The callback function to connect to
        """
        self.selection_widgets.buttons[name]['button'].clicked.connect(callback)

    def activate_selection_button(self) -> None:
        """Change the stylesheet of the data button to show it is activated."""
        self.selection_widgets.activate_data_button()

    def deactivate_selection_button(self) -> None:
        """Change the stylesheet of the data button to show it is deactivated."""
        self.selection_widgets.deactivate_data_button()

    def get_selected_path(self) -> Path:
        """
        Get the user selected path and set the text line edit to show the selected folder path.

        Returns
        -------
        selected_path: Path
            The user selected path that contains data to load
        """
        selected_path = Path(QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Folder'))
        self.selection_widgets.buttons['folder']['line'].setText(str(selected_path))
        return selected_path

    def set_selected_path(self, selected_path: Path | str) -> None:
        """
        Set the text line edit to show the selected folder path.

        Parameters
        ----------
        selected_path: Path or str
            The user selected path that contains data to load
        """
        self.selection_widgets.buttons['folder']['line'].setText(str(selected_path))

    # --------------------------------------------------------------------------------------------
    # LUT widget
    # --------------------------------------------------------------------------------------------
    def set_levels(self, levels) -> None:
        """See :meth:`LutWidget.set_lut_levels` for details."""
        self.lut_widget.set_lut_levels(levels)

    def set_lut(self, images: list, cbar: custom_widgets.ColorBar) -> None:
        """
        Add the LUT item if not already added and set the LUT for the given images and colorbar.

        Parameters
        ----------
        images : list
            A list of pyqtgraph ImageItem instances to be linked to the LUT.
        cbar : ColorBar
            A ColorBar object
        """
        self.lut_widget.add_lut()
        self.lut_widget.set_lut(images, cbar)

    def remove_lut(self) -> None:
        """See :meth:`LutWidget.remove_lut` for details."""
        self.lut_widget.remove_lut()

    # --------------------------------------------------------------------------------------------
    # Fit widget
    # --------------------------------------------------------------------------------------------
    def connect_lin_fit(self, callback: Callable) -> None:
        """
        Connect the linear fit checkbox to a callback.

        Parameters
        ----------
        callback: Callable
            The callback function to connect to
        """
        self.fit_widget.lin_fit_option.stateChanged.connect(callback)

    def add_points_to_display(self, points: list[pg.PlotDataItem]) -> None:
        """
        Add a list of points to the fit plot.

        Parameters
        ----------
        points: list[pg.PlotDataItem]
            A list of points to add to the fit plot
        """
        for point in points:
            self.add_point(point)

    def remove_points_from_display(self, points: list[pg.PlotDataItem]) -> None:
        """
        Remove a list of points to the fit plot.

        Parameters
        ----------
        points: list[pg.PlotDataItem]
            A list of points to remove to the fit plot
        """
        for point in points:
            self.remove_point(point)

    def remove_point(self, point: pg.PlotDataItem) -> None:
        """
        Add a point to the fit plot.

        Parameters
        ----------
        point: pg.PlotDataItem
            A point to add to the fit plot
        """
        self.fit_widget.fig_fit.removeItem(point)

    def add_point(self, point: pg.PlotDataItem) -> None:
        """
        Remove a point to the fit plot.

        Parameters
        ----------
        point: pg.PlotDataItem
            A point to remove to the fit plot
        """
        self.fit_widget.fig_fit.addItem(point)

    # --------------------------------------------------------------------------------------------
    # Upload dialog boxes
    # --------------------------------------------------------------------------------------------
    def upload_prompt(self) -> bool:
        """
        Show a message box to ask the user if they want to upload the channels and alignments.

        Returns
        -------
        bool:
            True if the user wants to upload the channels and alignments, False otherwise
        """
        upload = QtWidgets.QMessageBox.question(
            self, '', "Upload alignment?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        return upload == QtWidgets.QMessageBox.Yes

    def upload_info(self, uploaded: bool, info: str | None = None) -> None:
        """
        Show a message box to inform the user of the upload status.

        Parameters
        ----------
        uploaded: bool
            Whether the channels and alignments were saved.
        info: str or None
            The message to display to the user
        """
        if uploaded:
            QtWidgets.QMessageBox.information(self, 'Status', info)
        else:
            QtWidgets.QMessageBox.information(self, 'Status', "Channels not saved")
