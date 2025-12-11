from one.api import ONE
from brainbox.io.one import SpikeSortingLoader
import iblqt.tools

import ibl_alignment_gui.app.app_controller as app_controller


class IBLAlignmentGUIController(app_controller.AlignmentGUIController):

    def load_pid(self, pid, one=None):
        """
        Launch the Alignment GUI and automatically load data for a specific probe insertion.

        This function creates an Alignment GUI Controller, retrieves session information
        for the given probe insertion ID (pid), and programmatically navigates the GUI
        to select the corresponding subject and session. It then triggers the data loading
        process for the selected probe insertion.

        Parameters
        ----------
        pid : str or UUID
            The probe insertion ID (pid) for which to load spike sorting data and
            display in the alignment GUI.
        one : ONE, optional
            An instance of the ONE API client. If None, a new ONE instance will be
            created using default settings. Default is None.

        Returns
        -------
        app_controller.AlignmentGUIController
            An instance of the AlignmentGUIController with its view displayed.
        """
        one = one if one is not None else ONE()
        ssl = SpikeSortingLoader(one=one, pid=pid)
        session_info = one.eid2ref(ssl.eid)

        # now we pilot the QUI to select the dataset
        cbs = self.view.selection_widgets.dropdowns['subject']['combobox']
        for i in range(cbs.count()):
            if cbs.itemText(i) == session_info['subject']:
                break
        cbs.setCurrentIndex(i)
        print(cbs.currentIndex(), cbs.currentText())
        self.on_subject_selected(cbs.currentIndex())

        cbs = self.view.selection_widgets.dropdowns['session']['combobox']
        for i in range(cbs.count()):
            if cbs.itemText(i) == f'{str(session_info['date'])} {ssl.pname}':
                break
        cbs.setCurrentIndex(i)
        print(cbs.currentIndex(), cbs.currentText())
        self.on_session_selected(cbs.currentIndex())
        self.data_button_pressed()


def alignment_gui() -> app_controller.AlignmentGUIController:
    """
    Create and display an Alignment GUI Controller.

    This function initializes the Qt application (if not already created), creates
    an instance of the AlignmentGUIController, displays its view, and returns the
    controller instance.

    Returns
    -------
    app_controller.AlignmentGUIController
        An instance of the AlignmentGUIController with its view displayed.
    """
    app = iblqt.tools.get_or_create_app()
    agc = IBLAlignmentGUIController()
    agc.view.show()
    return agc
