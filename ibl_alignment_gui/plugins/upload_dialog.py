from typing import TYPE_CHECKING

from qtpy import QtWidgets

from ibl_alignment_gui.utils.qt.custom_widgets import CheckBoxGroup

if TYPE_CHECKING:
    from ibl_alignment_gui.app.app_controller import AlignmentGUIController

PLUGIN_NAME = "Upload dialog"


def setup(controller: 'AlignmentGUIController'):
    """
    Set up the upload dialog and connect its accepted signal to the callback.

    Parameters
    ----------
    controller: AlignmentController
        The main application controller.
    """
    controller.upload_dialog = UploadDialog(controller)


def display(controller: 'AlignmentGUIController'):
    """
    Configure and show the upload dialog.

    Parameters
    ----------
    controller:
        The main application controller.
    """
    controller.upload_dialog.setup()
    controller.upload_dialog.exec_()
    return controller.upload_dialog.shanks_to_upload


class UploadDialog(QtWidgets.QDialog):
    """
    Dialog for selecting shanks to upload.

    Parameters
    ----------
    controller: AlignmentGUIController
        The main application controller.
    """

    def __init__(self, controller: 'AlignmentGUIController'):
        super().__init__(controller.view)

        self.controller = controller
        self.setWindowTitle('Upload shanks')
        self.resize(300, 150)
        self.view = False
        self.shanks_to_upload: list = list()

    def setup(self) -> None:
        """Set up the dialog layout and widgets."""
        if not self.view:
            self.shank_options = CheckBoxGroup(['All'] + self.controller.all_shanks,
                                               'Select shanks to upload:', orientation='vertical')
            self.shank_options.set_checked([self.controller.model.selected_shank])
            self.shank_options.setup_callback(self.on_shank_button_clicked)

            button_box = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
            )
            button_box.accepted.connect(self.on_accept)
            button_box.rejected.connect(self.reject)
    
            # Assemble layout
            dialog_layout = QtWidgets.QVBoxLayout()
            dialog_layout.addWidget(self.shank_options)
            dialog_layout.addWidget(button_box)
            self.setLayout(dialog_layout)
            self.view = True
        else:
            self.shank_options.set_checked([self.controller.model.selected_shank])

    def on_shank_button_clicked(self, checked: bool, button: str) -> None:
        """
        Update the shank selections based on button clicks.

        If the clicked button is 'All', it checks all shanks when checked,
        or reverts to the currently selected shank when unchecked.

        Parameters
        ----------
        checked: bool
            Whether the button is checked or not.
        button: str
            The text of the button that was clicked.
        """
        if button == 'All' and checked:
            self.shank_options.set_checked(self.controller.all_shanks + ['All'])
        elif button == 'All' and not checked:
            self.shank_options.set_checked([self.controller.model.selected_shank])

    def on_accept(self) -> None:
        """Find the selected shanks and accept the dialog."""
        selected = self.shank_options.get_checked()
        self.shanks_to_upload = [s for s in selected if s != 'All']
        self.accept()
