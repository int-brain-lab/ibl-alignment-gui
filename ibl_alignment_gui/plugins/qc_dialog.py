from typing import TYPE_CHECKING

from qtpy import QtWidgets

from ibllib.qc.critical_reasons import CriticalInsertionNote

if TYPE_CHECKING:
    from ibl_alignment_gui.app.app_controller import AlignmentGUIController, AlignmentGUIView

PLUGIN_NAME = "QC dialog"


def setup(controller: 'AlignmentGUIController') -> None:
    """
    Set up the QC dialog and connect its accepted signal to the callback.

    Parameters
    ----------
    controller: AlignmentController
        The main application controller.
    """
    controller.qc_dialog = QCDialog(controller.view)
    controller.qc_dialog.accepted.connect(lambda: callback(controller))


def display(controller: 'AlignmentGUIController', shank: str) -> None:
    """
    Show the QC dialog.

    Parameters
    ----------
    controller:
        The main application controller.
    shank: str
        The shank identifier for which the QC dialog is displayed.
    """
    controller.qc_dialog.setWindowTitle(f"QC assessment {shank}")
    controller.qc_dialog.exec_()


def callback(controller: 'AlignmentGUIController'):
    """
    Gather QC inputs and update upload loader with the QC information.

    Parameters
    ----------
    controller: AlignmentController
        The main application controller.
    """
    # Get the uploader for the selected shank and default configuration
    upload = controller.model.get_selected_shank()[
        controller.model.default_config].loaders['upload']
    # Pass in the QC information from the dialog
    upload.set_user_qc(*controller.qc_dialog.get_qc())


class QCDialog(QtWidgets.QDialog):
    """
    Dialog for collecting QC information from the user.

    Parameters
    ----------
    view : AlignmentGUIView
        The main application view.
    """

    def __init__(self, view: 'AlignmentGUIView'):
        super().__init__(view)

        self.setWindowTitle('QC assessment')
        self.resize(300, 150)
        self.setup()

    def setup(self) -> None:
        """Set up the dialog layout and widgets."""
        # Alignment QC
        align_qc_label = QtWidgets.QLabel("Confidence of alignment:")
        self.align_qc = QtWidgets.QComboBox()
        self.align_qc.addItems(["High", "Medium", "Low"])

        # Ephys QC
        ephys_qc_label = QtWidgets.QLabel("QC for ephys recording:")
        self.ephys_qc = QtWidgets.QComboBox()
        self.ephys_qc.addItems(["Pass", "Warning", "Critical"])

        # Ephys QC descriptions
        self.desc_buttons = QtWidgets.QButtonGroup()
        self.desc_buttons.setExclusive(False)

        desc_layout = QtWidgets.QVBoxLayout()
        for i, label in enumerate(CriticalInsertionNote.descriptions_gui):
            checkbox = QtWidgets.QCheckBox(label)
            self.desc_buttons.addButton(checkbox, i)
            desc_layout.addWidget(checkbox)

        desc_group = QtWidgets.QGroupBox("Describe problem with recording:")
        desc_group.setLayout(desc_layout)

        # Force upload option
        resolve_label = QtWidgets.QLabel(
            "Do you want to resolve this alignment with the current alignment?")
        self.resolve = QtWidgets.QComboBox()
        self.resolve.addItem("No", False)
        self.resolve.addItem("Yes", True)

        # Dialog buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.on_accept)
        button_box.rejected.connect(self.reject)

        # Assemble layout
        dialog_layout = QtWidgets.QVBoxLayout()
        dialog_layout.addWidget(align_qc_label)
        dialog_layout.addWidget(self.align_qc)
        dialog_layout.addWidget(ephys_qc_label)
        dialog_layout.addWidget(self.ephys_qc)
        dialog_layout.addWidget(desc_group)
        dialog_layout.addWidget(resolve_label)
        dialog_layout.addWidget(self.resolve)
        dialog_layout.addWidget(button_box)
        self.setLayout(dialog_layout)

    def on_accept(self) -> None:
        """Validate the input before accepting."""
        ephys_qc = self.ephys_qc.currentText()
        ephys_desc = [btn.text() for btn in self.desc_buttons.buttons() if btn.isChecked()]

        if ephys_qc != 'Pass' and not ephys_desc:
            QtWidgets.QMessageBox.warning(self, "Missing Information",
                                          "You must select a reason for QC choice")
            return
        self.accept()

    def get_qc(self) -> tuple[str, str, list[str], bool]:
        """
        Retrieve the QC information from the dialog.

        Returns
        -------
        tuple[str, str, list[str], bool]
            A tuple containing alignment QC, ephys QC, ephys descriptions, and force resolve flag
        """
        align_qc = self.align_qc.currentText()
        ephys_qc = self.ephys_qc.currentText()
        ephys_desc = [btn.text() for btn in self.desc_buttons.buttons() if btn.isChecked()]
        force_resolve = self.resolve.currentData()

        return align_qc, ephys_qc, ephys_desc, force_resolve
