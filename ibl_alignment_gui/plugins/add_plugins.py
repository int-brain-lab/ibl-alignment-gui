from typing import TYPE_CHECKING

from ibl_alignment_gui.plugins.cluster_features import setup as setup_cluster_features
from ibl_alignment_gui.plugins.features_3d import setup as setup_3d_features
from ibl_alignment_gui.plugins.auto_align import setup as setup_auto_align
from ibl_alignment_gui.plugins.ephys_features import setup as setup_ephys_features
from ibl_alignment_gui.plugins.qc_dialog import setup as setup_qc_dialog
from ibl_alignment_gui.plugins.range_controller import setup as setup_control_range
from ibl_alignment_gui.plugins.upload_dialog import setup as setup_upload_dialog
from ibl_alignment_gui.plugins.channel_prediction import setup as setup_channel_prediction

if TYPE_CHECKING:
    from ibl_alignment_gui.app.app_controller import AlignmentGUIController

# TODO
# Don't have callbacks unless activated


class Plugins:
    """
    Class to manage and initialize plugins for the alignment GUI.

    Parameters
    ----------
    controller : AlignmentGUIController
        The main application controller.
    """

    def __init__(self, controller: 'AlignmentGUIController'):

        controller.plugin_options = controller.view.menu_widgets.addMenu('Plugins')
        controller.plugins = dict()

        setup_qc_dialog(controller)
        setup_upload_dialog(controller)
        setup_cluster_features(controller)
        setup_3d_features(controller)
        setup_auto_align(controller)
        setup_ephys_features(controller)
        setup_control_range(controller)
        setup_channel_prediction(controller)

        # Need to make sure they are only called if they are activated
        # setup_3d_features(controller)
