from typing import TYPE_CHECKING

from ibl_alignment_gui.plugins.allen.additional_plots_allen import setup as setup_allen_plots

# from ibl_alignment_gui.plugins.additional_plots import setup as setup_additional_plots
from ibl_alignment_gui.plugins.allen.custom_filters import setup as setup_custom_filters
from ibl_alignment_gui.plugins.channel_prediction import setup as setup_channel_prediction
from ibl_alignment_gui.plugins.cluster_features import setup as setup_cluster_features
from ibl_alignment_gui.plugins.ephys_features import setup as setup_ephys_features
from ibl_alignment_gui.plugins.features_3d import setup as setup_3d_features
from ibl_alignment_gui.plugins.qc_dialog import setup as setup_qc_dialog
from ibl_alignment_gui.plugins.range_controller import setup as setup_control_range
from ibl_alignment_gui.plugins.upload_dialog import setup as setup_upload_dialog

if TYPE_CHECKING:
    from ibl_alignment_gui.app.controllers.app_controller import AlignmentGUIController


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
        if not controller.offline:
            setup_ephys_features(controller)
        # Channel prediction is offline-capable: local model + features inference works without
        # ONE. Its S3/torch-only options are gated inside setup_channel_prediction.
        setup_channel_prediction(controller)
        setup_control_range(controller)
        setup_3d_features(controller)
        # setup_additional_plots(controller)
        if controller.allen:
            setup_custom_filters(controller)
            setup_allen_plots(controller)
