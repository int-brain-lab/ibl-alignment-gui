from types import MethodType
from typing import TYPE_CHECKING, Any

from ibl_alignment_gui.loaders.plot_loader import skip_missing
from ibl_alignment_gui.utils.helpers import shank_loop
from iblutil.util import Bunch

if TYPE_CHECKING:
    from ibl_alignment_gui.app.controllers.app_controller import AlignmentGUIController
    from ibl_alignment_gui.app.controllers.shank_controller import ShankController


PLUGIN_NAME = 'Additional Plots Allen'


def setup(controller: 'AlignmentGUIController') -> None:
    """
    Example to show how to add additional plots to the GUI.

    Parameters
    ----------
    controller: AlignmentGUIController
        The main application controller.
    """
    controller.plugins[PLUGIN_NAME] = Bunch()
    controller.plugins[PLUGIN_NAME]['activated'] = True
    # Attach callbacks to methods in the controller
    controller.plugins[PLUGIN_NAME]['load_data'] = add_plots


@shank_loop
def add_plots(_, items: 'ShankController', **kwargs) -> None:
    """
    Add additional plots to the plot loader.

    Parameters
    ----------
    _
    items: ShankController
        A ShankController instance.
    -------

    """
    # Add the additional data that may be required for the plots
    items.model.raw_data['rms_AP_main'] = items.model.loaders['data'].get_rms_data(
        'ephysTimeRmsAPMain'
    )
    items.model.raw_data['rms_LF_main'] = items.model.loaders['data'].get_rms_data(
        'ephysTimeRmsLFMain'
    )

    items.model.loaders['plots'].image_rms_ap_main = MethodType(
        image_rms_ap_main, items.model.loaders['plots']
    )

    items.model.loaders['plots'].image_rms_lf_main = MethodType(
        image_rms_lf_main, items.model.loaders['plots']
    )

    items.model.loaders['plots'].probe_rms_ap_main = MethodType(
        probe_rms_ap_main, items.model.loaders['plots']
    )

    items.model.loaders['plots'].probe_rms_lf_main = MethodType(
        probe_rms_lf_main, items.model.loaders['plots']
    )


@skip_missing(['rms_AP_main'])
def image_rms_ap_main(self) -> dict[str, Any]:
    return self._image_rms('rms_AP_main', plot_key='rms AP Main')


@skip_missing(['rms_LF_main'])
def image_rms_lf_main(self) -> dict[str, Any]:
    return self._image_rms('rms_LF_main', plot_key='rms LF Main')


@skip_missing(['rms_AP_main'])
def probe_rms_ap_main(self) -> dict[str, Any]:
    return self._probe_rms('rms_AP_main', plot_key='rms AP Main')


@skip_missing(['rms_LF_main'])
def probe_rms_lf_main(self) -> dict[str, Any]:
    return self._probe_rms('rms_LF_main', plot_key='rms LF Main')
