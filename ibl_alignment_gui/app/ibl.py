from one.api import ONE
from brainbox.io.one import SpikeSortingLoader
import iblqt.tools

import ibl_alignment_gui.app.app_controller as app_controller


def alignment(pid=None) -> app_controller.AlignmentGUIController:
    app = iblqt.tools.get_or_create_app()
    agc = app_controller.AlignmentGUIController()
    agc.view.show()
    return agc


def pid(pid, one=None):
    one = one if one is not None else ONE()
    ssl = SpikeSortingLoader(one=one, pid=pid)
    agc = alignment()
    session_info = one.eid2ref(ssl.eid)

    # now we pilot the QUI to select the dataset
    cbs = agc.view.selection_widgets.dropdowns['subject']['combobox']
    for i in range(cbs.count()):
        if cbs.itemText(i) == session_info['subject']:
            break
    cbs.setCurrentIndex(i)
    print(cbs.currentIndex(), cbs.currentText())
    agc.on_subject_selected(cbs.currentIndex())

    cbs = agc.view.selection_widgets.dropdowns['session']['combobox']
    for i in range(cbs.count()):
        if cbs.itemText(i) == f'{str(session_info['date'])} {ssl.pname}':
            break
    cbs.setCurrentIndex(i)
    print(cbs.currentIndex(), cbs.currentText())
    agc.on_session_selected(cbs.currentIndex())
    agc.data_button_pressed()
