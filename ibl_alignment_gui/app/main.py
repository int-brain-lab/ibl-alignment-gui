import argparse

from qtpy import QtWidgets

from ibl_alignment_gui.app.app_controller import AlignmentGUIController


def launch_app():
    parser = argparse.ArgumentParser()

    parser.add_argument('-y', '--yaml', required=False, type=str, help='Path to the YAML file')

    args = parser.parse_args()

    app = QtWidgets.QApplication([])
    mainapp = AlignmentGUIController(offline=True, csv=None, yaml=args.yaml)
    mainapp.view.show()
    app.exec_()


def launch_app_ibl():
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--csv', required=False, type=str, help='Path to the CSV file')

    args = parser.parse_args()

    app = QtWidgets.QApplication([])
    mainapp = AlignmentGUIController(offline=False, csv=args.csv, yaml=None)
    mainapp.view.show()
    app.exec_()


if __name__ == '__main__':
    launch_app()
