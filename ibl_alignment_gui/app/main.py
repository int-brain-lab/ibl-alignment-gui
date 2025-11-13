from qtpy import QtWidgets

from ibl_alignment_gui.app.app_controller import AlignmentGUIController

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Offline vs online mode')
    parser.add_argument(
        '-o', '--offline',
        required=False,
        default=False,
        help='Run in offline mode'
    )

    parser.add_argument(
        '-c', '--csv',
        required=False,
        type=str,
        help='Path to the CSV file'
    )

    args = parser.parse_args()

    app = QtWidgets.QApplication([])
    mainapp = AlignmentGUIController(offline=args.offline, csv=args.csv)
    mainapp.view.show()
    app.exec_()


# TODO:
# To run use: python .\ibl_alignment_gui\app\main.py