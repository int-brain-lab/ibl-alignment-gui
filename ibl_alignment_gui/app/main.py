import argparse

from qtpy import QtWidgets

from ibl_alignment_gui.app.app_controller import AlignmentGUIController


def launch_app() -> None:
    """Launch the alignment GUI application in offline mode with optional YAML file."""
    parser = argparse.ArgumentParser()

    parser.add_argument('-y', '--yaml', required=False, type=str, help='Path to the YAML file')

    args = parser.parse_args()

    app = QtWidgets.QApplication([])
    mainapp = AlignmentGUIController(offline=True, csv=None, yaml=args.yaml)
    mainapp.view.show()
    app.exec_()


def launch_app_ibl() -> None:
    """Launch the alignment GUI application in IBL mode.

    Optionally accepts a CSV file or a probe insertion id (pid) to auto-load. When a pid is
    given the subject, session and shank dropdowns are configured to that insertion and its
    data is loaded automatically.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--csv', required=False, type=str, help='Path to the CSV file')
    parser.add_argument(
        '-p', '--pid', required=False, type=str, help='Probe insertion id to auto-load'
    )

    args = parser.parse_args()

    if args.csv is not None and args.pid is not None:
        parser.error('--pid cannot be used together with --csv')

    app = QtWidgets.QApplication([])
    try:
        mainapp = AlignmentGUIController(offline=False, csv=args.csv, yaml=None, pid=args.pid)
    except ValueError as err:
        parser.error(str(err))
    mainapp.view.show()
    app.exec_()


if __name__ == '__main__':
    launch_app()
