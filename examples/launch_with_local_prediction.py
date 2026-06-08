"""Launch the alignment GUI with a local region-prediction model pre-wired.

Convenience launcher that opens the IBL alignment GUI on a local (yaml) session and pre-populates
the *Channel Prediction* plugin so that a locally-stored region-prediction model + per-channel
features parquet are used without navigating the file dialogs every time.

This is a thin wrapper around :class:`AlignmentGUIController` that does **not** modify any GUI code.
It simply:

1. Builds the controller in offline mode (optionally loading a YAML session config via ``--yaml``).
2. Sets the *Channel Prediction* plugin's ``local_model_dir`` / ``features_path`` so the dialogs
   are skipped and inference runs straight away.
3. Shows the window and enters the Qt event loop.

After the GUI opens (the session auto-loads when ``--yaml`` is given), click
``Plugins -> Channel Prediction -> Inference Model`` (or ``Inference Cumulative``) to overlay the
predicted regions on the reference histology.

Run with::
# Cazettes Lab datasets
python launch_with_local_prediction.py --yaml /mnt/s0/Data/2026_cazettes/2026_cazettes/Data/VF066/2025_12_04/alignment_gui_probe01.yaml


#Using OOP implementation
python launch_with_local_prediction.py --yaml /mnt/s0/Data/2026_cazettes/2026_cazettes/Data/VF066/2025_12_04/alignment_gui_probe01.yaml
python launch_with_local_prediction.py --yaml /mnt/s0/Data/2026_cazettes/2026_cazettes/Data/VF066/2025_12_04/alignment_gui_probe00.yaml
python launch_with_local_prediction.py --yaml /mnt/s0/Data/2026_cazettes/2026_cazettes/Data/VF065/2025_12_17/alignment_gui.yaml

# NuoLi's datasets
python launch_with_local_prediction.py --yaml /mnt/s0/Data/2026_nuo_li/Munni/20210527_g0_imec0/alignment_gui_DL021.yaml
python launch_with_local_prediction.py --yaml /mnt/s0/Data/2026_nuo_li/Munni/20210620_g0_imec0/alignment_gui_DL025.yaml

"""

# %% Imports and default paths
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from qtpy import QtWidgets

from ibl_alignment_gui.app.app_controller import AlignmentGUIController
from ibl_alignment_gui.plugins.channel_prediction import PLUGIN_NAME

# Default model dir (session-independent). Features now live in the session YAML, so --features is
# only an optional override for YAMLs that do not specify a `features` dataset.
DEFAULT_MODEL_DIR = Path(
    '/home/pranavrai/Work/int-brain-lab/projects/cazettes_sample_data_check'
    '/analysis/features/ea_active/models/2026_W12_Cosmos_careless-clover-dingo'
)

# Spatial Encoder (automatic alignment): local encoder model dir + reference-bank root. The bank
# root is the dir that contains <project>/<vintage>/agg_full/*.pqt (here ea_active/2026_W12).
DEFAULT_ENCODER_DIR = Path(
    '/home/pranavrai/Work/int-brain-lab/projects/allen_sample_data_check'
    '/analysis/temp_model/encoding_models/2026_W12'
)
DEFAULT_ENCODER_DATA = Path(
    '/home/pranavrai/Work/int-brain-lab/projects/cazettes_sample_data_check/analysis/features'
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger(__name__)


# %% CLI parsing
def parse_args() -> argparse.Namespace:
    """Parse CLI overrides for the session yaml, model dir, and features path."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--yaml', '-y', type=Path, default=None,
        help='Path to a session YAML config (same format as `alignment-gui -y`).')
    parser.add_argument(
        '--model-dir', type=Path, default=DEFAULT_MODEL_DIR,
        help=f'Trained model directory (containing folds/FOLD00/). Default: {DEFAULT_MODEL_DIR}')
    parser.add_argument(
        '--features', type=Path, default=None,
        help='Optional per-channel features parquet override (only needed if the YAML has no '
             '`features` dataset).')
    parser.add_argument(
        '--encoder-dir', type=Path, default=DEFAULT_ENCODER_DIR,
        help=f'Local Spatial Encoder model dir (SE_model_*.pt + *_vol_pca.npy). '
             f'Default: {DEFAULT_ENCODER_DIR}')
    parser.add_argument(
        '--encoder-data', type=Path, default=DEFAULT_ENCODER_DATA,
        help=f'Spatial Encoder reference-bank root (<project>/<vintage>/agg_full/). '
             f'Default: {DEFAULT_ENCODER_DATA}')
    return parser.parse_args()


# %% Path validation
def validate_paths(model_dir: Path, features_path: Path | None) -> None:
    """Sanity-check the model directory (and the features override, if given) before launching."""
    has_folds = (model_dir / 'folds' / 'FOLD00').is_dir() or (model_dir / 'FOLD00').is_dir()
    if not has_folds:
        raise SystemExit(
            f'Model directory has no "folds/FOLD00" (or "FOLD00") subdirectory:\n  {model_dir}')
    if features_path is not None and not features_path.is_file():
        raise SystemExit(f'Features file not found:\n  {features_path}')
    logger.info('Model dir:     %s', model_dir)
    if features_path is not None:
        logger.info('Features file: %s (override)', features_path)


# %% Main entry point
def main() -> None:
    """Build the controller, pre-fill the plugin paths, and run the Qt event loop."""
    args = parse_args()
    validate_paths(args.model_dir, args.features)

    # Qt application — must be created before any widgets.
    app = QtWidgets.QApplication([])

    # Offline mode + optional yaml session config (csv must be None offline).
    controller = AlignmentGUIController(
        offline=True,
        csv=None,
        yaml=str(args.yaml) if args.yaml is not None else None,
    )

    # The Channel Prediction plugin registers during the controller's __init__, seeding its state
    # keys with None. Pre-set the model dir (and a features override if given) so the file dialogs
    # are skipped on the first click. Features normally come from the session YAML.
    plugin_state = controller.plugins[PLUGIN_NAME]
    plugin_state['local_model_dir'] = args.model_dir
    if args.features is not None:
        plugin_state['features_path'] = args.features
    # Pre-wire the Spatial Encoder (automatic alignment) at local paths too, so it runs offline.
    plugin_state['local_encoder_dir'] = args.encoder_dir
    plugin_state['local_encoder_data'] = args.encoder_data
    logger.info(
        'Pre-populated %r plugin. Click Plugins -> %s -> Inference Model to run inference.',
        PLUGIN_NAME, PLUGIN_NAME)

    controller.view.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
