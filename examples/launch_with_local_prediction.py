"""Launch the alignment GUI with a local region-prediction model pre-wired.

Convenience launcher that opens the IBL alignment GUI on a local (yaml) session and pre-populates
the *Channel Prediction* plugin so that a locally-stored region-prediction model + per-channel
features parquet are used without navigating the file dialogs every time.

This is a thin wrapper around :class:`AlignmentGUIController` that does **not** modify any GUI code.
It simply:

1. Builds the controller in offline mode (optionally loading a YAML session config via ``--yaml``).
2. Pre-sets the *Channel Prediction* plugin's inference-model source (a local release, or the
   Hugging Face Hub release when ``--model-dir`` is omitted) and ``features_path`` so the dialogs
   are skipped and inference runs straight away.
3. Shows the window and enters the Qt event loop.

After the GUI opens (the session auto-loads when ``--yaml`` is given), click
``Plugins -> Channel Prediction -> Inference Model`` (or ``Inference Cumulative``) to overlay the
predicted regions on the reference histology.

Run with::

    python launch_with_local_prediction.py \
        --yaml /path/to/session/alignment_gui_probe00.yaml \
        --model-dir /path/to/ea-decoder-channel-xgboost/<vintage>

``--model-dir`` is a local release of ``int-brain-lab/ea-decoder-channel-xgboost`` (it holds
``ephysatlas_model.json``); omit it to download the Hub release. The automatic-alignment model
(``Load alignment model (Hugging Face)...``) loads from the Hub; to use a local release of it
instead, pre-wire it with ``--encoder-dir``::

    python launch_with_local_prediction.py \
        --yaml /path/to/session/alignment_gui_probe00.yaml \
        --model-dir /path/to/ea-decoder-channel-xgboost/<vintage> \
        --encoder-dir /path/to/ea-encoder-channel/<vintage>

"""

# %% Imports and default paths
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from qtpy import QtWidgets

from ibl_alignment_gui.app.controllers.app_controller import AlignmentGUIController
from ibl_alignment_gui.plugins.channel_prediction import PLUGIN_NAME

# Model locations are machine-specific, so they are passed on the command line rather than
# defaulted here. Features normally live in the session YAML, so --features is only an override
# for YAMLs that do not specify a `features` dataset. The automatic-alignment model comes from the
# Hugging Face Hub unless --encoder-dir points at a local release.

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
        '--model-dir', type=Path, default=None,
        help='Local release directory of the inference model (containing ephysatlas_model.json). '
             'Default: the Hugging Face Hub release.')
    parser.add_argument(
        '--features', type=Path, default=None,
        help='Optional per-channel features parquet override (only needed if the YAML has no '
             '`features` dataset).')
    parser.add_argument(
        '--encoder-dir', type=Path, default=None,
        help='Local release directory of the channel-level alignment model '
             '(int-brain-lab/ea-encoder-channel layout). Default: the Hugging Face Hub release.')
    return parser.parse_args()


# %% Path validation
def validate_paths(model_dir: Path | None, features_path: Path | None) -> None:
    """Sanity-check the model directory and the features override, if given, before launching."""
    if model_dir is not None and not (model_dir / 'ephysatlas_model.json').is_file():
        raise SystemExit(f'Model directory has no ephysatlas_model.json:\n  {model_dir}')
    if features_path is not None and not features_path.is_file():
        raise SystemExit(f'Features file not found:\n  {features_path}')
    logger.info('Model dir:     %s', model_dir or 'Hugging Face Hub release')
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
    from ibl_alignment_gui.plugins.ephys_atlas import inference  # noqa: PLC0415

    model_source = str(args.model_dir) if args.model_dir is not None else inference.MODEL_REPO_ID
    plugin_state[inference.MODEL_NAME] = {
        'source': (model_source, inference.MODEL_REVISION), 'model': None}
    if args.features is not None:
        plugin_state['features_path'] = args.features
    # Pre-wire the automatic-alignment model at a local release too, so it runs offline.
    if args.encoder_dir is not None:
        from ephysatlas.alignment import VINTAGE  # noqa: PLC0415 - heavy, only when pre-wiring

        plugin_state['Encoding'] = {'source': (str(args.encoder_dir), VINTAGE), 'model': None}
    logger.info(
        'Pre-populated %r plugin. Click Plugins -> %s -> Inference Model to run inference.',
        PLUGIN_NAME, PLUGIN_NAME)

    controller.view.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
