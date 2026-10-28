"""Launch the alignment GUI with a local region-prediction model pre-wired.

Convenience launcher that opens the IBL alignment GUI on a local (yaml) session and pre-populates
the *Channel Prediction* plugin so that locally-stored region-prediction models are used without
navigating the model dialogs every time. The per-channel features come from the session, as the
``features`` dataset of the YAML.

This is a thin wrapper around :class:`AlignmentGUIController` that does **not** modify any GUI code.
It simply:

1. Builds the controller in offline mode (optionally loading a YAML session config via ``--yaml``).
2. Pre-sets the *Channel Prediction* plugin's model sources (the local releases given with
   ``--model-dir``, otherwise each model's default release on the Hugging Face Hub) so the
   dialogs are skipped and the models run straight away.
3. Shows the window and enters the Qt event loop.

After the GUI opens (the session auto-loads when ``--yaml`` is given), click
``Prediction Plots -> XGBoost region prediction`` (or ``ProbeTransformer region prediction``,
``Auto-alignment``, or their other entries) to show the predicted regions alongside the ephys
plots, or ``Prediction Plots -> All predictions (side by side)`` to compare them.

Run with::

    python launch_with_local_prediction.py \
        --yaml /path/to/session/alignment_gui_probe00.yaml \
        --model-dir /path/to/ea-decoder-channel-xgboost/<vintage> \
        --model-dir /path/to/ea-decoder-channel-transformer/<vintage> \
        --model-dir /path/to/ea-encoder-channel/<vintage>

``--model-dir`` is a local release directory (it holds ``ephysatlas_model.json``) and can be given
once per model: the release's manifest ``model_class`` tells which model it is -- the XGBoost or
ProbeTransformer region decoder, or the automatic-alignment model (``ea-encoder-channel``). A
model without a ``--model-dir`` is downloaded from the Hugging Face Hub, at its default release
(for the automatic-alignment model, the one ibleatools names).

"""

# %% Imports and default paths
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from qtpy import QtWidgets

from ibl_alignment_gui.app.controllers.app_controller import AlignmentGUIController
from ibl_alignment_gui.plugins.channel_prediction import MODELS, PLUGIN_NAME, available_models

# Model locations are machine-specific, so they are passed on the command line rather than
# defaulted here. The features come from the session YAML's `features` dataset.

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger(__name__)


# %% CLI parsing
def parse_args() -> argparse.Namespace:
    """Parse CLI overrides for the session yaml and model dirs."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--yaml', '-y', type=Path, default=None,
        help='Path to a session YAML config (same format as `alignment-gui -y`).')
    parser.add_argument(
        '--model-dir', type=Path, action='append', default=[],
        help='Local release directory of a model (containing ephysatlas_model.json); repeat it '
             'for several models. Which model it is comes from the manifest model_class. '
             'Default: the Hugging Face Hub releases of the region decoders.')
    return parser.parse_args()


# %% Path validation
def model_sources(model_dirs: list[Path]) -> dict[str, str]:
    """Match each local release directory to the model it holds, from its manifest ``model_class``.

    Parameters
    ----------
    model_dirs : list of Path
        Local release directories, each holding an ``ephysatlas_model.json`` manifest.

    Returns
    -------
    dict of str to str
        The plugin-state key of each model (a region-decoder name, or ``'Encoding'`` for the
        automatic-alignment model) -> its release directory.

    Raises
    ------
    SystemExit
        If a directory has no manifest, declares an unknown ``model_class``, or two directories
        hold the same model.
    """
    state_keys = {spec.model_class: spec.key for spec in MODELS}
    sources = {}
    for model_dir in model_dirs:
        manifest = model_dir / 'ephysatlas_model.json'
        if not manifest.is_file():
            raise SystemExit(f'Model directory has no ephysatlas_model.json:\n  {model_dir}')
        model_class = json.loads(manifest.read_text()).get('model_class')
        if model_class not in state_keys:
            raise SystemExit(
                f'{model_dir} holds a {model_class!r} model; expected one of {sorted(state_keys)}'
            )
        key = state_keys[model_class]
        if key in sources:
            raise SystemExit(f'Two model directories for {key}:\n  {sources[key]}\n  {model_dir}')
        sources[key] = str(model_dir)
        logger.info('Model dir:     %s -> %s', model_dir, key)
    return sources


# %% Main entry point
def main() -> None:
    """Build the controller, pre-fill the plugin paths, and run the Qt event loop."""
    args = parse_args()
    sources = model_sources(args.model_dir)

    # Qt application — must be created before any widgets.
    app = QtWidgets.QApplication([])

    # Offline mode + optional yaml session config (csv must be None offline).
    controller = AlignmentGUIController(
        offline=True,
        csv=None,
        yaml=str(args.yaml) if args.yaml is not None else None,
    )

    # The Channel Prediction plugin registers during the controller's __init__, seeding its state
    # keys with None. Pre-set the model sources so the model dialogs are skipped on the first
    # click. The features come from the session YAML.
    plugin_state = controller.plugins[PLUGIN_NAME]

    # Every model offered (those needing torch only with torch): its local release when given,
    # else its default release on the Hugging Face Hub. The model loads on its first option.
    for spec in available_models():
        repo, revision = spec.default_source()
        plugin_state[spec.key] = {'source': (sources.get(spec.key, repo), revision), 'model': None}
        logger.info('Model:         %s -> %s@%s', spec.key, sources.get(spec.key, repo), revision)
    logger.info(
        'Pre-populated %r plugin. Pick a model option in Prediction Plots to run it.', PLUGIN_NAME)

    controller.view.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
