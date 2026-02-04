from collections import defaultdict
from pathlib import Path

import yaml
from pydantic import BaseModel

# -------------------------------
# Pydantic Models
# -------------------------------


class DatasetPaths(BaseModel):
    """
    Container for resolved dataset paths for a single probe.

    Attributes
    ----------
    spike_sorting : Path | None
        Path to spike sorting output directory
    processed_ephys : Path | None
        Path to processed electrophysiology data directory
    raw_ephys : Path | None
        Path to raw electrophysiology recordings directory
    task : Path | None
        Path to task data directory
    raw_task : Path | None
        Path to raw task data directory
    picks : Path | None
        Path to probe trajectory pick files directory
    histology : Path | None
        Path to histology volume directory
    output : Path | None
        Path to alignment output directory
    """

    spike_sorting: Path | None = None
    processed_ephys: Path | None = None
    raw_ephys: Path | None = None
    task: Path | None = None
    raw_task: Path | None = None
    picks: Path | None = None
    histology: Path | None = None
    output: Path | None = None


class Datasets(BaseModel):
    """
    Dataset configuration with optional path and backend specification.

    Attributes
    ----------
    path : Path
        Relative or absolute path to the dataset directory
    backend : str | None
        Data format backend (e.g., 'phylib', 'spikeglx')
    """

    path: Path | None = None
    backend: str | None = None


class Probe(BaseModel):
    """
    Configuration for a single probe.

    Attributes
    ----------
    datasets : dict[str, Datasets] | None
        Dictionary mapping dataset names to their configurations
    path : Path | None
        Probe-level base path for resolving relative dataset paths
    """

    datasets: dict[str, Datasets] | None = None
    path: Path | None = None  # Probe-level root


class Configuration(BaseModel):
    """
    Configuration for a single experimental configuration.

    Attributes
    ----------
    probes : dict[str, Probe]
        Dictionary mapping probe names to their configurations
    path : Path | None
        Configuration-level base path for resolving relative probe paths
    """

    probes: dict[str, Probe]
    path: Path | None = None  # Config-level root


class AlignmentYAML(BaseModel):
    """
    Root-level YAML configuration structure.

    Attributes
    ----------
    defaults : dict[str, Datasets]
        Default dataset configurations applied to all probes
    configurations : dict[str, Configuration]
        Dictionary mapping configuration names to their configurations
    path : Path | None
        Global root path for resolving all relative paths
    """

    defaults: dict[str, Datasets] | None = None
    configurations: dict[str, Configuration]
    path: Path | None = None  # Global root


# -------------------------------
# Path resolution logic
# -------------------------------


def resolve_path(
    dataset_path: Path | None = None,
    probe_path: Path | None = None,
    config_path: Path | None = None,
    global_path: Path | None = None,
    default_path: Path | None = None,
) -> Path | None:
    """
    Resolve dataset path using hierarchical path resolution.

    If path is absolute at any stage, it is returned immediately. Otherwise path
    is resolved progressively through the provided paths.

    Resolution order:
        dataset
        probe / dataset
        config / probe/ dataset
        global / config / probe / dataset

    If that path is still relative at the end, an error is raised.
    """
    # Pick value or default
    path = dataset_path if dataset_path is not None else default_path
    if path is None:
        return None

    resolved_path = Path(path).expanduser()

    # Absolute value wins immediately
    if resolved_path.is_absolute():
        return resolved_path.resolve()

    # Helper to prepend a root if present
    def prepend(root: Path | None, p: Path) -> Path:
        return root / p if root is not None else p

    # Progressive buildup
    resolved_path = prepend(probe_path, resolved_path)
    if resolved_path.is_absolute():
        return resolved_path.resolve()

    resolved_path = prepend(config_path, resolved_path)
    if resolved_path.is_absolute():
        return resolved_path.resolve()

    resolved_path = prepend(global_path, resolved_path)
    if resolved_path.is_absolute():
        return resolved_path.resolve()

    # Still relative → cannot resolve fully
    raise ValueError('No absolute root provided to resolve relative path.')


# -------------------------------
# Loader
# -------------------------------


def load_alignment_yaml(
    yaml_file: str,
) -> tuple[list[str], list[str], dict[str, dict[str, DatasetPaths]]]:
    """
    Load and parse alignment configuration YAML file.

    Resolves all dataset paths using hierarchical path resolution and applies
    defaults. Creates output directories if they don't exist.

    Parameters
    ----------
    yaml_file : str
        Path to the YAML configuration file

    Returns
    -------
    configs : list of str
        List of configuration names
    probes : list of str
        List of unique probe names across all configurations
    data_paths : A dict of dicts of DatasetPaths
        Nested dictionary of resolved paths:
        data_paths[config_name][probe_name] -> DatasetPaths

    Notes
    -----
    - If no 'configurations' section exists, creates a 'default' configuration
    - Falls back to raw_ephys path if processed_ephys is not specified
    - Falls back to spike_sorting path if output path is not specified
    - Creates output directories automatically with parents
    """
    yaml_file = Path(yaml_file)

    if not yaml_file.exists():
        raise FileNotFoundError(f'YAML file {yaml_file} does not exist')

    with open(yaml_file) as f:
        data = yaml.safe_load(f)

    # Support files without explicit 'configurations' section
    if 'configurations' not in data:
        probes = data.pop('probes', {})
        data['configurations'] = {'default': {'probes': probes}}

    alignment = AlignmentYAML(**data)
    global_path = alignment.path

    data_paths = defaultdict(dict)
    configs = []
    probes = []

    for cname, config in alignment.configurations.items():
        config_path = config.path
        configs.append(cname)

        for pname, probe in config.probes.items():
            probe_path = probe.path
            probes.append(pname)
            datasets = probe.datasets

            resolved_paths = DatasetPaths()

            def get_path(dname: str) -> str | None:
                """Get path for a specific dataset from probe configuration."""
                dataset = datasets.get(dname)
                return dataset.path if dataset else None

            def get_default_path(dname: str) -> str | None:
                """Get default path for a specific dataset from defaults section."""
                if alignment.defaults:
                    default_dataset = alignment.defaults.get(dname)
                    return default_dataset.path if default_dataset else None
                return None

            # Resolve all paths
            for dataset_name in [
                'spike_sorting',
                'processed_ephys',
                'raw_ephys',
                'picks',
                'histology',
                'output',
            ]:
                path_value = get_path(dataset_name)
                default_value = get_default_path(dataset_name)
                resolved_path = resolve_path(
                    path_value, probe_path, config_path, global_path, default_value
                )
                setattr(resolved_paths, dataset_name, resolved_path)

            if resolved_paths.processed_ephys is None:
                resolved_paths.processed_ephys = resolved_paths.raw_ephys

            if resolved_paths.output is None:
                resolved_paths.output = resolved_paths.spike_sorting

            # resolved_paths.output.mkdir(parents=True, exist_ok=True)

            data_paths[cname][pname] = resolved_paths

    assert len(configs) <= 2, (
        'More than two configurations found in YAML, alignment GUI supports up to two.'
    )

    return configs, list(set(probes)), data_paths
