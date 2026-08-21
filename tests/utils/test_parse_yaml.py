import sys
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from ibl_alignment_gui.utils.parse_yaml import load_alignment_yaml, resolve_path

FIXTURE_PATH = Path(__file__).parents[1].joinpath('fixtures', 'yaml')

# The documented examples use POSIX roots such as /mnt/data, which are not absolute on Windows,
# so resolve_path cannot build an absolute path from them there.
skip_on_windows = unittest.skipIf(
    sys.platform == 'win32', 'the documented examples use POSIX absolute paths'
)


class TestResolvePath(unittest.TestCase):
    """Test the resolve_path function."""

    def setUp(self):
        # Built from the temp directory so that the roots are absolute on every platform
        self.root = Path(tempfile.gettempdir()).resolve()

    def test_absolute_dataset_path_wins(self):
        absolute = self.root.joinpath('custom', 'kilosort')
        resolved = resolve_path(
            absolute,
            probe_path=Path('probe_00'),
            config_path=Path('session'),
            global_path=self.root.joinpath('other'),
        )
        self.assertEqual(resolved, absolute)

    def test_progressive_resolution(self):
        with self.subTest('probe root'):
            resolved = resolve_path(Path('kilosort'), probe_path=self.root.joinpath('probe_00'))
            self.assertEqual(resolved, self.root.joinpath('probe_00', 'kilosort'))

        with self.subTest('config root'):
            resolved = resolve_path(
                Path('kilosort'),
                probe_path=Path('probe_00'),
                config_path=self.root.joinpath('session'),
            )
            self.assertEqual(resolved, self.root.joinpath('session', 'probe_00', 'kilosort'))

        with self.subTest('global root'):
            resolved = resolve_path(
                Path('kilosort'),
                probe_path=Path('probe_00'),
                config_path=Path('session'),
                global_path=self.root,
            )
            self.assertEqual(resolved, self.root.joinpath('session', 'probe_00', 'kilosort'))

    def test_default_is_used_when_no_dataset_path(self):
        resolved = resolve_path(None, global_path=self.root, default_path=Path('pykilosort'))
        self.assertEqual(resolved, self.root.joinpath('pykilosort'))

    def test_dataset_path_takes_precedence_over_default(self):
        resolved = resolve_path(
            Path('kilosort'), global_path=self.root, default_path=Path('pykilosort')
        )
        self.assertEqual(resolved, self.root.joinpath('kilosort'))

    def test_returns_none_when_nothing_given(self):
        self.assertIsNone(resolve_path(None, global_path=self.root))

    def test_relative_with_no_absolute_root_raises(self):
        with self.assertRaises(ValueError):
            resolve_path(Path('kilosort'), probe_path=Path('probe_00'))

    def test_user_directory_is_expanded(self):
        resolved = resolve_path(Path('~/kilosort'))
        self.assertEqual(resolved, Path.home().joinpath('kilosort').resolve())

    def test_parent_references_are_normalised(self):
        resolved = resolve_path(self.root.joinpath('a', '..', 'b'))
        self.assertEqual(resolved, self.root.joinpath('b'))


@skip_on_windows
class TestExamples(unittest.TestCase):
    """Test the YAML examples given in the documentation."""

    def load(self, name):
        return load_alignment_yaml(FIXTURE_PATH.joinpath(f'{name}.yaml'))

    def test_single_probe(self):
        configs, probes, data_paths, space = self.load('single_probe')

        self.assertEqual(configs, ['default'])
        self.assertEqual(probes, ['probe_00'])
        self.assertEqual(space, 'ccf')
        paths = data_paths['default']['probe_00']
        self.assertEqual(paths.spike_sorting, Path('/path/to/probe_00/spike_sorting'))
        self.assertEqual(paths.histology, Path('/path/to/histology'))
        self.assertEqual(paths.output, Path('/path/to/probe_00/alignment_outputs'))

    def test_multi_probe(self):
        configs, probes, data_paths, _ = self.load('multi_probe')

        self.assertEqual(configs, ['default'])
        # The order reaches the shank tabs of the GUI, so it follows the yaml
        self.assertEqual(probes, ['shank_0', 'shank_1', 'shank_2', 'shank_3'])
        for shank in probes:
            with self.subTest(shank):
                paths = data_paths['default'][shank]
                root = Path('/path/to/session_data').joinpath(shank)
                self.assertEqual(paths.spike_sorting, root.joinpath('spike_sorting'))
                self.assertEqual(paths.processed_ephys, root.joinpath('processed_ephys'))
                self.assertEqual(paths.raw_ephys, root.joinpath('raw_ephys'))
                self.assertEqual(paths.picks, root.joinpath('picks'))
                self.assertEqual(paths.output, root.joinpath('output'))

    def test_dual_config(self):
        configs, probes, data_paths, _ = self.load('dual_config')

        self.assertEqual(configs, ['dense', 'sparse'])
        # The same probe appears in both configurations and is only listed once
        self.assertEqual(probes, ['probe_00'])
        self.assertEqual(
            data_paths['dense']['probe_00'].spike_sorting,
            Path('/path/to/dense_session/alf/probe00/kilosort'),
        )
        self.assertEqual(
            data_paths['sparse']['probe_00'].spike_sorting,
            Path('/path/to/sparse_session/alf/probe00/kilosort'),
        )
        # The histology comes from the defaults section and is shared
        self.assertEqual(
            data_paths['dense']['probe_00'].histology,
            data_paths['sparse']['probe_00'].histology,
        )

    def test_dual_config_multi_probe(self):
        configs, probes, data_paths, _ = self.load('dual_config_multi_probe')

        self.assertEqual(configs, ['dense', 'sparse'])
        self.assertEqual(probes, ['shank_0', 'shank_1'])
        for config in configs:
            for probe in probes:
                with self.subTest(f'{config}/{probe}'):
                    self.assertIn(probe, data_paths[config])

    def test_base_path_precedence(self):
        # Each of these is documented with the path it resolves to
        expected = {
            'base_path_global': '/mnt/data/probe_00/kilosort',
            'base_path_config': '/mnt/data/dense_session/kilosort',
            'base_path_probe': '/mnt/data/probe_00/kilosort',
            'base_path_dataset': '/custom/location/kilosort',
            'base_path_defaults_empty': '/mnt/data/probe_00/pykilosort',
        }
        for name, resolved in expected.items():
            with self.subTest(name):
                configs, probes, data_paths, _ = self.load(name)
                paths = data_paths[configs[0]][probes[0]]
                self.assertEqual(paths.spike_sorting, Path(resolved))

    def test_complete_hierarchy(self):
        expected = {
            ('dense', 'spike_sorting'): '/mnt/s0/Data/dense_session/probe_00/pykilosort',
            ('dense', 'processed_ephys'): '/mnt/s0/Data/dense_session/probe_00/pykilosort',
            ('dense', 'raw_ephys'): '/mnt/s0/Data/dense_session/probe_00/spikeglx',
            ('dense', 'picks'): '/custom/path/to/picks',
            ('dense', 'histology'): '/common/histology/subject_001',
            ('dense', 'output'): '/mnt/s0/Data/dense_session/probe_00/alignment_outputs',
            ('sparse', 'spike_sorting'): '/mnt/s0/Data/sparse_session/probe_00/kilosort',
            ('sparse', 'processed_ephys'): '/mnt/s0/Data/sparse_session/probe_00/raw_ephys_data',
            ('sparse', 'raw_ephys'): '/mnt/s0/Data/sparse_session/probe_00/spikeglx',
            ('sparse', 'picks'): '/mnt/s0/Data/sparse_session/probe_00/kilosort',
            ('sparse', 'histology'): '/common/histology/subject_001',
            ('sparse', 'output'): '/mnt/s0/Data/sparse_session/probe_00/kilosort',
        }
        configs, probes, data_paths, _ = self.load('complete_hierarchy')
        self.assertEqual(configs, ['dense', 'sparse'])

        for (config, dataset), resolved in expected.items():
            with self.subTest(f'{config}/{dataset}'):
                paths = data_paths[config]['probe_00']
                self.assertEqual(getattr(paths, dataset), Path(resolved))


class TestPathFallbacks(unittest.TestCase):
    """Test the fallbacks applied once the dataset paths have been resolved."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name).resolve()

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_yaml(self, datasets):
        """Write a single probe yaml with the given dataset entries."""
        lines = [f'path: {self.root}', 'probes:', '  probe_00:', '    datasets:']
        for name, value in datasets.items():
            lines += [f'      {name}:', f'        path: {value}']
        path = self.root.joinpath('session.yaml')
        path.write_text('\n'.join(lines) + '\n')

        return path

    def load(self, datasets):
        _, _, data_paths, _ = load_alignment_yaml(self.write_yaml(datasets))
        return data_paths['default']['probe_00']

    def test_ephys_fallbacks(self):
        with self.subTest('both given'):
            paths = self.load({'spike_sorting': 'ks', 'processed_ephys': 'proc',
                               'raw_ephys': 'raw'})
            self.assertEqual(paths.processed_ephys, self.root.joinpath('proc'))
            self.assertEqual(paths.raw_ephys, self.root.joinpath('raw'))

        with self.subTest('processed given, raw absent'):
            # The explicit processed_ephys is kept, and raw_ephys falls back to the spike sorting
            paths = self.load({'spike_sorting': 'ks', 'processed_ephys': 'proc'})
            self.assertEqual(paths.processed_ephys, self.root.joinpath('proc'))
            self.assertEqual(paths.raw_ephys, self.root.joinpath('ks'))

        with self.subTest('processed absent, raw given'):
            paths = self.load({'spike_sorting': 'ks', 'raw_ephys': 'raw'})
            self.assertEqual(paths.processed_ephys, self.root.joinpath('raw'))
            self.assertEqual(paths.raw_ephys, self.root.joinpath('raw'))

        with self.subTest('both absent'):
            paths = self.load({'spike_sorting': 'ks'})
            self.assertEqual(paths.processed_ephys, self.root.joinpath('ks'))
            self.assertEqual(paths.raw_ephys, self.root.joinpath('ks'))

    def test_output_fallbacks(self):
        with self.subTest('output given'):
            paths = self.load({'spike_sorting': 'ks', 'output': 'out'})
            self.assertEqual(paths.output, self.root.joinpath('out'))

        with self.subTest('falls back to the spike sorting'):
            paths = self.load({'spike_sorting': 'ks', 'picks': 'picks'})
            self.assertEqual(paths.output, self.root.joinpath('ks'))

        with self.subTest('falls back to the picks when there is no spike sorting'):
            paths = self.load({'picks': 'picks'})
            self.assertEqual(paths.output, self.root.joinpath('picks'))


class TestDefaults(unittest.TestCase):
    """Test how the defaults section supplies dataset paths."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name).resolve()

    def tearDown(self):
        self.temp_dir.cleanup()

    def load(self, defaults='', datasets='', probe_path=None):
        """Write and load a single probe yaml with the given defaults and dataset entries."""
        lines = [f'path: {self.root}']
        if defaults:
            lines += ['defaults:', *[f'  {line}' for line in defaults.splitlines()]]
        lines += ['probes:', '  probe_00:']
        if probe_path is not None:
            lines += [f'    path: {probe_path}']
        lines += ['    datasets:', *[f'      {line}' for line in datasets.splitlines()]]

        path = self.root.joinpath('session.yaml')
        path.write_text('\n'.join(lines) + '\n')
        _, _, data_paths, _ = load_alignment_yaml(path)

        return data_paths['default']['probe_00']

    def test_default_is_used_when_the_dataset_is_omitted(self):
        paths = self.load(
            defaults='raw_ephys:\n  path: spikeglx',
            datasets='spike_sorting:\n  path: ks',
        )
        self.assertEqual(paths.raw_ephys, self.root.joinpath('spikeglx'))

    def test_empty_specification_matches_an_omitted_dataset(self):
        # The documentation notes that both fall back to the default
        omitted = self.load(
            defaults='raw_ephys:\n  path: spikeglx',
            datasets='spike_sorting:\n  path: ks',
            probe_path='probe_00',
        )
        empty = self.load(
            defaults='raw_ephys:\n  path: spikeglx',
            datasets='spike_sorting:\n  path: ks\nraw_ephys: {}',
            probe_path='probe_00',
        )
        self.assertEqual(empty.raw_ephys, omitted.raw_ephys)
        self.assertEqual(empty.raw_ephys, self.root.joinpath('probe_00', 'spikeglx'))

    def test_dataset_path_overrides_the_default(self):
        paths = self.load(
            defaults='raw_ephys:\n  path: spikeglx',
            datasets='spike_sorting:\n  path: ks\nraw_ephys:\n  path: my_raw',
        )
        self.assertEqual(paths.raw_ephys, self.root.joinpath('my_raw'))

    def test_default_is_never_used_as_a_base_directory(self):
        # A dataset that gives its own relative path is resolved against the probe, config and
        # top level paths only; its default is not used at all, so the result is never
        # <default>/<dataset path>
        paths = self.load(
            defaults='raw_ephys:\n  path: base',
            datasets='spike_sorting:\n  path: ks\nraw_ephys:\n  path: my_raw',
            probe_path='probe_00',
        )
        self.assertEqual(paths.raw_ephys, self.root.joinpath('probe_00', 'my_raw'))
        self.assertNotEqual(paths.raw_ephys, self.root.joinpath('probe_00', 'base', 'my_raw'))

    def test_an_absolute_default_ignores_the_base_directories(self):
        absolute = self.root.joinpath('shared', 'histology')
        paths = self.load(
            defaults=f'histology:\n  path: {absolute}',
            datasets='spike_sorting:\n  path: ks',
            probe_path='probe_00',
        )
        self.assertEqual(paths.histology, absolute)

    def test_the_defaults_section_takes_datasets_not_a_base_path(self):
        # `path` under defaults is read as a dataset named "path", whose value must be a mapping
        text = '\n'.join([
            f'path: {self.root}', 'defaults:', '  path: /mnt/base', 'probes:', '  probe_00:',
            '    datasets:', '      spike_sorting:', '        path: ks',
        ])
        yaml_file = self.root.joinpath('session.yaml')
        yaml_file.write_text(text + '\n')

        with self.assertRaises(ValidationError):
            load_alignment_yaml(yaml_file)


class TestRequiredDatasets(unittest.TestCase):
    """Test the validation applied to the resolved dataset paths."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name).resolve()

    def tearDown(self):
        self.temp_dir.cleanup()

    def load(self, datasets):
        lines = [f'path: {self.root}', 'probes:', '  probe_00:', '    datasets:']
        for name, value in datasets.items():
            lines += [f'      {name}:', f'        path: {value}']
        path = self.root.joinpath('session.yaml')
        path.write_text('\n'.join(lines) + '\n')
        _, _, data_paths, _ = load_alignment_yaml(path)

        return data_paths['default']['probe_00']

    def test_spike_sorting_or_picks_is_required(self):
        with self.assertRaises(ValueError) as ctx:
            self.load({'histology': 'hist'})
        self.assertIn('probe_00', str(ctx.exception))

    def test_spike_sorting_alone_is_enough(self):
        paths = self.load({'spike_sorting': 'ks'})
        self.assertEqual(paths.spike_sorting, self.root.joinpath('ks'))
        self.assertIsNone(paths.picks)

    def test_picks_alone_is_enough(self):
        # Datasets with no spike sorting are supported; the output falls back to the picks
        paths = self.load({'picks': 'picks'})
        self.assertIsNone(paths.spike_sorting)
        self.assertEqual(paths.output, self.root.joinpath('picks'))


class TestMatchingProbes(unittest.TestCase):
    """Test that every configuration has to describe the same probes."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name).resolve()

    def tearDown(self):
        self.temp_dir.cleanup()

    def load(self, config_probes):
        lines = [f'path: {self.root}', 'configurations:']
        for cname, probes in config_probes.items():
            lines += [f'  {cname}:', '    probes:']
            for probe in probes:
                lines += [f'      {probe}:', '        datasets:', '          spike_sorting:',
                          f'            path: {probe}']
        path = self.root.joinpath('session.yaml')
        path.write_text('\n'.join(lines) + '\n')

        return load_alignment_yaml(path)

    def test_matching_probes(self):
        configs, probes, _, _ = self.load(
            {'dense': ['probe_00', 'probe_01'], 'sparse': ['probe_00', 'probe_01']}
        )
        self.assertEqual(configs, ['dense', 'sparse'])
        self.assertEqual(probes, ['probe_00', 'probe_01'])

    def test_mismatched_probe_names(self):
        # Left unchecked this fails later with a KeyError, when each configuration is indexed
        # with the union of the probes to build the shanks
        with self.assertRaises(ValueError) as ctx:
            self.load({'dense': ['probe_00'], 'sparse': ['probe_99']})
        self.assertIn('same probes', str(ctx.exception))
        self.assertIn('probe_99', str(ctx.exception))

    def test_a_configuration_missing_a_probe(self):
        with self.assertRaises(ValueError):
            self.load({'dense': ['probe_00', 'probe_01'], 'sparse': ['probe_00']})

    def test_a_single_configuration_is_not_compared(self):
        configs, probes, _, _ = self.load({'only': ['probe_00', 'probe_01']})
        self.assertEqual(configs, ['only'])
        self.assertEqual(probes, ['probe_00', 'probe_01'])


class TestLoaderErrors(unittest.TestCase):
    """Test the errors raised for missing or malformed yaml files."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name).resolve()

    def tearDown(self):
        self.temp_dir.cleanup()

    def write(self, text, name='session.yaml'):
        path = self.root.joinpath(name)
        path.write_text(text)

        return path

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            load_alignment_yaml(self.root.joinpath('does_not_exist.yaml'))

    def test_file_without_a_mapping(self):
        for label, text in [('empty', ''), ('bare scalar', 'a string\n'), ('list', '- a\n- b\n')]:
            with self.subTest(label), self.assertRaises(ValueError):
                load_alignment_yaml(self.write(text))

    def test_more_than_two_configurations(self):
        text = '\n'.join(
            [f'path: {self.root}', 'configurations:']
            + [f'  config_{i}:\n    probes:\n      probe_00:\n        datasets:'
               f'\n          spike_sorting:\n            path: ks' for i in range(3)]
        )
        with self.assertRaises(AssertionError):
            load_alignment_yaml(self.write(text + '\n'))

    def test_histology_space(self):
        base = [f'path: {self.root}', 'probes:', '  probe_00:', '    datasets:',
                '      spike_sorting:', '        path: ks']

        with self.subTest('defaults to ccf'):
            _, _, _, space = load_alignment_yaml(self.write('\n'.join(base) + '\n'))
            self.assertEqual(space, 'ccf')

        with self.subTest('read from the defaults section'):
            text = '\n'.join(
                [f'path: {self.root}', 'defaults:', '  histology:', '    path: hist',
                 '    space: anatomical'] + base[1:]
            )
            _, _, _, space = load_alignment_yaml(self.write(text + '\n'))
            self.assertEqual(space, 'anatomical')

        with self.subTest('a probe level space is ignored'):
            text = '\n'.join(base + ['      histology:', '        path: hist',
                                     '        space: anatomical'])
            _, _, _, space = load_alignment_yaml(self.write(text + '\n'))
            self.assertEqual(space, 'ccf')

        with self.subTest('an unknown space is rejected'):
            text = '\n'.join(
                [f'path: {self.root}', 'defaults:', '  histology:', '    path: hist',
                 '    space: nonsense'] + base[1:]
            )
            with self.assertRaises(ValidationError):
                load_alignment_yaml(self.write(text + '\n'))


if __name__ == '__main__':
    unittest.main()
