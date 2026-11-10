# Testing

Tests run with pytest but are written in `unittest` style. Follow these conventions in
new tests, and bring old tests in line when you touch them.

## Layout

- Mirror the package: `ibl_alignment_gui/loaders/data_loader.py` ->
  `tests/loaders/test_data_loader.py`. Each test directory has an `__init__.py`.
- Test data lives in `tests/fixtures/`; build its path from one module-level
  `FIXTURE_PATH = Path(__file__).parents[1].joinpath('fixtures')`.

## Structure

- One `unittest.TestCase` class per class or function under test, named
  `Test<Name>` (`TestDataLoaderLocal`, `TestFindFeaturesFile`).
- Shared state goes in `setUp` / `setUpClass`; cover variants of one behaviour with
  `self.subTest(...)`. Don't use pytest fixtures, `parametrize` or `conftest.py`.
- Helper classes and stand-ins are private: `_Controller`, `_Recorder`.
- Qt tests create the app once per class:
  `cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])`, and
  run headless (`QT_QPA_PLATFORM=offscreen`, as in CI).

## Naming and docstrings

- Name tests after the behaviour, not the method alone:
  `test_get_spike_sorting_path_prefers_iblsorter`, not `test_get_spike_sorting_path`
  or `test_one`.
- Every test has a one-line docstring stating the expected behaviour, without a
  trailing period: `"""Without spike sorting the loader returns exists=False"""`.

## Assertions

- Use `self.assert*`; `np.testing.assert_array_equal` / `assert_allclose` for arrays.
- Compare values with `assertEqual`, never `assertTrue(result, expected)`: the second
  argument is only the failure message, so that assertion passes for any truthy result.

## Mocking

- Use `unittest.mock`, patching with `with patch(...)` inside the test so the scope is
  visible. Patch where the name is looked up (`ibl_alignment_gui.loaders.geometry_loader.spikeglx`,
  not another module that imports the same package).
- Give mocks of this package's classes a spec (`MagicMock(spec=DataLoaderLocal)`), so a
  renamed or removed method fails the test.
- Mock ONE/Alyx and the file system at the boundary; tests never hit the network.

## Other rules

- Tests should be concise and run in little time.
- Warnings are errors (`filterwarnings = ["error"]`): fix the cause, or ignore a
  third-party warning narrowly in `pyproject.toml`.
- Skip tests that need the `ibl` extra with
  `@unittest.skipUnless(has_ibllib(), IBL_EXTRA_HINT)`.

See also [data-access.md](data-access.md) for mocking ONE/Alyx and raw data.
