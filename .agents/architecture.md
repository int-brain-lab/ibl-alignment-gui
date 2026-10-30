# Architecture

- **Backends:** an ABC defines each data source and backends subclass it with a suffix
  (`DataLoaderOne` / `DataLoaderLocal`, `GeometryLoaderOne` / `GeometryLoaderLocal`,
  `AlignmentLoaderOne` / `AlignmentLoaderLocal`). `handlers/probe_handler.py` picks the
  backend; nothing downstream checks which one it got.
- **Missing data does not raise:** loaders return a `Bunch` with `exists=False` and log
  a warning; plot methods are guarded with `@skip_missing([...])`. Missing data means
  fewer plots, not a crash.
- **Threading:** data loading runs in `app/load_worker.py` off the UI thread. Loaders
  and handlers must not touch Qt widgets.
- **MVC:** `app/controllers/` hold logic, `app/views/` hold widgets, `handlers/` hold
  state. Keep Qt out of `core/`, `loaders/` and `handlers/`.
- **Plugins:** optional features live in `plugins/` and are registered in
  `add_plugins.py`; they should not be required for the GUI to start.
- **Session config:** local sessions are described by a yaml parsed into `DatasetPaths`
  (`utils/parse_yaml.py`). New data locations are new `DatasetPaths` fields with a
  documented fallback.
