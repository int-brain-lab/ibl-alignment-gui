# AGENTS.md

Guidance for AI coding agents working on **ibl-alignment-gui**.

`ibl_alignment_gui` is a pyqtgraph/Qt app for aligning electrophysiology features along a probe
to the histology track. Entry points: `alignment-gui` (local), `alignment-gui-ibl` (ONE/Alyx,
needs the `ibl` extra), `alignment-gui-allen` (Allen DocDB, needs the `allen` extra).

## Package manager

This project uses [`uv`](https://docs.astral.sh/uv/).

## Commands

- Install/sync dev environment: `uv sync` (add `--extra ibl` to run the ONE/Alyx tests)
- Tests: `uv run pytest` (headless: `QT_QPA_PLATFORM=offscreen uv run pytest`)
- Lint: `uv run ruff check`
- Format: `uv run ruff format`
- All pre-commit hooks: `uv run pre-commit run --all-files`
- Build docs: `uv run sphinx-build -W --keep-going docs/source docs/build` (as in CI)

## Repository structure

- `ibl_alignment_gui/app/` — launchers, controllers, views, widgets, load worker
- `ibl_alignment_gui/handlers/` — probe, shank and alignment state; builds the loaders
- `ibl_alignment_gui/loaders/` — data, geometry, histology and alignment sources; plot data
- `ibl_alignment_gui/core/` — alignment maths
- `ibl_alignment_gui/plugins/` — optional features, registered in `add_plugins.py`
- `ibl_alignment_gui/backends/allen/`, `convertors/`, `utils/` — Allen mode, sorter-to-ALF
  scripts, yaml parsing and helpers
- `tests/` — mirrors the package; data in `tests/fixtures/`
- `docs/source/` — Sphinx docs (tutorials, how-to, explanation, reference)
- `examples/` — runnable example scripts

## Guides

Load these only when relevant to the task at hand:

- [Documentation lookup](.agents/docs-lookup.md) — check before answering any
  "how does X behave" question; don't guess when docs exist.
- [Architecture](.agents/architecture.md) — loader backends, missing data, threading.
- [Data access](.agents/data-access.md) — ONE/Alyx, raw data and optional extras.
- [References](.agents/references.md) — related IBL and SpikeInterface repositories.
- [Development principles](.agents/development-principles.md)
- [Code style](.agents/code-style.md) / [docstrings](.agents/docstrings.md)
- [Testing](.agents/testing.md)
- [Definition of done](.agents/definition-of-done.md) — checklist before finishing a code
  change.
- [Plan mode](.agents/plan-mode.md)
- [Pull requests](.agents/pull-requests.md)
