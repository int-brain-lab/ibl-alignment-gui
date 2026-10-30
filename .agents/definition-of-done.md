# Definition of done

Before considering work complete:

1. Run `uv run pytest`, `uv run ruff check` and `uv run ruff format --check`.
2. Add tests for new functionality.
3. Update existing tests if behavior changes.
4. Ensure examples still work if affected.
5. Update docstrings and documentation (`docs/source/`) where relevant.
6. Add a concise note under `## [Unreleased]` in `CHANGELOG.md` for user-facing
   changes (Keep a Changelog format).
7. The diff contains only what the stated goal needs.
