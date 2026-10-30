# Development principles

- Keep public APIs, the session yaml format and saved alignment files backwards
  compatible unless explicitly requested.
- Prefer small, focused changes; keep diffs minimal and avoid unrelated refactoring.
- State the goal in one sentence first; the smallest change achieving it is the scope.
- Review findings are input, not a to-do list: fix only bugs the change introduces;
  list the rest as follow-ups and ask before widening the scope.
- A simplification pass must shrink the diff.
- Avoid introducing new dependencies without justification; add them to
  `pyproject.toml` unpinned and commit the updated `uv.lock`. Heavy or IBL-only
  dependencies go in an extra, not the core dependencies.
- Prefer composition over large inheritance hierarchies.
- Follow the patterns in [architecture.md](architecture.md).
