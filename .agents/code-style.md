# Code style

- Favor correctness, backwards compatibility, and readability over clever
  implementations.
- Match surrounding coding style; formatting/lint style itself is enforced by
  `ruff format` / `ruff check` (see Commands in the root AGENTS.md). Run ruff through
  `uv run` so the version matches CI.
- Line length: 99 characters, including in docstrings. Single quotes. Trailing comma
  on multi-line calls and signatures.
- Type annotations are mandatory throughout new code: `X | None`, `list[int]`,
  `dict[str, np.ndarray]`. Use `from __future__ import annotations` and keep type-only
  imports under `if TYPE_CHECKING:`.
- Absolute imports only. `ibl*`, `one*` and `brainbox*` sort as first-party.
- Log with `logger = logging.getLogger(__name__)`, never `print`. Say what is missing
  and what the user will see, e.g. `'... was not found, some plots will not display'`.
- Units: SI inside the code (V, s), µm for probe geometry; convert only for display
  (`* 1e6`).
- Import Qt through `qtpy`, never `PyQt5` directly.
- Docstrings: see [docstrings.md](docstrings.md).
