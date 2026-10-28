(contributing-writing-docs)=
# Writing documentation

The documentation follows the [Diátaxis](https://diataxis.fr/) framework. Every page is one of
four kinds, and each kind has its own folder under `docs/source`:

| Kind | Folder | Answers | Written for |
|---|---|---|---|
| Tutorial | `tutorials/` | "Teach me" | Someone learning the GUI, on the sample data |
| How-to guide | `how-to/` | "How do I …?" | Someone who knows the GUI and has a task to do |
| Explanation | `explanation/` | "Why …?" | Someone who wants to understand |
| Reference | `reference/` | "What exactly is …?" | Someone looking up a fact |

A page does one of these jobs. When a tutorial needs to explain something, or a how-to needs a full
table of options, link to the explanation or reference page instead of including it.

## Page templates

### Tutorial

1. What you'll do: one sentence, and a picture of the result.
2. Before you start: what to install and which data to download.
3. Numbered steps. Each step that changes the display ends with what the learner should see.
4. What you've learned.
5. Next steps: links to the next tutorial and related pages.

Tutorials never offer alternatives ("if you use OpenEphys …"). They use the sample data, so every
step has one right outcome.

### How-to guide

1. Title starting "How to …".
2. One sentence on the goal.
3. Prerequisites.
4. Steps.
5. Related: links to the reference and explanation pages.

How-to guides can be short. They assume the reader already knows the GUI.

### Explanation

1. The question the page answers.
2. Discussion, with diagrams where they help.
3. Trade-offs and limits.
4. Further reading.

Explanation pages contain no steps to follow and no exhaustive option lists.

### Reference

1. One sentence on what the page covers.
2. Entries in a consistent format: the same fields for every item.
3. Defaults and fallbacks stated for every option.

Reference pages contain no instructions. Tables of keyboard shortcuts, session YAML keys and
datasets live only in `reference/`; other pages link to them rather than repeating them.

## Files and labels

- New pages are Markdown (`.md`), parsed by [MyST](https://myst-parser.readthedocs.io/). Only
  `api.rst`, `changelog.rst` and the autosummary templates are reStructuredText.
- File names are lowercase with underscores, e.g. `how_alignment_works.md`. Tutorials are
  numbered, e.g. `01_first_alignment.md`.
- Every page starts with a label, prefixed by its kind, so that it can be linked to wherever it
  lives:

  ```markdown
  (ref-session-yaml)=
  # Session YAML
  ```

  Prefixes: `tutorial-`, `howto-`, `explanation-`, `ref-`, `contributing-`.
- Link to pages through their label, `` {ref}`ref-session-yaml` ``, rather than their path, so
  that moving a page doesn't break links to it.
- When a page moves or is removed, add its old path to `redirects` in `conf.py`.

## MyST syntax used in these docs

````markdown
Keyboard shortcut:        {kbd}`Shift+A`
Link to a label:          {ref}`ref-session-yaml`
Link with custom text:    {ref}`the session YAML keys <ref-session-yaml>`
Link to a page:           {doc}`../installation`
Python object:            {class}`~ibl_alignment_gui.loaders.data_loader.DataLoaderLocal`
Package version:          {{ version_code }}

```{note}
An admonition. Also: warning, tip, important.
```

```{list-table}
:header-rows: 1
:widths: 30 70

* - Shortcut
  - Action
* - {kbd}`Enter`
  - Apply the fit
```

::::{tab-set}
:::{tab-item} Linux and macOS
...
:::
:::{tab-item} Windows
...
:::
::::

:::{dropdown} Title
Hidden until clicked.
:::
````

Simple tables can use the Markdown pipe syntax instead of `list-table`.

## Placeholders

Mark anything still to be written with a `TODO` admonition, so it stands out on the page:

```markdown
:::{admonition} TODO
:class: warning

Screenshot of the GUI after loading the data.
:::
```

Pages with placeholders can be merged into working branches, but the documentation workflow fails
on any `TODO` in a pull request to `main`, so they cannot reach the published site.

## Images

Screenshots go in `_static/`, in a folder per page, e.g. `_static/tutorials/01_first_alignment/`.
Take GUI screenshots at the same window size (1600 × 1000) and in the light theme, so that pages
look consistent.

## Building

```console
uv run --group doc sphinx-build -W --keep-going docs/source docs/build
```

`-W` turns warnings into errors, as in CI, so a broken link or cross-reference fails the build.
Open `docs/build/index.html` to view the result. To rebuild automatically while editing:

```console
uv run --group doc sphinx-autobuild docs/source docs/build
```
