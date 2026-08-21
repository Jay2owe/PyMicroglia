# Wiki Context

This wiki is the long-form PyMicroglia reference. Future maintainers should use
it to expand user-facing documentation without losing the conventions started
here.

## Current Style

- Explain the purpose in plain language before implementation detail.
- Prefer copy-pasteable examples using public `pymicroglia` imports.
- Put complete workflows under `pipelines/` or `workflows/` and individual
  registered operations under `actions/`.
- Treat scientific refusal conditions, display-only products, controls, and
  provenance as user-facing behavior.
- Separate returned Python objects from files saved to disk.
- Use path placeholders; never copy private local paths into public pages.

Read these first:

- [README](README.md) for the entry point and coverage map.
- [Documentation standard](documentation-standard.md) for page shapes.
- [API reference](api-reference.md) for the public Python surface.
- [Analysis flow](concepts/analysis-flow.md) for the package vocabulary.

## Folder Map

| Folder | Purpose |
|---|---|
| `getting-started/` | Installation, first run, and first review. |
| `concepts/` | Analysis flow, artefact storage, provenance, and scientific boundaries. |
| `actions/` | Registered operations and how to call them. |
| `pipelines/` | End-to-end analyses. |
| `results/` | `Recording`, `Batch`, files, sidecars, and run records. |
| `workflows/` | Task-based guides that cross several APIs. |
| `troubleshooting/` | Common failures and diagnostic routes. |
| `glossary/` | Short searchable definitions. |
| `developer/` | Maintainer-facing implementation and documentation guidance. |

## Sources of Truth

- Public imports: `src/pymicroglia/__init__.py`
- Command line: `src/pymicroglia/cli.py`
- Action catalogue: `src/pymicroglia/data/actions.json`
- Live action bindings: `src/pymicroglia/registry.py`
- Pipelines: `src/pymicroglia/pipelines/`
- Artefact store: `src/pymicroglia/store/`
- Result objects: `src/pymicroglia/results.py`
- Tests: `tests/`

## Expansion Workflow

1. Read this file, the relevant existing page, and the source or tests.
2. Put the new material in the most specific folder.
3. Add a page to the relevant index and `.pages` file.
4. Run `python -m mkdocs build --strict`.
5. Keep commands safe on large, synced microscopy data: never introduce a
   recursive scan where the package deliberately uses a one-folder lookup.

## What Not to Do

- Do not infer behavior from an action name; inspect the catalogue and target.
- Do not present display-only filtering as measurement.
- Do not omit required controls or refusal conditions from workflow pages.
- Do not promise that a cache hit exists; explain the key and the fallback.
- Do not copy long implementation blocks into the wiki.
