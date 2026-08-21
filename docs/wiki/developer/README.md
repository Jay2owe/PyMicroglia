# Developer Documentation

## Sources of Truth

| Concern | Source |
|---|---|
| Public imports | `src/pymicroglia/__init__.py` |
| Command line | `src/pymicroglia/cli.py` |
| Action descriptions and declared defaults | `src/pymicroglia/data/actions.json` |
| Live action bindings and claim rules | `src/pymicroglia/registry.py` |
| Run capture and equivalent scripts | `src/pymicroglia/recording.py` |
| Pipeline order and run-folder policy | `src/pymicroglia/pipelines/__init__.py` |
| Artefact keys and storage | `src/pymicroglia/store/` |
| Lazy result objects | `src/pymicroglia/results.py` |
| Behavioral guarantees | `tests/` |

## Design Boundaries to Preserve

- Keep imports cheap; scientific modules load on use.
- Keep the package self-contained from the historical protocol scripts.
- Bind actions by dotted name and validate arguments before execution.
- Record resolved function defaults, not only catalogue defaults.
- Keep run recording soft: losing audit support must not lose a scientific result.
- Keep rhythm analysis hard: a missing scientific dependency must not silently
  skip analysis.
- Register before cleaning and keep display after measurement.
- Refuse ambiguous stored inputs.
- Avoid recursive scans of synced microscopy folders.

## Extending Documentation

Read [Documentation maintenance](docs-maintenance.md), the root maintainer
`CONTEXT.md`, and the [documentation standard](../documentation-standard.md)
before adding pages.
