# Documentation Maintenance

## Local Preview

Install the documentation dependencies:

```powershell
python -m pip install -r docs/requirements.txt
```

Run a local development server:

```powershell
python -m mkdocs serve
```

Build with warnings treated as failures before committing:

```powershell
python -m mkdocs build --strict
```

## Update Routes

| Code change | Documentation to check |
|---|---|
| Public import added or removed | `api-reference.md` |
| Action added or renamed | `actions/README.md`, catalogue, and relevant workflow |
| Action parameters changed | Detailed action page and examples |
| Pipeline stages or refusals changed | Pipeline page and `concepts/analysis-flow.md` |
| Artefact key or tier behavior changed | `concepts/artefact-store.md` and results pages |
| Doctor field changed | `troubleshooting/doctor.md` |
| New stable user route | `workflows/` and wiki home |

## Action Catalogue

The shipped action catalogue is generated from the protocol vocabulary and
then owned by this package:

```powershell
python tools/regenerate_catalogue.py
python tools/regenerate_catalogue.py --check
```

Do not hand-copy a parameter list into a wiki page without checking
`pymicroglia describe ACTION`; defaults are action-specific.

## Navigation

MkDocs Awesome Pages reads `.pages` files. Add a page to the nearest folder's
`.pages` file and to its index page. Root navigation belongs in
`docs/wiki/.pages`.

`CONTEXT.md` files are maintainer instructions and are excluded from the public
site by `mkdocs.yml`.

## Publication

Read the Docs builds from `.readthedocs.yaml` using `mkdocs.yml`. A documentation
change becomes public after it lands on the repository's default branch and the
Read the Docs build succeeds.

## Before Committing

```powershell
python -m mkdocs build --strict
python tools/regenerate_catalogue.py --check
python -m pytest -q
```

For documentation-only changes, the strict documentation build and catalogue
check are the minimum; run the full test suite when examples exercise changed
code.
