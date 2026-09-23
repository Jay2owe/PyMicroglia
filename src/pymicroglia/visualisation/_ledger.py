"""Record rendered figures in the shared artefact ledger."""
from pathlib import Path


def record(figures, table, *, sources, settings, claim, artefacts=(), notes=(), drawn=None):
    from .. import store, __version__
    from .bundle import _artefact_rows
    from auto_organotypic.store.ledger import ledger_path
    files = [Path(row["path"]) for row in sources if row.get("exists")]
    # A figure drawn directly from in-memory data is reproducible from its CSV.
    source = store.collection(files or [table], path=str(Path(table).parent))
    extra = {"table": Path(table).name, "sources": sources, "claim": claim,
             "notes": list(notes), "drawn": drawn,
             "artefacts": _artefact_rows(artefacts)}
    for path in figures:
        store.claim("figure", source, {**settings, "output": Path(path).name},
                    path=path, method_version=__version__, display_only=True, extra=extra)
    return ledger_path(Path(table).parent)
