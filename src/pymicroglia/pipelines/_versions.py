"""Content fingerprints for packaged renderers and their preparation code."""
from pathlib import Path
from ._contracts import content_id
from ._screening import file_hash


def rendering(source):
    root = Path(__file__).parents[1]
    files = {Path(source), root / "pipelines" / "_saved_figures.py"}
    for directory in ("visualisation", "figure_tables"):
        files.update((root / directory).rglob("*.py"))
    return content_id({p.relative_to(root).as_posix(): file_hash(p) for p in sorted(files)})
