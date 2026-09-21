"""Every movie's tables, concatenated once, so a comparison is a filter.

A run measures each movie on its own and writes each one its own folder. That
is right for measuring and wrong for reading, because the first thing anyone
wants is two movies side by side.

Nothing here computes. Every row is already stamped with the movie it came
from, the condition it belongs to and the subject it was taken from, so
pooling is stacking. What this module adds is the record of what was stacked:
which movies contributed to each table, how many rows each gave, which
columns one movie had that another did not, and which tables a movie never
produced at all. That record is the load-bearing part: a movie that declared
no object set has no object columns, and a pooled table that quietly fills
them with NaN looks exactly like a movie whose cells touched no objects.

Ported from Motion's ``analysis/pool.py`` on 2026-09-21. The pooled tables
land flat in ``<run>/pooled/`` with one ledger; each table's record says
which kind folder (``measure`` or ``tracker``) it was pooled from, so the
origin contract survives the concatenation. The record is a section of the
run manifest rather than a ``pooled/manifest.json`` of its own.

No pandas at import time: this module is imported by the action registry.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

__all__ = ["STAMP_COLUMNS", "SOURCE_FOLDERS", "movie_folders", "pool_run", "pool"]

#: What ``run.stamp`` puts on the front of every written table. A table
#: without them cannot be pooled into anything a later stage can group by.
STAMP_COLUMNS = ("stem", "condition", "subject")

#: The kind folders a run's per-movie tables are split across.
SOURCE_FOLDERS = ("measure", "tracker")


def movie_folders(run_dir: Path) -> list[str]:
    """The stems a run measured, in a fixed order: the folders under ``measure/``."""
    base = Path(run_dir) / SOURCE_FOLDERS[0]
    if not base.is_dir():
        return []
    return [p.name for p in sorted(base.iterdir()) if p.is_dir() and not p.name.startswith(".")]


def _ordered_union(columns_per_movie: list[list[str]]) -> list[str]:
    """The union of several column lists, in first-seen order."""
    ordered: list[str] = []
    seen: set[str] = set()
    for columns in columns_per_movie:
        for column in columns:
            if column not in seen:
                seen.add(column)
                ordered.append(column)
    return ordered


def _read(path: Path) -> "pd.DataFrame":
    """One movie's table, read back exactly as it was written."""
    import pandas as pd

    return pd.read_csv(path)


def pool_run(run_dir: str | Path, *, run_label: str | None = None) -> dict:
    """Write ``<run>/pooled/`` from the per-movie folders already in ``run_dir``.

    Reads only per-movie folders and writes only into ``pooled/``, so
    re-running it after adding a movie is correct. It still refuses to
    overwrite, in line with immutable runs.

    Returns the manifest fragment the caller writes into the run manifest.
    """
    import pandas as pd

    from .. import store
    from .declare import declared_tables
    from .run import POOLED_FOLDER, write_table

    run_dir = Path(run_dir)
    pooled_dir = run_dir / POOLED_FOLDER
    if pooled_dir.exists() and any(p.is_file() for p in pooled_dir.iterdir()):
        raise FileExistsError(
            f"{pooled_dir} already holds pooled tables; analysis runs are "
            "immutable, delete it deliberately or pool into a new run"
        )

    stems = movie_folders(run_dir)
    if not stems:
        raise ValueError(
            f"{run_dir} holds no movie folders; a run this package wrote has "
            f"one folder per movie under '{SOURCE_FOLDERS[0]}'"
        )

    # Collected first, written second, so a table that turns out to be
    # unpoolable stops the whole thing before half a pooled folder exists.
    collected: dict[tuple[str, str], dict[str, tuple[Path, pd.DataFrame]]] = {}
    for stem in stems:
        for folder in SOURCE_FOLDERS:
            source = run_dir / folder / stem
            if not source.is_dir():
                continue
            for path in sorted(source.glob("*.csv")):
                table = _read(path)
                missing = [c for c in STAMP_COLUMNS if c not in table.columns]
                if missing:
                    raise ValueError(
                        f"{path} is missing {missing}, so its rows cannot be told "
                        "apart from another movie's once pooled; every table a run "
                        "writes is stamped, so this file was not written by this package"
                    )
                collected.setdefault((folder, path.stem), {})[stem] = (path, table)

    declared = declared_tables()
    label = run_label or run_dir.name
    record: dict[str, dict] = {}
    for (folder, name), by_movie in sorted(collected.items()):
        contributing = [stem for stem in stems if stem in by_movie]
        union = _ordered_union([list(by_movie[stem][1].columns) for stem in contributing])
        frames = [by_movie[stem][1].reindex(columns=union) for stem in contributing]
        pooled = pd.concat(frames, ignore_index=True, sort=False)

        source = store.collection([by_movie[stem][0] for stem in contributing],
                                  path=str(pooled_dir / f"{name}.csv"))
        written = write_table(pooled, name=name, folder=pooled_dir, source=source,
                              params={"run": label, "table": name, "pooled": True,
                                      "movies": contributing},
                              output=declared.get(name))
        written["origin_folder"] = folder
        written["movies"] = contributing
        written["rows_per_movie"] = {stem: int(len(by_movie[stem][1])) for stem in contributing}
        columns_missing = {
            stem: [c for c in union if c not in by_movie[stem][1].columns]
            for stem in contributing
        }
        written["columns_missing"] = {k: v for k, v in columns_missing.items() if v}
        written["movies_absent"] = [stem for stem in stems if stem not in by_movie]
        record[name] = written

    return {
        "movies": stems,
        "folders": list(SOURCE_FOLDERS),
        "tables": record,
    }


def pool(run_dir) -> dict[str, Any]:
    """Concatenate an existing run's per-movie tables into ``<run>/pooled/``.

    Separate from ``measure`` so that a run measured before pooling existed can
    be pooled without re-measuring. The record of what was pooled goes into
    the run manifest's ``pooled`` section.
    """
    from .run import read_manifest, write_manifest

    run = Path(run_dir)
    manifest = read_manifest(run)
    fragment = pool_run(run, run_label=manifest.get("run", {}).get("run_label"))
    manifest["pooled"] = fragment
    write_manifest(run, manifest)
    return fragment
