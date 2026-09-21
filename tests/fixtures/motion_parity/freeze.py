"""Freeze what the Motion analysis produced on the synthetic movie.

One-shot. It reads finished output folders and writes ``expected.json``, the
record every later port stage compares against. It never runs the analysis
itself: the Motion commands are run in the Motion checkout (see README.md),
and this script only fingerprints what they wrote.

    python tests/fixtures/motion_parity/freeze.py \
        --run <run folder> \
        --figure-log <json written by the figure driver> \
        --pipeline rhythm-discovery=<folder> --pipeline method-audit=<folder> ... \
        --extra states=<folder> --extra cluster=<folder> \
        --commands <json of the standalone command outcomes> \
        --copy-tables \
        --out tests/fixtures/motion_parity/expected.json

What is frozen, and what is not:

* every CSV: SHA-256, row count, column names, and the first and last three
  rows as they are written (Motion writes nine significant figures);
* every figure: the SHA-256 of each plotted-data CSV in its bundle, or the
  refusal text when the figure declined the synthetic movie. The SVG is not
  frozen: fonts and matplotlib versions move it;
* every pipeline demo: the status of each step and the SHA-256 of each
  artefact its ``execution-result.json`` names, plus the verification checks;
* the whole ``manifest.json`` of the measurement run.

``--copy-tables`` also copies the run's CSV and JSON files (never its stacks
or figures) under ``run/`` beside ``expected.json``, so a later stage can diff a
whole table rather than three rows of it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
EXPECTED = HERE / "expected.json"
SAMPLE_ROWS = 3

#: Folders inside a run that hold pixels or drawings rather than tables.
SKIPPED_FOLDERS = {"figures", "stacks", "__pycache__"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv_as_written(path: Path) -> pd.DataFrame:
    """The CSV exactly as text: no dtype inference, no NaN substitution."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)


def table_record(path: Path, *, samples: bool = True) -> dict:
    """Hash, shape and columns of one CSV, with its first and last rows if asked.

    The measurement run's tables carry samples (the stage asks for them, and
    they are what a reviewer reads when a hash moves). A pipeline demo writes
    hundreds of tables whose artefact hashes the runner already records, so
    those are frozen by hash and shape alone, or the record would be 29 MB.
    """
    try:
        frame = read_csv_as_written(path)
    except pd.errors.EmptyDataError:
        # A table with no header at all: written when a step had nothing to say.
        frame = pd.DataFrame()
    record = {"sha256": sha256(path), "rows": int(len(frame)), "columns": list(frame.columns)}
    if samples:
        record["head"] = frame.head(SAMPLE_ROWS).to_dict(orient="records")
        record["tail"] = frame.tail(SAMPLE_ROWS).to_dict(orient="records")
    return record


def csv_files(folder: Path) -> list[Path]:
    found = []
    for path in sorted(folder.rglob("*.csv")):
        relative = path.relative_to(folder)
        if any(part in SKIPPED_FOLDERS for part in relative.parts[:-1]):
            continue
        found.append(path)
    return found


def freeze_tables(folder: Path, *, samples: bool = True) -> dict[str, dict]:
    """Every CSV under ``folder``, keyed by its POSIX path relative to it."""
    return {path.relative_to(folder).as_posix(): table_record(path, samples=samples)
            for path in csv_files(folder)}


def freeze_measure(run: Path) -> dict:
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    named = 0
    for movie in manifest.get("movies", []):
        named += len(movie.get("tables", {}))
    named += len(manifest.get("pooled", {}).get("tables", {}))
    named += 1 if manifest.get("statistics", {}).get("path") else 0
    return {
        "tables": freeze_tables(run),
        "manifest": manifest,
        "manifest_table_count": named,
    }


def freeze_figures(run: Path, log: dict) -> dict:
    """One entry per figure the driver attempted, in the driver's own order.

    ``log`` maps a builder file name to ``{"returncode", "stdout", "stderr",
    "slug"}``. A builder that exited non-zero is recorded with its refusal text
    and no plotted hash: the refusal is the expected outcome on this movie.
    """
    figures = {}
    root = run / "figures"
    for builder, outcome in log.items():
        slug = outcome.get("slug")
        entry: dict = {"builder": builder, "review": bool(outcome.get("review")),
                       "returncode": int(outcome["returncode"])}
        bundle = root / slug if slug else None
        plotted = {}
        if bundle is not None and bundle.is_dir():
            for path in sorted(bundle.rglob("*.csv")):
                plotted[path.relative_to(bundle).as_posix()] = sha256(path)
        if outcome["returncode"] != 0 or not plotted:
            text = (outcome.get("stderr") or outcome.get("stdout") or "").strip()
            entry["refusal"] = text.splitlines()[-1] if text else ""
        if plotted:
            entry["plotted_sha256"] = plotted
        figures[slug or builder] = entry
    return figures


def freeze_pipeline(folder: Path, scrub_root: Path | None = None) -> dict:
    """Step statuses and artefact hashes from every ``execution-result.json``.

    The runner writes one record per step per cache entry, under
    ``<pipeline>/<science|renders>/<step>/<cache id>/<execution id>/``. A step
    that ran twice (the demos re-render with changed presentation) has two
    entries; both are kept, in path order. An artefact's hash is the one the
    record carries, checked against the file beside the record when it exists.
    """
    steps: dict[str, list] = {}
    for result_path in sorted(folder.rglob("execution-result.json")):
        document = json.loads(result_path.read_text(encoding="utf-8"))
        result = document.get("result", document)
        step = result.get("step") or result_path.parent.name
        artifacts = {}
        listed = result.get("artifacts") or []
        if isinstance(listed, dict):
            listed = [{"name": name, **(value if isinstance(value, dict) else {"path": value})}
                      for name, value in listed.items()]
        for item in listed:
            name = item.get("name") or item.get("path")
            recorded = item.get("sha256")
            candidate = result_path.parent / str(item.get("path") or name)
            actual = sha256(candidate) if candidate.is_file() else None
            artifacts[name] = {
                "sha256": recorded or actual,
                "present": candidate.is_file(),
                "matches_record": (actual == recorded) if (actual and recorded) else None,
            }
        steps.setdefault(step, []).append({
            "status": result.get("status"),
            "reason": result.get("reason"),
            "artifacts": artifacts,
            "record": result_path.relative_to(folder).as_posix(),
        })
    verification = None
    proof = folder / "verification.json"
    if proof.exists():
        verification = scrub(json.loads(proof.read_text(encoding="utf-8")), scrub_root)
    return {"steps": steps, "verification": verification,
            "tables": freeze_tables(folder, samples=False)}


def scrub(document, roots: dict[str, Path] | None):
    """Absolute paths under each root become ``<name>/...``, so the record is portable.

    Every spelling of a root is replaced, because Windows records carry
    backslashes, JSON doubles them, and the same path typed on a POSIX shell
    carries forward slashes. The scrubbed remainder is written with forward
    slashes whatever it arrived with.
    """
    if not roots:
        return document
    forms = []
    for name, root in roots.items():
        for form in (str(root), root.as_posix()):
            if form:
                forms.append((form, f"<{name}>"))
    forms.sort(key=lambda pair: -len(pair[0]))
    runs_of_backslashes = re.compile(r"\\+")

    def clean(value):
        if isinstance(value, str):
            # A path copied through JSON, then into a CSV cell, then into JSON
            # again arrives with its backslashes doubled once per hop. One
            # backslash is one separator whatever the hop count.
            candidate = runs_of_backslashes.sub("\\\\", value)
            hit = False
            for form, marker in forms:
                if form in candidate:
                    candidate = candidate.replace(form, marker)
                    hit = True
            return candidate.replace("\\", "/") if hit else value
        if isinstance(value, dict):
            return {clean(k): clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value
    return clean(document)


def parse_roots(items: list[str]) -> dict[str, Path]:
    """``NAME=PATH`` pairs; a bare path is named ``fixture``."""
    roots = {}
    for item in items:
        name, sep, path = item.partition("=")
        if not sep:
            name, path = "fixture", item
        roots[name] = Path(path).resolve()
    return roots


def copy_tables(run: Path, target: Path) -> list[str]:
    """Copy the run's CSV and JSON files under ``target``, skipping pixels."""
    copied = []
    for path in sorted(run.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".csv", ".json"}:
            continue
        relative = path.relative_to(run)
        if any(part in SKIPPED_FOLDERS for part in relative.parts[:-1]):
            continue
        # One movie: every pooled table is byte-identical to the per-movie one
        # (expected.json still records its hash), so copying it would double
        # the fixture for nothing. The pooled manifest is kept.
        if relative.parts[0] == "pooled" and path.suffix.lower() == ".csv":
            continue
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        copied.append(relative.as_posix())
    return copied


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--figure-log", type=Path)
    parser.add_argument("--pipeline", action="append", default=[], metavar="NAME=FOLDER")
    parser.add_argument("--extra", action="append", default=[], metavar="NAME=FOLDER")
    parser.add_argument("--commands", type=Path,
                        help="JSON recording the standalone commands' outcomes "
                             "(pool, window, contrasts, videos, states, cluster)")
    parser.add_argument("--counts", type=Path, help="JSON of expected counts")
    parser.add_argument("--environment", type=Path, help="JSON of versions and commits")
    parser.add_argument("--scrub", action="append", default=[], metavar="NAME=PATH",
                        help="absolute paths under PATH are recorded as <NAME>/...; "
                             "repeatable, and a bare PATH is named fixture")
    parser.add_argument("--copy-tables", action="store_true")
    parser.add_argument("--out", type=Path, default=EXPECTED)
    args = parser.parse_args(argv)

    roots = parse_roots(args.scrub)
    run = args.run.resolve()
    expected: dict = {"measure": freeze_measure(run)}
    if args.figure_log:
        log = json.loads(args.figure_log.read_text(encoding="utf-8"))
        expected["figures"] = freeze_figures(run, log)
    pipelines = {}
    for item in args.pipeline:
        name, _, folder = item.partition("=")
        pipelines[name] = freeze_pipeline(Path(folder).resolve(), roots)
    expected["pipelines"] = pipelines
    extras = {}
    for item in args.extra:
        name, _, folder = item.partition("=")
        extras[name] = {"tables": freeze_tables(Path(folder).resolve())}
    expected["extra"] = extras
    if args.commands:
        expected["commands"] = json.loads(args.commands.read_text(encoding="utf-8"))
    if args.counts:
        expected["counts"] = json.loads(args.counts.read_text(encoding="utf-8"))
    if args.environment:
        expected["environment"] = json.loads(args.environment.read_text(encoding="utf-8"))
    if args.copy_tables:
        expected["measure"]["copied"] = copy_tables(run, args.out.parent / "run")

    # The whole record is scrubbed, samples included: a pipeline table that
    # carries an absolute path in a cell is still frozen by its hash, and its
    # sample rows are readable on another machine.
    expected = scrub(expected, roots)
    args.out.write_text(json.dumps(expected, indent=1, sort_keys=True, default=str) + "\n",
                        encoding="utf-8")
    print(f"{len(expected['measure']['tables'])} tables, "
          f"{len(expected.get('figures', {}))} figures, "
          f"{len(pipelines)} pipelines -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
