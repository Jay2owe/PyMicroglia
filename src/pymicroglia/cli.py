r"""The ``pymicroglia`` command line.

A thin shim: it parses arguments and prints JSON, and every answer it gives
comes from the registry or the kit. Nothing here knows anything the package does
not, which is what stops the command line and the API drifting apart.

    pymicroglia doctor
    pymicroglia discover
    pymicroglia describe remove_cosmic_rays
    pymicroglia validate remove_cosmic_rays seed_z=8
    pymicroglia run remove_cosmic_rays source=stack.tif --claim "spike check"
    pymicroglia runs --limit 10 --contains MCG_04
    pymicroglia result 20260820-141233-a1b2c3 --script > redo.py
    pymicroglia notebook 20260820-141233-a1b2c3 --request "the day-4 dip"
    pymicroglia note "the day-4 dip is the drug arriving" --run 20260820-141233-a1b2c3
    pymicroglia scan "D:\...\AI_Exports\registered_2026-07-23"
    pymicroglia store --evict
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from typing import Any

from . import knowledge
from ._optional import kit

__all__ = ["main"]


def _emit(payload: Any) -> int:
    print(json.dumps(payload, indent=2, default=str))
    return 0 if (not isinstance(payload, dict) or payload.get("ok", True)) else 1


def _pairs(items: list[str]) -> dict[str, Any]:
    """``seed_z=8`` -> ``{"seed_z": 8}``.

    Values are read as Python literals so numbers, tuples and booleans survive;
    anything that is not a literal stays the string it was typed as, which is
    what a path needs.
    """
    out: dict[str, Any] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"expected key=value, got {item!r}")
        try:
            out[key.strip()] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            out[key.strip()] = value
    return out


def build_parser() -> argparse.ArgumentParser:
    # Raw, so the usage examples in the module docstring stay one per line
    # instead of being reflowed into a single paragraph.
    parser = argparse.ArgumentParser(
        prog="pymicroglia", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="is the control layer healthy, and which Python is this")
    sub.add_parser("discover", help="reconcile the registry against what is importable")

    describe = sub.add_parser("describe", help="what an action does and what it takes")
    describe.add_argument("action", nargs="?")

    validate = sub.add_parser("validate", help="check an action and its arguments")
    validate.add_argument("action")
    validate.add_argument("params", nargs="*", metavar="key=value")

    run = sub.add_parser("run", help="run an action")
    run.add_argument("action")
    run.add_argument("params", nargs="*", metavar="key=value")
    run.add_argument("--claim", default="", help="one sentence: what this run is meant to show")
    run.add_argument("--out", default=None, help="output folder")
    run.add_argument("--notebook", action="store_true",
                     help="append this run to the project's reproducibility "
                          "notebook as well as recording it")
    run.add_argument("--request", default="",
                     help="what you asked for, in your words. The one thing in "
                          "a record that cannot be reconstructed later")

    scan = sub.add_parser(
        "scan", help="index one results folder's stored artefacts")
    scan.add_argument("folders", nargs="+", metavar="FOLDER",
                      help="a results folder. Not recursive, by rule: a "
                           "recursive search over a synced path hydrates "
                           "online-only files by the gigabyte")

    store_cmd = sub.add_parser("store", help="what the artefact store holds")
    store_cmd.add_argument("--evict", action="store_true",
                           help="bring the rebuildable tier back under its cap")

    verify = sub.add_parser("verify", help="hash a source file in full, once")
    verify.add_argument("source")
    verify.add_argument("--full", action="store_true",
                        help="read the whole file. Slow on purpose: one stack "
                             "is a 10.8 GB read")

    runs = sub.add_parser(
        "runs", help="recent runs from the global index: 'have we done this "
                     "before?', answered by reading one file")
    runs.add_argument("--limit", type=int, default=20,
                      help="0 for every row")
    runs.add_argument("--action", default=None, help="only this action")
    runs.add_argument("--since", default=None,
                      help="ISO date or timestamp; rows at or after it")
    runs.add_argument("--contains", default=None,
                      help="substring of the action, the claim or the project. "
                           "A recording's name is in the claim, so this finds "
                           "every run that touched it")
    runs.add_argument("--all-projects", action="store_true",
                      help="include runs from other projects sharing the index")
    runs.add_argument("--table", action="store_true",
                      help="one line per run instead of JSON")

    result = sub.add_parser("result", help="one run's full record")
    result.add_argument("run_id")
    result.add_argument("--script", action="store_true",
                        help="print only the equivalent script, so it can be "
                             "redirected into a file and run")

    notebook = sub.add_parser(
        "notebook", help="append a recorded run to the reproducibility notebook")
    notebook.add_argument("run_id")
    notebook.add_argument("--request", default="",
                          help="what you asked for, in your words")
    notebook.add_argument("--out", default=None,
                          help="results folder to write the notebook under. "
                               "Defaults to the folder the record sits in")

    note = sub.add_parser("note", help="append an interpretation to the lab notebook")
    note.add_argument("text")
    note.add_argument("--run", dest="run_ids", action="append", default=[])
    note.add_argument("--store", default=None)

    return parser


def _ran(action: str, done: dict[str, Any]) -> dict[str, Any]:
    """What a person needs after a run: did it work, what was written, and the
    id to look it up by.

    Not the action's return value. That is a cleaned series holding its own
    cosmic-ray mask, and printing one is twenty-six million booleans where a
    report should be. The record has it, summarised, and ``result <run_id>``
    hands the whole thing back.
    """
    record = done.get("record") or {}
    payload: dict[str, Any] = {
        "ok": True,
        "action": action,
        "run_id": record.get("run_id"),
        "recorded": done.get("recorded", False),
        "outputs": record.get("outputs") or [],
        "artefacts": [f"{item.get('stage')} {item.get('digest')}"
                      for item in record.get("artefacts") or []],
        "claim": record.get("claim", ""),
        "seconds": record.get("duration_s"),
        "result": record.get("result"),
    }
    if done.get("notebook"):
        payload["notebook"] = done["notebook"]
    if not payload["recorded"]:
        payload["note"] = (
            "the analysis ran; the record did not. 'pymicroglia doctor' says "
            "whether the audit layer is present.")
    return payload


def _results_root(record_path: Any) -> str | None:
    """The results folder a stored record belongs to.

    A record lives at ``<results>/.analysis-kit/records/<id>.json``, so the
    folder it describes is three levels up. Derived rather than stored, because
    a results folder copied onto another machine must still find its own
    notebook.
    """
    if not record_path:
        return None
    from pathlib import Path

    path = Path(str(record_path))
    return str(path.parent.parent.parent) if len(path.parents) >= 3 else None


def _needs_kit(what: str) -> dict[str, Any]:
    return {
        "ok": False,
        "error": "kit_missing",
        "message": (
            f"{what} reads the audit index, which analysis_kit provides, and it is "
            "not importable. Analysis itself is unaffected - run 'pymicroglia "
            "doctor' to see which interpreter is in use."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = args.command

    if command == "doctor":
        return _emit(knowledge.doctor())
    if command == "discover":
        return _emit(knowledge.discover())
    if command == "describe":
        return _emit(knowledge.describe(args.action))
    if command == "validate":
        return _emit(knowledge.validate(args.action, _pairs(args.params)))

    if command == "run":
        from .registry import ClaimRequired
        from .run import ActionInvalid, ActionPending, run_recorded

        roots = [args.out] if args.out else []
        try:
            # Passed, never set in the environment: an environment variable
            # set here would outlive this call and make every later run in
            # the process claim it came from the command line too.
            done = run_recorded(args.action, claim=args.claim, entry="cli",
                                output_roots=roots, notebook=args.notebook,
                                request=args.request, **_pairs(args.params))
        except ActionPending as exc:
            return _emit({"ok": False, "error": "pending", "message": str(exc)})
        except ClaimRequired as exc:
            return _emit({"ok": False, "error": "claim_required",
                          "message": str(exc)})
        except ActionInvalid as exc:
            return _emit({"ok": False, "error": "invalid", "message": str(exc)})
        return _emit(_ran(args.action, done))

    if command in {"scan", "store", "verify"}:
        from . import store as artefacts

        if command == "scan":
            reports = [artefacts.scan(folder) for folder in args.folders]
            return _emit({"ok": all(r.get("ok") for r in reports),
                          "scanned": reports})
        if command == "store":
            payload = artefacts.status()
            if args.evict:
                payload["evicted"] = artefacts.budget.evict()
            return _emit({"ok": True, **payload})
        source = (artefacts.verify_source(args.source) if args.full
                  else artefacts.fingerprint(args.source))
        return _emit({"ok": True, "source": source.as_dict()})

    ak = kit()
    if command == "runs":
        if ak is None:
            return _emit(_needs_kit("runs"))
        rows = ak.audit.runs(
            limit=args.limit,
            project=None if args.all_projects else "pymicroglia",
            action=args.action, since=args.since, contains=args.contains)
        if args.table:
            print(ak.audit.index.summarise(rows))
            return 0
        return _emit({"ok": True, "runs": rows, "count": len(rows)})

    if command == "result":
        if ak is None:
            return _emit(_needs_kit("result"))
        found = ak.audit.result(args.run_id)
        if not found.get("ok"):
            # Reported at the top level, like every other failure this command
            # line can return, rather than nested where a caller has to know to
            # look for it.
            return _emit({"ok": False, "error": "not_found",
                          "message": found.get("error", ""),
                          "run_id": args.run_id})
        if not args.script:
            return _emit({"ok": True, "result": found})
        # Printed bare, not as JSON: the point of --script is that the output is
        # a runnable file, and a JSON-quoted one would not be.
        print(str(found["record"].get("script") or "").rstrip("\n"))
        return 0

    if command == "notebook":
        if ak is None:
            return _emit(_needs_kit("notebook"))
        from .recording import append_to_notebook

        found = ak.audit.result(args.run_id)
        if not found.get("ok"):
            return _emit({"ok": False, "error": "not_found",
                          "message": found.get("error", "")})
        root = args.out or _results_root(found["row"].get("record"))
        if root is None:
            return _emit({"ok": False, "error": "no_root",
                          "message": "the record does not say which results "
                                     "folder it belongs to; pass --out"})
        path = append_to_notebook(found["record"], root=root,
                                  request=args.request)
        return _emit({"ok": True, "notebook": str(path), "run_id": args.run_id})

    if command == "note":
        if ak is None:
            return _emit(_needs_kit("note"))
        from . import config

        path = ak.audit.note_run(
            args.text,
            store_root=args.store or config.store_root(),
            run_ids=args.run_ids,
        )
        return _emit({"ok": True, "notebook": str(path)})

    return _emit({"ok": False, "error": "unknown_command", "message": command})


if __name__ == "__main__":
    sys.exit(main())
