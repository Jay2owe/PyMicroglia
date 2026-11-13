"""Every package function a speed pass observed, with its cost and a disposition.

    python tools/speed_pass_function_map.py PROFILE.prof [PROFILE.prof ...] \
        --decisions decisions.json --out function_map.json

Reads cProfile captures from ``tools/benchmark_drop_and_run.py --profile`` and
keeps the functions defined in PyMicroglia or Auto-Organotypic (any source copy:
the live trees or a snapshot). Each gets the largest cumulative and own time it
had in any capture, and a disposition:

- the one ``decisions.json`` gives it (``{"module:function": {"disposition":
  ..., "reason": ...}}``, keys matched as suffixes of ``path:function``), or
- ``already cheap`` when its cumulative time stayed under ``--cheap`` of its
  capture's total.

Anything else is ``investigate`` and makes the command exit 1, so a map with an
undecided costly function cannot be handed in as complete. Functions the
captures never reached are not listed: the map says what was observed, not
what exists.
"""
from __future__ import annotations

import argparse
import json
import pstats
import re
import sys
from pathlib import Path

PACKAGES = ("pymicroglia", "auto_organotypic")
DISPOSITIONS = {"improved", "already cheap", "required work", "residual", "owned elsewhere",
                "unobserved", "investigate"}


def _module(filename: str) -> str | None:
    parts = re.split(r"[\\/]", filename)
    for n in range(len(parts) - 1, -1, -1):
        if parts[n] in PACKAGES:
            return "/".join(parts[n:])
    return None


def observed(profiles: list[Path]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for profile in profiles:
        stats = pstats.Stats(str(profile)).stats
        total = max(cum for (_, _, _, cum, _) in stats.values())
        for (filename, line, name), (_, calls, own, cum, _) in stats.items():
            module = _module(filename)
            if module is None:
                continue
            key = f"{module}:{name}"
            row = out.setdefault(key, {"function": key, "line": line, "calls": 0,
                                       "cumulative_s": 0.0, "own_s": 0.0, "share": 0.0,
                                       "captures": []})
            row["calls"] = max(row["calls"], calls)
            row["cumulative_s"] = max(row["cumulative_s"], round(cum, 4))
            row["own_s"] = max(row["own_s"], round(own, 4))
            row["share"] = max(row["share"], round(cum / total, 5))
            row["captures"].append(profile.stem)
    return out


def decide(rows: dict[str, dict], decisions: dict[str, dict], cheap: float) -> list[dict]:
    for key, row in rows.items():
        given = next((value for pattern, value in decisions.items()
                      if key.endswith(pattern)), None)
        if given is not None:
            if given.get("disposition") not in DISPOSITIONS or not given.get("reason"):
                raise SystemExit(f"{key}: a decision needs a known disposition and a reason")
            row.update(disposition=given["disposition"], reason=given["reason"])
        elif row["share"] < cheap:
            row.update(disposition="already cheap",
                       reason=f"under {cheap:.0%} of every capture it appeared in")
        else:
            row.update(disposition="investigate", reason="no decision recorded")
    return sorted(rows.values(), key=lambda row: -row["cumulative_s"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("profiles", nargs="+", type=Path)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cheap", type=float, default=0.01,
                        help="share of a capture below which a function is cheap (default 1%%)")
    args = parser.parse_args()
    decisions = json.loads(args.decisions.read_text(encoding="utf-8"))
    rows = decide(observed(args.profiles), decisions, args.cheap)
    open_rows = [row["function"] for row in rows if row["disposition"] == "investigate"]
    args.out.write_text(json.dumps({"captures": [p.name for p in args.profiles],
                                    "cheap_share": args.cheap, "open": open_rows,
                                    "functions": rows}, indent=1), encoding="utf-8")
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["disposition"]] = counts.get(row["disposition"], 0) + 1
    print(json.dumps({"functions": len(rows), "dispositions": counts, "open": open_rows}, indent=1))
    return 1 if open_rows else 0


if __name__ == "__main__":
    sys.exit(main())
