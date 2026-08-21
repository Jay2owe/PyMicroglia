#!/usr/bin/env python
"""Keep the /pymicroglia action catalogue honest against the live registry.

Division of labour, and the reason this exists:

* **This script owns the signatures.** Everything between the
  ``pymicroglia-auto-reference`` markers in the skill's
  ``references/action-catalog.md`` is rebuilt from the registry — names,
  summaries, mutating flags, parameter tables with units and defaults, and what
  each action binds to. Those facts drift the moment somebody adds a parameter,
  and a stale parameter table is worse than none: an agent will confidently pass
  an argument that no longer exists.
* **People own the teaching prose.** When to prefer one action over another, why
  a threshold sits where it does, which actions take six hours — none of that is
  derivable from a signature and none of it is touched here. Everything outside
  the markers survives byte for byte.

The rendering, the splice and the staleness diff all live in
``analysis_kit.reference_gen``; this file only says which registry, which
catalogue, and what to call itself in the banner.

There is no Codex mirror to keep in step: ``~/.codex/skills`` and
``~/.claude/skills`` are the same shared folder, so Codex reads these words
already.

Usage::

    python tools/update_pymicroglia_references.py           # rewrite, report
    python tools/update_pymicroglia_references.py --check   # exit 1 if stale
    python tools/update_pymicroglia_references.py --quiet   # silent on success
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SLUG = "pymicroglia"
GENERATOR = "tools/update_pymicroglia_references.py"

#: The skill is global: it lives in the shared folder both ~/.claude/skills and
#: ~/.codex/skills point at, so there is one copy and no mirror.
SKILL_DIR = Path.home() / ".claude" / "skills" / "pymicroglia"
REFERENCE_DIR = SKILL_DIR / "references"
CATALOG = REFERENCE_DIR / "action-catalog.md"


def _registry():
    """The package's own registry, told where its docs live.

    Imported through the installed package rather than by walking up from this
    file, so a checkout that is not the installed one cannot silently generate
    a catalogue for the wrong code.
    """
    from pymicroglia.registry import build_registry

    return build_registry(reference_dir=REFERENCE_DIR)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate the /pymicroglia action catalogue.")
    parser.add_argument("--check", action="store_true",
                        help="write nothing; exit 1 and say what drifted")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress output on success")
    args = parser.parse_args(argv)

    try:  # summaries carry unicode; never crash a Windows console
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:  # pragma: no cover - older or oddly wrapped streams
        pass

    from analysis_kit import reference_gen

    changes = reference_gen.plan_updates(
        _registry(), CATALOG, slug=SLUG, generator=GENERATOR,
        reference_dir=REFERENCE_DIR)
    stale = [change for change in changes if change.stale]

    if args.check:
        if not stale:
            if not args.quiet:
                print("pymicroglia reference docs are in sync.")
            return 0
        print(f"pymicroglia reference docs are stale ({len(stale)} file(s)). "
              f"Run: python {GENERATOR}", file=sys.stderr)
        for change in stale:
            print(reference_gen.diff_summary(change), file=sys.stderr)
        return 1

    written = reference_gen.apply_updates(changes)
    if not args.quiet:
        if written:
            print(f"updated {len(written)} file(s): "
                  + ", ".join(path.name for path in written))
        else:
            print("pymicroglia reference docs already in sync; nothing written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
