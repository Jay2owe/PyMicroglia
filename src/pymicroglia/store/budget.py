"""How large the rebuildable tier may grow, and what goes first when it does.

Tier B is pure convenience, so the only real question is how much of the one
drive this machine has it may occupy. The cap defaults to 64 GB — roughly two or
three working stacks — and is a configuration value, not a constant:
``PYMICROGLIA_CACHE_GB`` moves it.

Eviction is least-recently-used, ordered by a timestamp the store writes into
each sidecar on every read. Windows updates file access times lazily and can be
configured not to update them at all, so asking the filesystem which array was
used last gives an answer that is quietly wrong.

Nothing here touches tier A. A derived artefact is the analysis and is never
evicted; only pixels that rebuild from it are.
"""

from __future__ import annotations

from typing import Any

from .. import config
from . import tier_b

__all__ = ["usage", "over_by", "evict", "room_for", "report"]


def usage() -> int:
    """Bytes currently held by tier B."""
    return sum(entry["bytes"] for entry in tier_b.entries())


def over_by(cap: int | None = None) -> int:
    """How far over the cap the cache is; zero when it is under."""
    limit = config.cache_cap_bytes() if cap is None else int(cap)
    return max(0, usage() - limit)


def evict(cap: int | None = None, *, headroom: int = 0) -> dict[str, Any]:
    """Delete least-recently-used arrays until the cache fits.

    ``headroom`` reserves space for something about to be written, so a caller
    can make room before starting a long fill rather than discovering the cap
    half way through.

    Anything currently mapped is skipped, not waited for. Skipping costs a
    little disk until the next call; taking it would corrupt a live read.
    """
    limit = config.cache_cap_bytes() if cap is None else int(cap)
    target = max(0, limit - int(headroom))
    protected = tier_b.held()

    remaining = tier_b.entries()
    total = sum(entry["bytes"] for entry in remaining)
    before = total
    evicted: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for entry in sorted(remaining, key=lambda e: (e["last_read"], e["path"])):
        if total <= target:
            break
        if entry["path"] in protected:
            skipped.append(entry)
            continue
        if tier_b.forget(entry["stage"], entry["digest"]):
            total -= entry["bytes"]
            evicted.append(entry)
        else:
            skipped.append(entry)

    return {
        "cap_bytes": limit,
        "headroom_bytes": int(headroom),
        "before_bytes": before,
        "after_bytes": total,
        "freed_bytes": before - total,
        "evicted": [e["path"] for e in evicted],
        "held": [e["path"] for e in skipped],
        "over": max(0, total - limit),
    }


def room_for(nbytes: int, *, cap: int | None = None) -> dict[str, Any]:
    """Make room for an array of ``nbytes`` before writing it."""
    return evict(cap, headroom=int(nbytes))


def report() -> dict[str, Any]:
    """What the cache is holding, for ``doctor`` and for a person."""
    entries = tier_b.entries()
    total = sum(entry["bytes"] for entry in entries)
    cap = config.cache_cap_bytes()
    return {
        "root": str(config.store_root()),
        "arrays": len(entries),
        "used_gb": round(total / 1024 ** 3, 2),
        "cap_gb": round(cap / 1024 ** 3, 2),
        "free_gb": round(config.free_bytes() / 1024 ** 3, 1),
        "over_cap": max(0, total - cap) > 0,
        "held": len(tier_b.held()),
    }
