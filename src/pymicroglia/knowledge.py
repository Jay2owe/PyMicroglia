"""What an agent asks instead of reading source.

``describe`` and ``discover`` return live JSON from the installed package, so
nobody has to open a module to find out what it takes. ``doctor`` answers "is
this healthy?" in one call, and — the part that matters here — names the
interpreter it is running in, because "the audit silently did nothing" and "you
ran the wrong Python" look identical from the outside.

The kit does the work when it is installed. When it is not, small local
equivalents return the same shapes, so the commands answer either way.
"""

from __future__ import annotations

import sys
from typing import Any

from . import catalogue, config, recording
from ._optional import kit, kit_version
from .registry import (CLAIM_TEMPLATES, NEEDS_A_CLAIM, REGISTRY, pending,
                       pending_reason)

__all__ = ["describe", "discover", "doctor", "validate"]


def _live_params(name: str) -> list[dict[str, Any]]:
    """The action's parameters, with the defaults the function actually applies.

    The catalogue records what the protocol script this was copied from
    declared, and in a handful of places the package deliberately settled
    somewhere else — ``run_controls`` searches 16-32 h where the engine searched
    15-40, and one engine default is the *string* ``"3.0 / 8.0"``. ``describe``
    is what an agent reads before passing an argument, so it has to report what
    will happen, not what the protocol used to say.

    The catalogue is still the source for names, types, units and prose. Only
    the default is taken from the code, and only when the target resolves.
    """
    from .registry import live_choices, live_defaults

    live = live_defaults(name)
    choices = live_choices(name)
    rows = catalogue.action_params(name)
    for row in rows:
        if row["name"] in live:
            row["default"] = live[row["name"]]
        if row["name"] in choices:
            row["choices"] = choices[row["name"]]
    return rows


def _entry(name: str) -> dict[str, Any]:
    spec = catalogue.action(name) or {}
    return {
        "name": name,
        "summary": spec.get("summary", ""),
        "mutates": bool(spec.get("mutates")),
        "destructive": bool(spec.get("destructive")),
        "display_only": bool(spec.get("display_only")),
        "params": _live_params(name),
        "binds_to": spec.get("method", ""),
        # Asked of the recording layer rather than read off the catalogue: only
        # five actions inherited one from the protocol they were copied from,
        # and the rest declare it on the module that does the work.
        "method_version": recording._method_version(name),
        "pending": REGISTRY.resolve(name) is None,
        # Empty unless pending. For a seam it names the dotted target that
        # does not resolve, which is what to install.
        "pending_reason": pending_reason(spec.get("method", "")),
        # Whether a run of this needs a sentence saying what it was meant to
        # show. Reported here so an agent learns it from `describe` rather than
        # from a refusal after it has already assembled the arguments.
        "claim_required": name in NEEDS_A_CLAIM,
        "claim_template": CLAIM_TEMPLATES.get(name, ""),
    }


def describe(action: str | None = None) -> dict[str, Any]:
    """Every action and every documented argument, as plain JSON.

    Parameter rows carry the default that applies to *this* action. The kit's
    shared vocabulary cannot: one name may honestly have different defaults in
    different actions, so the default is merged in here from the catalogue.
    """
    from . import __version__

    if action is not None and action not in REGISTRY:
        return {
            "ok": False,
            "error": "unknown_action",
            "message": f"Unknown action: {action!r}.",
            "available": REGISTRY.names(),
        }

    header = {"ok": True, "project": "pymicroglia", "version": __version__}
    if action is None:
        return {**header, "actions": [_entry(name) for name in REGISTRY.names()]}
    return {**header, **_entry(action)}


def discover() -> dict[str, Any]:
    """Reconcile the registry against what is actually importable."""
    from . import __version__

    ak = kit()
    waiting = pending()
    if ak is not None:
        report = dict(ak.discover(REGISTRY))
    else:
        report = {
            "ok": True,
            "project": "pymicroglia",
            "version": __version__,
            "registered": REGISTRY.names(),
            "modules": {},
            "missing": waiting,
            "unregistered": [],
            "undocumented": [],
            "removed": [],
            "undeclared": REGISTRY.undeclared_params(),
            "reference_dir": None,
        }
    report["pending"] = waiting
    report["actions"] = len(REGISTRY.names())
    report["params"] = len(catalogue.param_docs())
    report["param_uses"] = sum(len(e["params"]) for e in catalogue.actions())
    return report


def _video_export_available() -> bool:
    """Whether an ffmpeg this package can drive is present.

    Reported here rather than discovered at the end of a six-hour pipeline,
    which is when the engines this was ported from used to find out.
    """
    try:
        from .video import encode

        return bool(encode.available())
    except ImportError:
        return False


def _dehydrated(root) -> int:
    """How many stored arrays the sync client has freed to online-only.

    Counted from the store's own listing rather than by walking the folder, so
    a store on a slow or absent drive costs one ``stat`` per entry and no
    directory scan of anything else.
    """
    from .store import tier_b

    try:
        return sum(1 for entry in tier_b.entries()
                   if config.is_placeholder(entry["path"]))
    except OSError:
        return 0


def _imagej() -> dict[str, Any]:
    """Whether Fiji was found, and where it was looked for.

    Reported so an absent bridge is checkable rather than silent — the same
    discipline the audit layer gets. Nothing measured needs it: it is the door
    to the one step a person does by hand.
    """
    from . import imagej

    try:
        return imagej.status()
    except Exception as exc:  # pragma: no cover - status is already guarded
        return {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}


def _workbench_version() -> str | None:
    """``circadian_workbench``'s version, or ``None`` if it is not importable.

    Asked here so ``doctor`` can say whether a rhythm can be tested at all,
    before somebody starts a six-hour run that ends at the periodogram.
    """
    try:
        import circadian_workbench
    except ImportError:
        return None
    return getattr(circadian_workbench, "__version__", "unknown")


def _workbench_api_version() -> str | None:
    """Friendly application programming interface version, when usable."""

    try:
        import circadian_workbench
    except ImportError:
        return None
    required = ("call", "trace", "population", "phases", "channels")
    if not all(callable(getattr(circadian_workbench, name, None)) for name in required):
        return None
    return str(getattr(circadian_workbench, "PUBLIC_API_VERSION", "unknown"))


def doctor() -> dict[str, Any]:
    """Is this control layer healthy, and which interpreter is answering?"""
    from . import __version__

    ak = kit()
    waiting = pending()
    complaints: list[str] = []

    if ak is None:
        complaints.append(
            "analysis_kit is not importable, so runs are not being recorded. "
            "Install it with: pip install -e <path to analysis-kit>"
        )

    undeclared = REGISTRY.undeclared_params()
    if undeclared:
        complaints.append(f"parameters used by an action but never documented: {undeclared}")

    root = config.store_root()
    synced = config.inside_dropbox(root)
    # Counted whatever the root looks like: the folder name is how Dropbox is
    # recognised, and OneDrive, a network share or a mapped drive can dehydrate
    # a file without the word "dropbox" appearing anywhere in the path.
    dehydrated = _dehydrated(root)
    if dehydrated:
        # Not "it is in Dropbox" — that is now on purpose. This is the state
        # that makes a synced store worse than no store: an array that is a hit
        # by every test the store can make, and a download when it is read.
        complaints.append(
            f"{dehydrated} stored array(s) under {root} have been freed to "
            "online-only by the sync client, so reading one is a download "
            "rather than a read. Right-click the folder in Dropbox and choose "
            "\"Make available offline\", or set PYMICROGLIA_STORE to a local path."
        )

    # Two optional dependencies, and they are not equivalent. The audit layer
    # above is soft: losing a run record must never break the science, so its
    # absence is a complaint. The workbench is hard: rhythm analysis *is* the
    # science, so its absence is reported here and raises when used, rather
    # than being quietly skipped.
    workbench = _workbench_version()
    workbench_api = _workbench_api_version()
    # A third case again, and softer than both. A missing Fiji costs the one
    # manual step and nothing else, so it is reported and never complained
    # about: an unattended run on a server has no Fiji by design.
    fiji = _imagej()

    return {
        # Pending actions are the expected state during the build-out, so they
        # are reported but do not make the layer unhealthy.
        "ok": not complaints,
        "project": "pymicroglia",
        "version": __version__,
        "interpreter": sys.executable,
        "analysis_kit": kit_version() or None,
        "circadian_workbench": workbench,
        "circadian_api_version": workbench_api,
        "video_export_available": _video_export_available(),
        "rhythm_analysis_available": workbench_api is not None,
        "imagej": fiji,
        "hand_roi_available": bool(fiji.get("ok")),
        "actions": len(REGISTRY.names()),
        "pending_actions": len(waiting),
        "pending": waiting,
        "store_root": str(root),
        "store_synced": synced,
        "store_dehydrated": dehydrated,
        "store_cap_gb": round(config.cache_cap_bytes() / 1024 ** 3, 1),
        "store_free_gb": round(config.free_bytes(root) / 1024 ** 3, 1),
        "complaints": complaints,
        "fix": REGISTRY.fix_hint if complaints else None,
    }


def validate(action: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Check an action name and its arguments before anything runs."""
    params = dict(params or {})
    if action not in REGISTRY:
        return {
            "ok": False,
            "error": "unknown_action",
            "message": f"Unknown action: {action!r}.",
            "available": REGISTRY.names(),
        }

    from .registry import check_parameters

    known = {row["name"] for row in catalogue.action_params(action)}
    unknown = sorted(set(params) - known)
    # What the code can see inside the arguments the catalogue only names:
    # an option under a nested group that no module declares, say.
    problems = check_parameters(action, params) if not unknown else []
    result: dict[str, Any] = {
        "ok": not unknown and not problems,
        "action": action,
        "unknown_params": unknown,
        "problems": problems,
        "pending": REGISTRY.resolve(action) is None,
    }
    if unknown:
        result["message"] = (
            f"{action} does not take {unknown}. Ask 'describe {action}' for what it does take."
        )
    elif problems:
        result["message"] = "; ".join(problems)
    if result["pending"]:
        why = pending_reason(REGISTRY.binds_to(action))
        result["note"] = (
            f"{action} is declared but not yet implemented; it binds to "
            f"{REGISTRY.binds_to(action)!r}, "
            + (f"and {why}" if why else "which does not exist yet.")
        )
    return result
