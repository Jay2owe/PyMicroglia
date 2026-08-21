"""The action catalogue this package ships.

Thirteen actions and the parameters they take, generated once from the protocol
scripts they were copied from and then owned by this package. It is a data file
rather than a run-time read of those scripts, because PyMicroglia has to work
whether or not they are on the machine — and because the catalogue is ours to
rename, extend and correct without touching them.

Regenerate with ``python tools/regenerate_catalogue.py``.

One parameter name means one thing across the project, so the shared vocabulary
holds each name's canonical type, units and prose. Defaults are per action,
because the same name honestly carries different values in different protocols:
``crf`` is 18 in one video exporter and 20 in another, and flattening that would
be a lie. :func:`action_params` merges the two.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from .params import ParamDoc

__all__ = ["load", "actions", "action", "action_names", "param_docs", "action_params"]

DATA = Path(__file__).parent / "data" / "actions.json"


@lru_cache(maxsize=1)
def load() -> dict[str, Any]:
    """The whole catalogue, parsed once."""
    return json.loads(DATA.read_text(encoding="utf-8"))


def actions() -> list[dict[str, Any]]:
    return list(load()["actions"])


def action_names() -> list[str]:
    return sorted(entry["name"] for entry in load()["actions"])


def action(name: str) -> dict[str, Any] | None:
    return next((entry for entry in load()["actions"] if entry["name"] == name), None)


@lru_cache(maxsize=1)
def param_docs() -> tuple[ParamDoc, ...]:
    """The shared vocabulary, as ``ParamDoc`` entries.

    No defaults here on purpose — a shared entry that carried one action's
    default would misreport every other action that uses the name.
    """
    return tuple(
        ParamDoc(
            name=row["name"],
            type=row["type"],
            description=row["description"],
            required=bool(row.get("required", False)),
            default=None,
            units=row.get("units", "-"),
        )
        for row in load()["params"]
    )


def action_params(name: str) -> list[dict[str, Any]]:
    """Parameter rows for one action, each carrying *that action's* default."""
    entry = action(name)
    if entry is None:
        return []
    shared = {doc.name: doc for doc in param_docs()}
    rows = []
    for param in entry["params"]:
        doc = shared.get(param)
        row = doc.as_dict() if doc is not None else {
            "name": param, "type": "?", "units": "-", "required": False,
            "default": None, "description": "",
        }
        row["default"] = entry["defaults"].get(param)
        rows.append(row)
    return rows
