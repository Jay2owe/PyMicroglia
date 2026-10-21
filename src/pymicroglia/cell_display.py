"""Display-only cell decisions, shared by outlined images and movies."""
from __future__ import annotations

import csv
import hashlib
from io import StringIO
import json
from pathlib import Path
import re

DEFAULT_GREY = "#aaaaaa"


def rgb(value):
    """An explicit RGB triplet or six-digit hexadecimal colour."""
    if isinstance(value, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", value):
        return tuple(int(value[i:i + 2], 16) for i in (1, 3, 5))
    if (isinstance(value, (list, tuple)) and len(value) == 3
            and all(type(v) is int and 0 <= v <= 255 for v in value)):
        return tuple(value)
    raise ValueError("filter_grey_colour must be #RRGGBB or three integer RGB values")


def import_decisions(text, stem, *, filename="decisions.csv"):
    """Read a recording-specific pass/fail CSV without inventing missing decisions."""
    if not isinstance(text, str) or len(text.encode("utf-8")) > 4 * 1024**2:
        raise ValueError("Decision CSV must contain at most 4 MiB of text")
    reader = csv.DictReader(StringIO(text.lstrip("\ufeff")))
    names = reader.fieldnames or []
    column = next((key for key in ("passed", "passes_fixed_circle") if key in names), None)
    if "identity" not in names or not column:
        raise ValueError("Decision CSV needs identity and passed (or passes_fixed_circle) columns")
    values = {}
    for row in reader:
        if "recording" in names and row["recording"] != stem:
            continue
        try:
            identity = int(row["identity"])
        except (TypeError, ValueError):
            raise ValueError("Decision identities must be positive whole numbers") from None
        if identity <= 0 or str(identity) in values:
            raise ValueError("Decision identities must be positive and unique within a recording")
        value = (row.get(column) or "").strip().lower()
        if value not in {"true", "false", "1", "0"}:
            raise ValueError(f"Cell {identity}: decision must be true or false")
        values[str(identity)] = value in {"true", "1"}
    if not values:
        raise ValueError(f"No decisions found for recording {stem}")
    return {"schema_version": 1, "stem": stem, "values": values,
            "source_name": Path(filename).name,
            "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}


def decision_values(decisions, stem):
    """Validate portable imported evidence and its recording binding."""
    if decisions is None:
        return {}
    if not isinstance(decisions, dict) or decisions.get("schema_version") != 1:
        raise ValueError("Imported decisions require schema_version 1")
    if decisions.get("stem") != stem:
        raise ValueError("Imported decisions belong to a different recording")
    values = decisions.get("values")
    if not isinstance(values, dict):
        raise ValueError("Imported decisions need an identity-to-boolean values mapping")
    result = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key.isdecimal() or int(key) <= 0 or type(value) is not bool:
            raise ValueError("Imported decisions require positive identity keys and boolean values")
        if int(key) in result:
            raise ValueError("Imported decision identities must be unique")
        result[int(key)] = value
    return result


def resolve(spec, identities, *, grey_colour=DEFAULT_GREY):
    """Resolve a frozen selection or saved-evidence recipe to colour overrides.

    Returns colour overrides plus a portable receipt. A failed decision changes
    ink only: all identities and all photon pixels remain available.
    """
    colour = rgb(grey_colour)
    if spec is None:
        return {}, None
    if isinstance(spec, (str, Path)):
        spec = json.loads(Path(spec).read_text(encoding="utf-8-sig"))
    if not isinstance(spec, dict):
        raise ValueError("cell_filter must be a saved filter JSON or a filter recipe")
    if "rows" not in spec:
        from .ui.exploration_filters import preview_filters
        required = {"run", "stem", "expression"}
        allowed = required | {"mode", "unknown", "analysis_ids", "decisions", "grey_colour", "schema_version"}
        if not required <= spec.keys() or spec.keys() - allowed:
            raise ValueError("A cell-filter recipe needs run, stem, expression and known filter options")
        spec = preview_filters(spec["run"], spec["stem"], spec["expression"],
                               **{key: spec[key] for key in ("mode", "unknown", "analysis_ids", "decisions") if key in spec})
    if spec.get("schema_version") != 1 or not isinstance(spec.get("rows"), list):
        raise ValueError("Saved cell filter requires schema_version 1 and decision rows")
    decisions = {}
    for row in spec["rows"]:
        if not isinstance(row, dict) or type(row.get("identity")) is not int or row["identity"] <= 0 or type(row.get("included")) is not bool:
            raise ValueError("Saved filter rows require positive original identities and boolean included values")
        if row["identity"] in decisions:
            raise ValueError("Saved filter contains duplicate identities")
        decisions[row["identity"]] = row["included"]
    # A frozen receipt must not silently acquire modified decisions.
    digest = hashlib.sha256(json.dumps(spec["rows"], sort_keys=True, allow_nan=False).encode()).hexdigest()
    if spec.get("evidence_sha256") and digest != spec["evidence_sha256"]:
        raise ValueError("Saved filter evidence has changed")
    named = sorted({int(i) for i in identities if int(i) > 0})
    missing = set(named) - decisions.keys()
    if missing:
        raise ValueError(f"Cell filter has no decision for rendered identities: {sorted(missing)}")
    grey = [i for i in named if not decisions[i]]
    receipt = {"schema_version": 1, "display_only": True, "rendered_identities": named,
               "grey_identities": grey, "grey_colour": list(colour),
               "evidence_sha256": digest,
               **{key: spec[key] for key in ("expression", "mode", "unknown", "run", "stem",
                                             "observations_sha256", "analysis_ids", "decisions") if key in spec}}
    return {i: colour for i in grey}, receipt
