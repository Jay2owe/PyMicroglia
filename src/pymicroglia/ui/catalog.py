"""Checked scientific navigation for the PyMicroglia workbench.

The JSON assigns a scientific home; live descriptions and defaults always come
from the installed action registry.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MAP_FILE = Path(__file__).with_name("capability_map.json")


def load_map() -> dict[str, Any]:
    from ..visualisation.figures import load
    data = json.loads(MAP_FILE.read_text(encoding="utf-8"))
    figures = load()
    present = {(row['name'], row.get('view')) for row in data['inventory'] if row['kind'] == 'figure view'}
    pages = {page['key'] for page in data['pages']}
    for key, spec in figures.items():
        for view in spec.views:
            if (key, view.key) not in present:
                data['inventory'].append({'kind': 'figure view', 'name': key, 'view': view.key,
                    'title': spec.title, 'summary': spec.title, 'requires_saved_pipeline': spec.saved,
                    'home': 'micro/' + (spec.family if spec.family in pages else 'results and export'),
                    'style': spec.grammar})
    for row in data['inventory']:
        if row['kind'] == 'figure view':
            row['alias_of'] = figures[row['name']].alias_of
    return data


def validate_map(data: dict[str, Any] | None = None) -> list[str]:
    """Return precise gaps or repeated homes against the live registries."""
    from pymicroglia.catalogue import actions
    from pymicroglia.measure.modules import MODULE_NAMES
    from pymicroglia.pipelines import PIPELINE_NAMES
    from pymicroglia.visualisation.figures import load

    data = data or load_map()
    errors: list[str] = []
    pages = data.get("pages", [])
    order = [page.get("key") for page in pages]
    if order[:2] != ["rhythms", "tissue"]:
        errors.append("Signal and rhythms and Tissue and territory must lead the sidebar")
    if len(order) != len(set(order)):
        errors.append("a scientific page appears twice")
    rows = data.get("inventory", [])
    expected_actions = {item["name"] for item in actions()}
    figures = load()
    expected = {
        "figure view": {(name, view.key) for name, spec in figures.items() for view in spec.views},
        "action": {(name, "") for name in expected_actions - set(figures)},
        "pipeline": {(name, "") for name in PIPELINE_NAMES},
        "measurement module": {(name, "") for name in MODULE_NAMES},
    }
    for kind, wanted in expected.items():
        found = [(row.get("name"), row.get("view", "")) for row in rows if row.get("kind") == kind]
        missing, extra = wanted - set(found), set(found) - wanted
        repeated = {item for item in found if found.count(item) > 1}
        if missing or extra or repeated:
            errors.append(f"{kind}: missing={sorted(missing)}, extra={sorted(extra)}, repeated={sorted(repeated)}")
    for row in rows:
        home = row.get("home", "")
        if home.startswith("micro/") and home.split("/", 1)[1] not in order and home != "micro/results and export":
            errors.append(f"unknown page for {row.get('name')}: {home}")
    return errors
