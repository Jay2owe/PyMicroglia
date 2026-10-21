import json
from pathlib import Path
from pymicroglia.ui.catalog import load_map
import pytest
from pymicroglia.ui.exploration_catalog import exploration_catalog, figure_route, protected_fingerprints

ROOT = Path(__file__).resolve().parents[1]


def test_every_capability_has_one_route():
    result = exploration_catalog()
    assert len(result["capabilities"]) == len(load_map()["inventory"])
    assert len({v["key"] for v in result["views"]}) == 9
    allowed = {v["key"] for v in result["views"]} | {"reports", "measure", "methods", "pipelines"}
    assert all(r["route"] in allowed for r in result["capabilities"])


def test_native_tissue_and_gantt_views_have_their_existing_visual_homes():
    from pymicroglia.figure_tables.tissue import PANEL_ORDER
    result = exploration_catalog()
    tissue = [r for r in result["capabilities"] if r["name"] == "tissue_tectonics" and r["kind"] == "figure view"]
    assert {r["view"] for r in tissue} == set(PANEL_ORDER)
    assert all(r["route"] == "tissue-map" for r in tissue)
    assert figure_route({"name": "cell_lifecycle_events", "view": "timelines"}) == "event-timeline"
    assert figure_route({"name": "cells_on_screen", "view": "lifespan"}) == "event-timeline"


def test_accepted_workspaces_remain_unchanged():
    baseline = json.loads((ROOT / "docs/workbench-layout/preserved-workspaces.json").read_text())
    assert protected_fingerprints(ROOT) == baseline


def test_new_figure_view_requires_an_explicit_visual_home():
    with pytest.raises(ValueError, match="Choose one visual home"):
        figure_route({"name": "new_figure", "view": "unmapped_view"})
