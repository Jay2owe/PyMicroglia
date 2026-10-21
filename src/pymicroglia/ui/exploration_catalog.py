"""Single visual homes; existing scientific actions remain the export interface."""
from __future__ import annotations

import hashlib
from pathlib import Path
from .catalog import load_map

VIEWS = (
    ("image", "Cell image", "follow"),
    ("outline-history", "Outline history", "follow"),
    ("time-series", "Measurement trace", "follow"),
    ("matrix", "Recording barcode", "follow"),
    ("event-timeline", "Track timeline / Gantt", "follow"),
    ("phase-clock", "Phase clock", "follow"),
    ("tissue-map", "Tissue tectonics / map", "tissue"),
    ("relationships", "Cell relationships", "tissue"),
    ("distribution", "Cell and group comparison", "compare"),
)
GROUPS = [
    {"key": "follow", "title": "Follow a cell"},
    {"key": "tissue", "title": "Tissue & neighbours"},
    {"key": "compare", "title": "Compare cells & groups"},
    {"key": "events", "title": "Events", "page": True},
    {"key": "rhythm-audit", "title": "Rhythm audit", "page": True},
]
# Routing concerns the saved view's visual grammar, never its biological topic.
GRAMMARS = {
    "image": {"images", "tiles", "input", "diagnostics"},
    "outline-history": {"outlines"},
    "time-series": {"analyses", "traces", "raw", "detrended", "native", "processed", "wall_clock", "realigned", "carried", "curves", "breath", "small_multiples", "triggered", "profile", "coefficients", "lags", "rate", "rates", "coverage", "accumulation", "original", "anchored"},
    "matrix": {"matrix", "barcode", "kymograph", "wall", "period_peak_matrix", "offset", "within-cell", "within-sample", "across-samples", "method_grid", "similarity", "membership"},
    "event-timeline": {"raster", "timelines", "intervals", "observations", "ribbon", "history", "spans"},
    "phase-clock": {"dial", "rose"},
    "tissue-map": {"map", "field", "positions", "first_coverage", "cumulative_occupancy", "unique_cells", "speed", "significant_period", "splitting_events", "phase"},
    "relationships": {"connections", "tree", "families", "flow", "relationships"},
    "distribution": {"histogram", "period_histogram", "scatter", "distribution", "cells", "samples", "contrasts", "effects", "paired", "spread", "ranking", "strength", "stability", "periods", "period", "amplitude", "amplitudes", "duration", "missingness", "alpha", "dwell", "changes", "controls", "comparison", "distance", "pairs"},
}

# Composite figures have their own configuration in the action dialog. Keep
# this list explicit so a newly registered visual view cannot disappear into
# the reports page without somebody choosing its home.
REPORT_FIGURES = frozenset("""
accepted_state_card active_span audit_decisions audit_disagreement audit_performance
audit_periods breakout_triggered_average cell_lifecycle_events
cell_lifecycle_summary cell_report_card cell_state_time_summaries cell_trace_card
cells_on_screen change_ledger conservation_ledger contact_ledger contrast_forest
independence_map intervention_cell_report intervention_response_overview
intervention_response_patterns intervention_response_timing measurement_lag_profiles
measurement_relationship_overview negative_space neighbour_coordination
null_channel_phase_test patch_ledger phase_compass pixel_fate_flow
predictability_clock recurrence_wall regime_programmes regime_reporter_bridge
regime_ribbon regime_transitions reporter_shape_timing rhythm_agreement
rhythm_cell_report rhythm_pair_report rhythm_screen rhythm_screen_summary
rhythm_time_matrix rhythmicity_above_noise second_verse
spatial_coordination_connections spatial_coordination_overview
spatial_coordination_pair_card spatial_coordination_samples state_histories
state_profiles state_support_summary surveillance_not_translocation
territory_anchoring
""".split())


def figure_route(item: dict) -> str:
    if item['name'] == 'cell_group_comparison':
        return 'distribution'
    if item["name"] == "cells_on_screen" and item.get("view") == "lifespan":
        return "event-timeline"
    if item["name"] == "cell_lifecycle_events" and item.get("view") == "timelines":
        return "event-timeline"
    if item["name"] in REPORT_FIGURES:
        return "reports"
    matches = [key for key, names in GRAMMARS.items() if item.get("view") in names]
    if len(matches) != 1:
        raise ValueError(f"Choose one visual home for {item['name']}.{item.get('view')}: {matches}")
    return matches[0]


def exploration_catalog() -> dict:
    source = load_map()
    routes = []
    for item in source["inventory"]:
        route = "methods"
        if item["kind"] == "pipeline":
            route = "pipelines"
        elif item["kind"] == "measurement module":
            route = "measure"
        elif item["kind"] == "figure view":
            route = figure_route(item)
        routes.append({**item, "route": route})
    return {"version": 1, "groups": GROUPS,
            "views": [{"key": k, "title": t, "group": g} for k, t, g in VIEWS],
            "capabilities": routes}


def protected_fingerprints(root: Path) -> dict[str, str]:
    """Pin only the accepted workspaces; extras may evolve independently."""
    ui = root / "src/pymicroglia/ui"
    html = (ui / "static/index.html").read_text(encoding="utf-8")
    js = (ui / "static/app.js").read_text(encoding="utf-8")
    server = (ui / "server.py").read_text(encoding="utf-8")
    def block(text, start, end):
        return text[text.index(start):text.index(end, text.index(start))]
    values = {"styles": (ui / "static/styles.css").read_bytes(),
              "desktop": (ui / "desktop.py").read_bytes(),
              "openAuto": block(js, "  function openAuto()", "  function renderNav()"),
              "motion_handlers": block(js, "  async function openMotion()", "  async function initialize()"),
              "motion_proxy": block(server, '    @app.api_route("/api/motion/', "    if STATIC.is_dir()")}
    for identifier, tag in [("motion-sidebar", "aside"), ("auto-workspace", "section"), ("motion-workspace", "section")]:
        start = html.index(f'<{tag} id="{identifier}"')
        end = html.index(f"</{tag}>", start) + len(f"</{tag}>")
        values[identifier] = html[start:end]
    return {k: hashlib.sha256(v if isinstance(v, bytes) else v.encode()).hexdigest() for k, v in values.items()}


if __name__ == "__main__":
    import csv
    import json
    root = Path(__file__).resolve().parents[3]
    folder = root / "docs/workbench-layout"
    (folder / "preserved-workspaces.json").write_text(json.dumps(protected_fingerprints(root), indent=2))
    rows = exploration_catalog()["capabilities"]
    with (folder / "capability-to-view.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
