"""Browse cell measurements without computing or changing saved observations.

The catalogue owns browsing groups and recommended choices. Available values
and their units still come from the selected recording's cell-frame table.
Missing choices are suggestions, never synthetic values or runnable inputs.
"""
from __future__ import annotations

GROUPS = (
    ("signal", "Signal"),
    ("shape", "Size & shape"),
    ("branches", "Branches"),
    ("movement", "Movement & change"),
    ("territory", "Territory"),
    ("neighbours", "Neighbours & contacts"),
)

# Each recommendation has exactly one browsing home and a point-of-use hint.
RECOMMENDED = (
    ("signal", "integrated_density", "Total light", "intensity", "Signal summed inside the mask."),
    ("signal", "expanded_integrated_density", "Expanded-outline total light", "expanded_intensity", "Signal summed inside exclusive expanded outlines."),
    ("signal", "corrected_expanded_integrated_density", "Expanded total minus background", "expanded_intensity", "Expanded-outline total after subtracting its cell-free local background."),
    ("signal", "signal_mean", "Mean light", "intensity", "Signal averaged inside the mask."),
    ("signal", "corrected_integrated_density", "Background-corrected total light", "intensity", "Total light after background correction."),
    ("signal", "corrected_mean", "Background-corrected mean light", "intensity", "Mean light after background correction."),
    ("signal", "punctate_fraction", "Punctate fraction", "intensity", "Fraction of mask pixels classified as punctate."),
    ("shape", "area_px", "Area", "morphology", "Number of pixels inside the cell mask."),
    ("shape", "perimeter_px", "Perimeter", "morphology", "Length around the cell outline."),
    ("shape", "equivalent_diameter_px", "Equivalent diameter", "morphology", "Diameter of a circle with the same area."),
    ("shape", "circularity", "Circularity", "morphology", "How close the outline is to a circle."),
    ("shape", "solidity", "Solidity", "morphology", "Cell area divided by its convex enclosing area."),
    ("shape", "aspect_ratio", "Elongation", "morphology", "Major axis divided by minor axis."),
    ("shape", "ramification_index", "Outline complexity", "morphology", "Perimeter relative to that expected for the cell area."),
    ("branches", "branch_count", "Branch count", "morphology", "Process branches after contracting junction clusters."),
    ("branches", "total_branch_px", "Total branch length", "morphology", "Sum of process branch lengths."),
    ("branches", "longest_branch_px", "Longest branch", "morphology", "Length of the longest process branch."),
    ("branches", "skeleton_endpoints", "Number of tips", "morphology", "Endpoints of the process skeleton."),
    ("branches", "sholl_max_intersections_scale_cell", "Maximum radial crossings", "sholl", "Maximum outline crossings across cell-scaled radial rings."),
    ("branches", "sholl_critical_radius_scale_cell", "Radius of maximum crossings", "sholl", "Radius with the most crossings, using cell-scaled rings."),
    ("movement", "speed", "Speed", "motility", "Centroid movement per saved time interval."),
    ("movement", "soma_step_px_gapless", "Cell-body movement", "motility", "Body-centre displacement between consecutive observed frames; gaps stay missing."),
    ("movement", "displacement_from_first_px", "Displacement from start", "motility", "Distance from the first observed centroid."),
    ("movement", "turnover_index", "Footprint turnover", "surveillance", "Mask pixels gained and lost relative to their union."),
    ("movement", "gained_px", "Gained area", "surveillance", "Pixels newly inside the mask since the preceding frame."),
    ("movement", "lost_px", "Lost area", "surveillance", "Pixels no longer inside the mask since the preceding frame."),
    ("movement", "area_change_px", "Net area change", "surveillance", "Gained pixels minus lost pixels."),
    ("movement", "extension_bias", "Extension bias", "surveillance", "Balance of extension and retraction."),
    ("movement", "turn_angle_rad", "Turning angle", "walk", "Turn relative to the preceding measured step."),
    ("territory", "cumulative_unique_px", "Territory covered so far", "territory", "Unique pixels covered up to this frame."),
    ("territory", "new_px", "Newly explored area", "territory", "Pixels visited for the first time in this frame."),
    ("territory", "revisit_fraction", "Revisited fraction", "territory", "Share of the current footprint visited previously."),
    ("neighbours", "nearest_edge_px", "Nearest outline distance", "neighbours", "Minimum distance to another cell's outline."),
    ("neighbours", "nearest_neighbour_px", "Nearest centre distance", "neighbours", "Distance to the nearest other cell centroid."),
    ("neighbours", "neighbours_within_radius", "Nearby cell count", "neighbours", "Cells within the radius recorded by the measurement run."),
    ("neighbours", "local_density", "Local density", "neighbours", "Recorded local cell density."),
)

_RECOMMENDATIONS = {row[1]: row for row in RECOMMENDED}
_BRANCH_KEYS = {"skeleton_px", "skeleton_junctions", "skeleton_branches", "tip_branch_count", "max_branch_order", "skeleton_length_um"}
_MODULE_GROUP = {"intensity": "signal", "channels": "signal", "morphology": "shape", "sholl": "branches",
                 "motility": "movement", "surveillance": "movement", "motion_evidence": "movement", "walk": "movement",
                 "territory": "territory", "territory_shape": "territory", "neighbours": "neighbours", "contacts": "neighbours",
                 "objects": "neighbours", "object_geometry": "neighbours"}


def _group(metric):
    key = metric["key"]
    if key in _RECOMMENDATIONS:
        return _RECOMMENDATIONS[key][0]
    if key.startswith(("branch_", "sholl_")) or "branch" in key or key in _BRANCH_KEYS:
        return "branches"
    if key.startswith(("signal_", "original_integrated", "expanded_", "expansion_", "corrected_", "background_", "punctate", "integrated_density", "bright_fraction")) or key == "dff":
        return "signal"
    if key in {"area_um2", "perimeter_um", "equivalent_diameter_um"}:
        return "shape"
    # Already-registered ownership is metadata; do not import analysis engines
    # merely to display a measurement menu.
    from pymicroglia.measure.declare import list_modules, list_derived
    for module in (*list_modules(), *list_derived()):
        if any(column.name == key for column in module.produces):
            return _MODULE_GROUP.get(module.name, "other")
    return {"morphology": "shape", "motility": "movement", "intensity": "signal", "territory": "territory"}.get(metric.get("role"), "other")


def measurement_catalogue(metrics):
    """Return six agreed groups, all saved metrics and missing recommendations.

    Availability means the existing reader found numeric cell-frame values.
    Never offer a missing value as an input to a plot or rhythm calculation.
    """
    rows = []
    available = {metric["key"] for metric in metrics}
    # Calibrated variants satisfy a recommendation without a duplicate warning.
    variants = {"area_px": "area_um2", "perimeter_px": "perimeter_um", "equivalent_diameter_px": "equivalent_diameter_um"}
    for metric in metrics:
        recommendation = _RECOMMENDATIONS.get(metric["key"])
        rows.append({**metric, "group": _group(metric), "available": True,
                     "help": recommendation[4] if recommendation else "Saved cell-frame measurement.", "status": "Available"})
    for group, key, label, module, help_text in RECOMMENDED:
        if key not in available and variants.get(key) not in available:
            rows.append({"key": key, "label": label, "unit": "", "group": group,
                         "available": False, "module": module, "status": "Not measured yet",
                         "help": f"{help_text} Requires saved {module} cell-frame measurements."})
    groups = [{"key": key, "label": label} for key, label in GROUPS]
    if any(row["group"] == "other" for row in rows):
        groups.append({"key": "other", "label": "Other saved measurements"})
    return {"version": 1, "groups": groups, "metrics": rows}
