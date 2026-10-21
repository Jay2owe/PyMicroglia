"""Read-only cell cohort expressions with explicit missing-evidence semantics.

Example::

    rule = {"op": "AND", "children": [
        {"field": "rhythmic", "metric": "signal_mean", "cmp": "eq", "value": True},
        {"field": "missing_pct", "cmp": "lte", "value": 10},
        {"op": "NOT", "children": [{"field": "has_merge", "cmp": "eq", "value": True}]}]}
    result = evaluate(cells, rule, mode="keep", unknown="keep")

``cells`` maps original identities to facts. No mask, identity or result is edited.
"""
from __future__ import annotations

import math

SCHEMA_VERSION = 1
NUMERIC_COMPARATORS = {"lt": "<", "lte": "≤", "eq": "=", "neq": "≠", "gte": "≥", "gt": ">", "between": "between (inclusive)", "not_between": "outside (exclusive)"}
CATEGORY_COMPARATORS = {"eq": "is", "neq": "is not"}


def _field(key, label, group, kind="number", *, help, metric=False, values=None):
    return {"key": key, "label": label, "group": group, "kind": kind, "help": help,
            "metric": metric, "values": values,
            "comparators": NUMERIC_COMPARATORS if kind == "number" else CATEGORY_COMPARATORS}


FIELDS = {f["key"]: f for f in (
    _field("review_passed", "Imported decision passed", "Review decisions", "bool", help="Imported per-cell pass/fail evidence, such as fixed-circle period agreement. Missing decisions remain unknown; importing does not rerun the underlying test."),
    _field("identity", "Cell identity", "Observations", help="The saved cell identity; filtering never renumbers it."),
    _field("observed_frames", "Observed frames", "Observations", help="Exact named mask observations, independent of display smoothing."),
    _field("missing_frames", "Missing frames: whole recording", "Observations", help="Includes frames before the first and after the last observation."),
    _field("missing_pct", "Missing frames: whole recording (%)", "Observations", help="100 × missing frames / all recording frames."),
    _field("internal_missing_pct", "Missing frames: within lifespan (%)", "Observations", help="100 × internal missing frames / frames from first to last observed mask."),
    _field("longest_gap_frames", "Longest internal gap (frames)", "Observations", help="Longest consecutive absence between first and last observation. Excludes leading and trailing absences."),
    _field("longest_gap_hours", "Longest internal gap (h)", "Observations", help="Missing frames × the verified regular sampling interval. Unavailable for an irregular or unknown clock."),
    _field("gap_count", "Number of internal gaps", "Observations", help="Number of separate consecutive internal absences."),
    _field("leading_missing_frames", "Frames before first observation", "Observations", help="Native recording frames preceding the first observed mask."),
    _field("trailing_missing_frames", "Frames after last observation", "Observations", help="Native recording frames following the last observed mask."),
    _field("span_hours", "First-to-last observation span (h)", "Observations", help="Elapsed time between first and last observed mask."),
    _field("rhythmic", "Significant rhythm", "Rhythms", "bool", metric=True, help="Saved verdict for the chosen measurement. Untested or insufficient evidence is unknown, not non-rhythmic."),
    _field("rhythm_tested", "Rhythm verdict available", "Rhythms", "bool", metric=True, help="A saved positive or negative rhythm verdict exists for this measurement. Choose No to explicitly filter untested or insufficient cells."),
    _field("period_hours", "Saved rhythm period (h)", "Rhythms", metric=True, help="Saved estimated period for this measurement; a fitted period does not imply significance."),
    _field("rhythm_cycles", "Cycles covered by rhythm test", "Rhythms", metric=True, help="Saved cycles-observed value, when exported by the rhythm engine."),
    _field("rhythm_observations", "Observations used by rhythm test", "Rhythms", metric=True, help="Saved finite observations used by the rhythm test; may differ from named masks or display samples."),
    _field("rhythm_p", "Rhythm p value", "Rhythms", metric=True, help="Saved test p value, when exported by the selected engine."),
    _field("rhythm_q", "Rhythm adjusted p value", "Rhythms", metric=True, help="Saved multiple-test adjusted p value; no correction is recalculated by filtering."),
    _field("has_merge", "Any recorded merge / residency", "Events", "bool", help="Recorded merge or shared-host residency, including both participants. Lifecycle absorption candidates also count; this is saved evidence, not a new biological classification."),
    _field("has_split", "Any recorded split / division", "Events", "bool", help="Recorded split, division or candidate birth with a parent; includes both participants."),
    _field("has_contact_separation", "Any tracker contact separation", "Events", "bool", help="Native tracker event contact_separate. This does not classify a biological division."),
    _field("has_birth", "Any candidate birth", "Events", "bool", help="Native lifecycle event 'born'. Candidate status is retained; an appearance alone does not count."),
    _field("has_death", "Any candidate death", "Events", "bool", help="Native lifecycle event 'died'. Candidate status is retained; a disappearance alone does not count."),
    _field("has_entry", "Entered the field", "Events", "bool", help="Native lifecycle event 'entered_field'."),
    _field("has_exit", "Left the field", "Events", "bool", help="Native lifecycle event 'left_field'."),
    _field("has_candidate_event", "Any candidate lifecycle event", "Events", "bool", help="At least one native lifecycle event explicitly marked candidate."),
    _field("event_count", "Lifecycle event count", "Events", help="Saved lifecycle events excluding censored present-at-start/end markers."),
    _field("min_distance_px", "Closest approach: centroids (px)", "Neighbours", help="Minimum same-frame centroid distance to any other original cell across all exact observations. No interpolation or self comparison; neighbours are measured before filtering."),
    _field("min_distance_um", "Closest approach: centroids (µm)", "Neighbours", help="Same exact centroid distance converted using the saved positive microns-per-pixel calibration. Unavailable without that calibration."),
    _field("metric_mean", "Measurement mean", "Measurements", metric=True, help="Mean of finite saved values in observed frames. Display temporal smoothing is not applied."),
    _field("metric_min", "Measurement minimum", "Measurements", metric=True, help="Minimum finite saved value."),
    _field("metric_max", "Measurement maximum", "Measurements", metric=True, help="Maximum finite saved value."),
    _field("metric_missing_pct", "Measurement missing (%)", "Measurements", metric=True, help="Percentage of all recording frames without a finite value, including absent masks and unavailable measurements."),
)}


def validate(expression, *, metrics=None):
    """Reject malformed expressions before touching any data."""
    count = 0
    def visit(node, depth=0):
        nonlocal count
        count += 1
        if count > 200 or depth > 12:
            raise ValueError("Use at most 200 filter nodes and 12 nested groups")
        if not isinstance(node, dict):
            raise ValueError("Each filter must be a rule or logical group")
        if "op" in node:
            if set(node) - {"op", "children"}:
                raise ValueError("Unknown logical-group parameter")
            op, children = node["op"], node.get("children")
            if op not in {"AND", "OR", "NOT", "XOR"} or not isinstance(children, list):
                raise ValueError("Choose AND, OR, NOT or XOR and a list of children")
            if op == "NOT" and len(children) != 1:
                raise ValueError("NOT takes exactly one rule or group")
            if not children and depth:
                raise ValueError("Nested groups need at least one rule")
            for child in children:
                visit(child, depth + 1)
            return
        if set(node) - {"field", "cmp", "value", "metric"}:
            raise ValueError("Unknown filter-rule parameter")
        field = FIELDS.get(node.get("field"))
        if field is None or node.get("cmp") not in field["comparators"]:
            raise ValueError("Choose a supported filter field and comparison")
        value = node.get("value")
        if field["kind"] == "bool":
            if not isinstance(value, bool):
                raise ValueError("Boolean filters require true or false")
        else:
            values = value if node["cmp"] in {"between", "not_between"} else [value]
            if not isinstance(values, list) or len(values) != (2 if node["cmp"] in {"between", "not_between"} else 1):
                raise ValueError("A range requires two bounds")
            if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in values):
                raise ValueError("Filter thresholds must be finite numbers")
            if len(values) == 2 and values[0] > values[1]:
                raise ValueError("The lower range bound must not exceed the upper bound")
        if field["metric"]:
            if not isinstance(node.get("metric"), str) or not node["metric"]:
                raise ValueError("Choose a measurement for this filter")
            if metrics is not None and node["metric"] not in metrics:
                raise ValueError("Filter measurement is not available: " + node["metric"])
        elif "metric" in node:
            raise ValueError("This filter does not take a measurement")
    visit(expression)
    return expression


def rules(expression):
    if "op" in expression:
        return [r for child in expression["children"] for r in rules(child)]
    return [expression]


def _compare(actual, cmp, target):
    if actual is None:
        return None
    if cmp == "eq": return actual == target
    if cmp == "neq": return actual != target
    if cmp == "lt": return actual < target
    if cmp == "lte": return actual <= target
    if cmp == "gt": return actual > target
    if cmp == "gte": return actual >= target
    inside = target[0] <= actual <= target[1]
    return inside if cmp == "between" else not inside


def _match(facts, node, leaves):
    if "op" not in node:
        key = node["field"] + (":" + node["metric"] if "metric" in node else "")
        actual = facts.get(key)
        answer = _compare(actual, node["cmp"], node["value"])
        leaves.append({"rule": node, "actual": None if actual == math.inf else actual,
                       "no_neighbour": actual == math.inf, "match": answer})
        return answer
    values = [_match(facts, c, leaves) for c in node["children"]]
    op = node["op"]
    if op == "NOT": return None if values[0] is None else not values[0]
    if op == "AND": return False if False in values else None if None in values else True
    if op == "OR": return True if True in values else None if None in values else False
    return None if None in values else bool(sum(values) % 2)


def evaluate(cells, expression, *, mode="exclude", unknown="keep"):
    """Return selected original identities and per-rule evidence, without edits.

    AND: a known false wins. OR: a known true wins. NOT keeps unknown unknown.
    XOR: an odd number of true rules. Unknown policy is applied only afterwards.
    """
    validate(expression)
    if mode not in {"keep", "exclude"} or unknown not in {"keep", "exclude", "error"}:
        raise ValueError("Choose keep/exclude mode and keep/exclude/error for unknown evidence")
    rows, included = [], []
    for identity, facts in sorted(cells.items()):
        evidence = []
        match = _match(facts, expression, evidence)
        if match is None and unknown == "error":
            raise ValueError(f"Cell {identity} has unknown evidence for this expression")
        keep = unknown == "keep" if match is None else match if mode == "keep" else not match
        if keep: included.append(identity)
        rows.append({"identity": identity, "match": match, "included": keep, "evidence": evidence})
    return {"schema_version": SCHEMA_VERSION, "expression": expression, "mode": mode, "unknown": unknown,
            "identities": included, "total": len(cells), "included": len(included),
            "excluded": len(cells) - len(included), "unknown_count": sum(r["match"] is None for r in rows), "rows": rows}
