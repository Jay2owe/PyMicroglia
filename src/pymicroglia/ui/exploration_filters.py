"""Exact saved evidence for the shared cell filter; never launches an analysis."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import math
from statistics import mean

import numpy as np

from pymicroglia.cell_filters import FIELDS, evaluate, rules, validate
from .exploration_data import _observations, number, read_rows, recording, table_path


def filter_catalogue(run, stem):
    root, saved, movie = recording(run, stem)
    _, _, _, metrics = _observations(root, movie)
    return {"schema_version": 1, "fields": list(FIELDS.values()), "metrics": metrics,
            "microns_per_pixel": number(saved.get("settings", {}).get("microns_per_pixel")),
            "groups": {"AND": "All rules must match", "OR": "Any rule may match",
                       "NOT": "Invert a rule or group; unknown stays unknown",
                       "XOR": "An odd number of rules must match"}}


def _presence(rows, frames):
    indices = sorted(rows)
    first, last, n = indices[0], indices[-1], len(frames)
    gaps = [b-a-1 for a, b in zip(indices, indices[1:]) if b-a > 1]
    longest = max(gaps, default=0)
    clock = [f["hours"] for f in frames]
    delta = np.diff(np.asarray(clock, dtype=float))
    regular = len(delta) and np.isfinite(delta).all() and np.allclose(delta, delta[0]) and delta[0] > 0
    return {"observed_frames": len(indices), "missing_frames": n-len(indices),
            "missing_pct": 100*(n-len(indices))/n,
            "internal_missing_pct": 100*sum(gaps)/(last-first+1),
            "longest_gap_frames": longest, "longest_gap_hours": 0 if longest == 0 else float(longest*delta[0]) if regular else None,
            "gap_count": len(gaps), "leading_missing_frames": first,
            "trailing_missing_frames": n-last-1,
            "span_hours": clock[last]-clock[first] if clock[last] is not None and clock[first] is not None else None}


def _event_facts(root, movie, observations, frames):
    from .exploration_history import load_history
    lifecycle = table_path(root, movie, "lifecycle_events")
    events = read_rows(lifecycle)
    history, sources, message = load_history(root, movie, observations, frames)
    tracker = table_path(root, movie, "history_merge_split_events")
    tracking = read_rows(tracker)
    # Absence only means "no recorded evidence" if its table was actually saved.
    lifecycle_known = lifecycle is not None
    merge_known = tracker is not None or bool(sources) and "history_residency" in history
    facts = {identity: {"has_merge": False if merge_known or lifecycle_known else None,
                        "has_split": False if tracker or lifecycle_known else None,
                        "has_contact_separation": False if tracker else None,
                        **{key: False if lifecycle_known else None for key in
                           ("has_birth", "has_death", "has_entry", "has_exit", "has_candidate_event")},
                        "event_count": 0 if lifecycle_known else None} for identity in observations}
    def flag(identities, key):
        for identity in identities:
            if identity in facts: facts[identity][key] = True
    for row in events:
        identity, partner = number(row.get("identity")), number(row.get("partner_identity"))
        ids = [identity, partner]
        event = row.get("event")
        if identity in facts:
            if event not in {"present_at_start", "present_at_end"}: facts[identity]["event_count"] += 1
            if row.get("event_status") == "candidate": facts[identity]["has_candidate_event"] = True
        key = {"born": "has_birth", "died": "has_death", "entered_field": "has_entry", "left_field": "has_exit"}.get(event)
        if key: flag([identity], key)
        if event == "absorbed": flag(ids, "has_merge")
        if event == "born" and partner: flag(ids, "has_split")
    for row in history.get("history_residency", []):
        if number(row.get("host_identity")):
            flag([number(row.get("identity")), number(row["host_identity"])], "has_merge")
    for row in tracking:
        # Native history table schema; a merge/residency is distinct from contact.
        event = str(row.get("event_type", row.get("event", row.get("kind", "")))).lower()
        ids = [number(row.get(k)) for k in ("identity", "partner_identity", "host_identity", "identity_a", "identity_b")]
        if row.get("candidate_identities"):
            from types import SimpleNamespace
            from pymicroglia.figure_tables.territory_values import _event_identities
            ids.extend(_event_identities(SimpleNamespace(**row)))
        if "merge" in event or "resident" in event: flag(ids, "has_merge")
        if "split" in event or "division" in event: flag(ids, "has_split")
        if event == "contact_separate": flag(ids, "has_contact_separation")
    return facts, [p for p in (lifecycle, tracker) if p], sources, message


@lru_cache(maxsize=2)
def _mask_positions(path, size, modified):
    from .exploration_context import _labels
    labels = _labels(path, size, modified)
    positions = []
    for image in labels:
        y, x = np.nonzero(image)
        ids, inv = np.unique(image[y, x], return_inverse=True)
        counts = np.bincount(inv)
        xs, ys = np.bincount(inv, weights=x), np.bincount(inv, weights=y)
        positions.append({int(i): (float(xs[j]/counts[j]), float(ys[j]/counts[j])) for j, i in enumerate(ids)})
    return positions


def _distances(root, movie, observations, frames, microns):
    from scipy.spatial import cKDTree
    from .exploration_images import declared_source, signature
    path = declared_source(root, movie, "labels")
    if path:
        positions = _mask_positions(*signature(path))
        if len(positions) != len(frames):
            raise ValueError("Distance masks do not align with the saved recording clock")
        for f, positions_at_frame in enumerate(positions):
            named = {i for i, rows in observations.items() if f in rows}
            if set(positions_at_frame) != named:
                raise ValueError("Distance masks do not match saved per-frame identity membership")
        source = str(path)
    else:
        # Native morphology/motility preserve centroid_x/y in pixels even on
        # calibrated runs; physical distances are converted only after comparison.
        positions = [{} for _ in frames]
        for identity, rows in observations.items():
            for f, row in rows.items():
                x, y = number(row.get("centroid_x")), number(row.get("centroid_y"))
                if x is not None and y is not None: positions[f][identity] = (x, y)
        source = "saved cell_frame centroids"
    minimum, complete = {i: math.inf for i in observations}, {i: True for i in observations}
    for f, points in enumerate(positions):
        named = {i for i, rows in observations.items() if f in rows}
        if set(points) != named:
            # An absent neighbour coordinate could hide a closer encounter for any cell.
            for i in named: complete[i] = False
        if len(points) < 2: continue
        ids = list(points)
        distances, _ = cKDTree(list(points.values())).query(list(points.values()), k=2)
        for identity, distance in zip(ids, distances[:, 1]): minimum[identity] = min(minimum[identity], float(distance))
    return minimum, complete, source


def preview_filters(run, stem, expression, *, mode="exclude", unknown="keep", analysis_ids=None, decisions=None):
    """Evaluate a reproducible cohort against full native frames, without writes."""
    root, saved, movie = recording(run, stem)
    path, observations, frames, metrics = _observations(root, movie)
    metric_keys = {m["key"] for m in metrics}
    validate(expression, metrics=metric_keys)
    leaves = rules(expression)
    if analysis_ids is not None and (not isinstance(analysis_ids, dict) or set(analysis_ids) - metric_keys):
        raise ValueError("Saved analysis selections must map available measurements to analysis identifiers.")
    if len(leaves)*len(observations) > 100_000:
        raise ValueError("This filter exceeds the 100,000 cell/rule interactive evidence budget")
    requested = {r["field"] for r in leaves}
    sources = {"observations": str(path)}
    facts = {i: {"identity": i, **_presence(rows, frames)} for i, rows in observations.items()}
    from ..cell_display import decision_values
    imported = decision_values(decisions, stem)
    if "review_passed" in requested:
        for identity in facts:
            facts[identity]["review_passed"] = imported.get(identity)
        sources["imported_decisions"] = {key: decisions.get(key) for key in ("source_name", "source_sha256")} if decisions else None
    event_fields = {k for k in FIELDS if FIELDS[k]["group"] == "Events"}
    if requested & event_fields:
        events, paths, history_sources, message = _event_facts(root, movie, observations, frames)
        for i in facts: facts[i].update(events[i])
        sources.update({"event_tables": list(map(str, paths)), "tracking_history": history_sources, "history_message": message})
    rhythm_fields = {k for k in FIELDS if FIELDS[k]["group"] == "Rhythms"}
    if requested & rhythm_fields:
        from .exploration_results import saved_results
        from .exploration_rhythms import load_attached_rhythms
        if analysis_ids is None:
            rhythms = saved_results(root, movie)["rhythms"]
            selected = {metric: rhythms for metric in metric_keys}
        else:
            loaded = {identifier: load_attached_rhythms(root, movie, identifier) or {}
                      for identifier in set(analysis_ids.values()) if identifier}
            selected = {metric: loaded.get(analysis_ids.get(metric), {}) for metric in metric_keys}
        verdicts = {(r["identity"], metric): r for metric, rhythms in selected.items()
                    if not rhythms.get("preview") for r in rhythms.get("rows", []) if r["metric"] == metric}
        sources["rhythms"] = {metric: {k: rhythms.get(k) for k in ("source", "analysis_id", "resolved_options")}
                              for metric, rhythms in selected.items()}
        for rule in leaves:
            if rule["field"] not in rhythm_fields: continue
            metric = rule["metric"]
            for i in facts:
                row = verdicts.get((i, metric), {})
                status = row.get("status")
                rhythmic = True if status == "rhythmic" else False if status in {"non_rhythmic", "nonrhythmic", "not_rhythmic", "not rhythmic"} else None
                value = {"rhythmic": rhythmic, "rhythm_tested": rhythmic is not None,
                         "period_hours": number(row.get("period_hours")),
                         "rhythm_cycles": number(row.get("cycles_observed")),
                         "rhythm_observations": number(row.get("observations")),
                         "rhythm_p": number(row.get("p_value", row.get("p"))),
                         "rhythm_q": number(row.get("q_value", row.get("q")))}[rule["field"]]
                facts[i][rule["field"]+":"+metric] = value
    for metric in {r["metric"] for r in leaves if FIELDS[r["field"]]["group"] == "Measurements"}:
        for i, rows in observations.items():
            values = [v for row in rows.values() if (v := number(row.get(metric))) is not None]
            facts[i].update({"metric_mean:"+metric: mean(values) if values else None,
                             "metric_min:"+metric: min(values) if values else None,
                             "metric_max:"+metric: max(values) if values else None,
                             "metric_missing_pct:"+metric: 100*(len(frames)-len(values))/len(frames)})
    if requested & {"min_distance_px", "min_distance_um"}:
        microns = number(saved.get("settings", {}).get("microns_per_pixel"))
        distance, complete, source = _distances(root, movie, observations, frames, microns)
        sources["distance"] = {"source": source, "microns_per_pixel": microns, "definition": "exact same-frame mask centroids; original full cohort"}
        for i in facts:
            facts[i]["min_distance_px"] = distance[i] if complete[i] else None
            facts[i]["min_distance_um"] = distance[i]*microns if complete[i] and microns and microns > 0 else None
    result = evaluate(facts, expression, mode=mode, unknown=unknown)
    from .exploration_images import declared_source
    raw = declared_source(root, movie, "raw")
    labels = declared_source(root, movie, "labels")
    result["render_sources"] = {"source": str(raw) if raw else "", "labels": str(labels) if labels else "",
                                "source_frame_offset": int(movie.get("spec", {}).get("source_frame_offset", 0))}
    clock = np.diff(np.asarray([frame["hours"] for frame in frames], dtype=float))
    if len(clock) and np.isfinite(clock).all() and clock[0] > 0 and np.allclose(clock, clock[0]):
        result["render_sources"]["frame_interval_h"] = float(clock[0])
    from .exploration_data import _cell, overview_indices, PREVIEW_CELLS
    _, indices = overview_indices(len(frames))
    result["overview_cells"] = [_cell(i, observations[i], indices, metrics, movie) for i in result["identities"][:PREVIEW_CELLS]]
    from .exploration_history import _fingerprint
    result.update({"run": str(root), "stem": stem, "sources": sources,
                   "observations_sha256": _fingerprint(path), "frame_count": len(frames), "exact": True,
                   "analysis_ids": analysis_ids, "decisions": decisions,
                   "scope": "Display cohort; saved data and rhythm test populations are unchanged"})
    result["evidence_sha256"] = hashlib.sha256(json.dumps(result["rows"], sort_keys=True, allow_nan=False).encode()).hexdigest()
    result["equivalent_script"] = ("import json\nfrom pymicroglia.ui.exploration_filters import preview_filters\n"
                                   f"result = preview_filters({str(root)!r}, {stem!r}, {expression!r}, mode={mode!r}, unknown={unknown!r}, analysis_ids={analysis_ids!r}, decisions={decisions!r})\n"
                                   f"assert result['observations_sha256'] == {result['observations_sha256']!r}\n"
                                   f"assert result['evidence_sha256'] == {result['evidence_sha256']!r}, 'Saved filter evidence has changed'\n"
                                   "print(json.dumps(result, ensure_ascii=False))\n")
    return result
