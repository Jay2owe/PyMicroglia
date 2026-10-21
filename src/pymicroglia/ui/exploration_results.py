"""Adapt saved analysis rows; exploration never fits or classifies a cell."""
from .exploration_data import number, read_rows, table_path


def _table(root, movie, name, limit=10_000):
    path = table_path(root, movie, name)
    if path is None:
        return [], None, "Not saved"
    try:
        rows = read_rows(path)
        return rows[:limit], str(path), f"Showing {min(len(rows), limit)} / {len(rows)} saved rows"
    except (ValueError, OSError) as exc:
        return [], str(path), str(exc)


def saved_results(root, movie):
    rows, path, message = _table(root, movie, "rhythms")
    rhythms = []
    for row in rows:
        identity = number(row.get("identity"))
        period = number(row.get("best_period_hours"))
        phase = number(row.get("best_phase_fraction"))
        hours = number(row.get("best_phase_hours"))
        if phase is None and hours is not None and period is not None and period > 0:
            phase = (hours / period) % 1
        if identity is None or identity <= 0 or period is None or period <= 0:
            continue
        rhythms.append({"identity": int(identity), "metric": row.get("metric"), "period_hours": period,
                        "phase_hours": hours, "phase_fraction": phase if phase is not None and 0 <= phase <= 1 else None,
                        "status": row.get("rhythm_status", "unknown"),
                        "method": row.get("period_estimation_method", "saved estimator")})
    rhythm_payload = {"rows": rhythms, "metrics": sorted({r["metric"] for r in rhythms if r["metric"]}), "source": path,
                      "message": message if rows else "No saved rhythm phases. Choose a rhythm pipeline to calculate them.",
                      "meaning": "Saved peak phase as a fraction of each cell's fitted period; fit availability is separate from rhythmicity."}
    from .exploration_rhythms import load_attached_rhythms
    rhythm_payload = load_attached_rhythms(root, movie) or rhythm_payload
    events, event_sources = [], []
    for name in ("lifecycle_events", "regime_transitions"):
        rows, path, message = _table(root, movie, name)
        if path:
            event_sources.append({"table": name, "path": path, "message": message})
        for row in rows:
            identity = number(row.get("identity"))
            frame = number(row.get("frame_index", row.get("to_frame_index")))
            if identity is None or frame is None or frame < 0:
                continue
            label = row.get("event") or (str(row.get("from_regime", "")) + " → " + str(row.get("to_regime", "")))
            status = row.get("event_status")
            events.append({"identity": int(identity), "frame_index": int(frame), "label": label + (f" ({status})" if status else ""),
                           "source": "states" if name == "regime_transitions" else "lifecycle", "table": name})
    relations = []
    for name in ("contacts_frame", "contacts"):
        rows, path, message = _table(root, movie, name)
        by_dilation = {}
        for row in rows:
            a, b = number(row.get("identity_a")), number(row.get("identity_b"))
            if a is None or b is None or a <= 0 or b <= 0:
                continue
            frame = number(row.get("frame_index"))
            link = {"identity_a": int(a), "identity_b": int(b), "frame_index": None if frame is None else int(frame),
                    "hours_in_contact": number(row.get("hours_in_contact")), "dilation_px": number(row.get("dilation_px")),
                    "shared_boundary_px": number(row.get("shared_boundary_px"))}
            by_dilation.setdefault(row.get("dilation_px", "saved"), []).append(link)
        for dilation, links in by_dilation.items():
            relations.append({"key": f"{name}:{dilation}", "title": ("Contacts at selected frame" if name == "contacts_frame" else "Contacts over recording") + f" · dilation {dilation} px",
                              "rows": links, "source": path, "message": message,
                              "meaning": "Saved shared-boundary contacts under the recorded dilation; no proximity or correlation is inferred by the browser."})
    return {"rhythms": rhythm_payload, "events": {"rows": events, "sources": event_sources, "message": "Only saved lifecycle/state events and observation gaps are shown."},
            "relations": {"sources": relations, "message": "No saved contacts. Enable the contacts measurement module to produce these links."}}
