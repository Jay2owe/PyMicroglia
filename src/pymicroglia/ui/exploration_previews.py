"""Read-only, cached previews for runs without saved phase/contact results.

The owning analysis APIs do the calculation. No tracks, measurements or run
files are changed, and a phase preview never declares a cell rhythmic.
"""
from functools import lru_cache
import json

import numpy as np

from .exploration_data import (_observations, number, recording,
                               PREVIEW_CELLS)
from .exploration_results import saved_results


def _module_settings(saved, movie, name):
    settings = dict(saved.get("settings", {}).get("modules", {}).get(name, {}))
    for entry in movie.get("modules", []):
        if entry.get("module") == name:
            settings.update(entry.get("parameters", {}))
    return settings


@lru_cache(maxsize=12)
def _phases(root, movie_json, size, modified, metric, period):
    from pathlib import Path
    from pymicroglia import workbench

    movie = json.loads(movie_json)
    _, observations, frames, metrics = _observations(Path(root), movie)
    if metric not in {m["key"] for m in metrics}:
        raise ValueError("The phase metric is not available in this recording.")
    rows, omitted = [], 0
    for identity in sorted(observations)[:PREVIEW_CELLS]:
        pairs = [(frames[f]["hours"], number(row.get(metric)))
                 for f, row in observations[identity].items()]
        pairs = [(t, v) for t, v in pairs if t is not None and v is not None]
        pairs.sort()
        # Preserve real time and missing observations; never fill a gap with 0.
        if len(pairs) < 6 or pairs[-1][0] - pairs[0][0] < period or np.ptp([v for _, v in pairs]) == 0:
            omitted += 1
            continue
        time, values = np.asarray(pairs, dtype=float).T
        fit = workbench.cosine_fit_batch(time, values[None], period)
        phase = number(fit["phase_h"][0])
        amplitude = number(fit["amplitude"][0])
        if phase is None or amplitude is None or amplitude <= 0:
            omitted += 1
            continue
        rows.append({"identity": identity, "metric": metric,
                     "period_hours": period, "phase_hours": phase,
                     "phase_fraction": (phase / period) % 1,
                     "status": "unknown", "method": "fixed-period cosinor",
                     "r_squared": number(fit["r_squared"][0])})
    return {"rows": rows, "metrics": [metric], "preview": True,
            "source": "Circadian Workbench fixed-period cosinor",
            "message": "" if rows else "No varying traces span a full preview cycle.",
            "meaning": f"{period:g} h fixed-cycle phase preview; not a period estimate or rhythmicity test.",
            "omitted_cells": omitted, "period_hours": period,
            "workbench_version": workbench.WORKBENCH_VERSION}


def load_phase_preview(run, stem, metric=None):
    root, saved, movie = recording(run, stem)
    existing = saved_results(root, movie)["rhythms"]
    if existing["rows"]:
        return existing
    path, _, _, metrics = _observations(root, movie)
    keys = {m["key"] for m in metrics}
    if metric is None:
        metric = next((k for k in ("integrated_density_mean3", "integrated_density", "signal_mean") if k in keys), metrics[0]["key"])
    from pymicroglia import workbench
    settings = _module_settings(saved, movie, "rhythms")
    period = float(settings.get("fixed_period_hours", workbench.PERIOD_ANALYSIS_DEFAULTS["fixed_period_hours"]))
    if not np.isfinite(period) or period <= 0:
        raise ValueError("The preview cycle must be a finite positive duration.")
    stat = path.stat()
    return _phases(str(root), json.dumps(movie, sort_keys=True), stat.st_size,
                   stat.st_mtime_ns, metric, period)


@lru_cache(maxsize=32)
def _contacts(path, size, modified, frame, settings_json):
    from .exploration_images import _plane
    from pymicroglia.measure.context import MeasurementContext, Scale
    from pymicroglia.measure.modules.contacts import measure

    labels = _plane(path, size, modified, frame)
    if labels.dtype.kind not in "iu" or np.any(labels < 0):
        raise ValueError("Saved masks must contain nonnegative integer identities.")
    identities = [int(i) for i in np.unique(labels) if i]
    if len(identities) > 1000:
        raise ValueError("This frame exceeds the 1,000-cell contact preview budget.")
    # Contacts use labels only; do not read raw image pixels for this preview.
    context = MeasurementContext(stem="preview", labels=labels[None], raw=labels[None],
                                 scale=Scale(1), identities=identities,
                                 params={"contacts": json.loads(settings_json)})
    rows = measure(context)["contacts_frame"].to_dict("records")
    return rows, identities


def load_contact_preview(run, stem, frame):
    root, saved, movie = recording(run, stem)
    existing = saved_results(root, movie)["relations"]
    if existing["sources"]:
        return existing
    _, _, frames, _ = _observations(root, movie)
    if not 0 <= frame < len(frames):
        raise ValueError("Frame is outside the saved recording.")
    from .exploration_images import declared_source, signature
    from pymicroglia.measure.modules.contacts import DEFAULTS
    settings = {**DEFAULTS, **_module_settings(saved, movie, "contacts")}
    settings["include_unclaimed"] = False  # only the declared identity masks
    rows, identities = _contacts(*signature(declared_source(root, movie, "labels")),
                                frame, json.dumps(settings, sort_keys=True))
    configured = settings["dilation_px"]
    dilations = configured if isinstance(configured, (list, tuple)) else [configured]
    sources = []
    for dilation in dilations:
        sources.append({"key": f"preview_contacts:{dilation}",
                        "title": f"Current-frame contacts · dilation {dilation} px",
                        "preview": True, "frame_index": frame, "identities": identities,
                        "rows": [{**row, "frame_index": frame, "hours": frames[frame]["hours"]}
                                 for row in rows if row["dilation_px"] == dilation],
                        "source": str(declared_source(root, movie, "labels")),
                        "meaning": f"Saved masks reaching one another at {dilation} px dilation; minimum {settings['min_shared_boundary_px']} boundary pixels.",
                        "message": ""})
    return {"sources": sources, "preview": True, "message": ""}
