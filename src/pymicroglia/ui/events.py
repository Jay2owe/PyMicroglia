"""Browse recorded events and prepare requests for the existing cell-film action."""
from __future__ import annotations

import base64
from functools import lru_cache
import hashlib
from io import BytesIO
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw

from .exploration_data import _observations, number, read_rows, recording, table_path
from .exploration_context import _labels, load_track_timeline
from .exploration_images import _contrast, declared_source, geometry, plane, signature
from .exploration_rhythms import _atomic_json

MAX_EVENTS = 10_000
GROUPS = {"merges": "Merges / hiding", "births": "Births / divisions",
          "arrivals": "Arrivals", "endings": "Endings", "gaps": "Other gaps",
          "states": "State changes", "boundaries": "Recording boundaries"}


def _key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:24]


def _folder(root, stem):
    return root / "AI_Exports/event_videos" / _key(stem)


@lru_cache(maxsize=16)
def _fingerprint(path, size, modified):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _inputs(root, movie):
    paths = {name: declared_source(root, movie, name) for name in ("labels", "raw")}
    paths["measurements"] = table_path(root, movie, "cell_frame")
    return {name: _fingerprint(*signature(path)) for name, path in paths.items() if path and path.is_file()} | {
        "movie": _key(movie)}


def _saved_clips(root, movie, artifacts):
    path = _folder(root, movie["stem"]) / "clips.json"
    if not path.is_file():
        return []
    if path.stat().st_size > 8 * 1024**2:
        raise ValueError("The event-clip index exceeds the interactive metadata budget")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or data.get("inputs") != _inputs(root, movie):
        return []
    output = []
    for clip in data.get("clips", [])[:1200]:
        file = (root / clip["path"]).resolve()
        if not file.is_relative_to(root) or file.suffix.lower() != ".mp4":
            raise ValueError("An event clip leaves the selected run")
        if file.is_file():
            item = {**clip, "path": str(file)}
            if artifacts:
                item["token"] = artifacts.select_files([file], root)[0]["token"]
                script = file.parent / "reproduce.py"
                if script.is_file():
                    item["script_token"] = artifacts.select_files([script], root)[0]["token"]
            output.append(item)
    return output


def load_events(run, stem=None, artifacts=None):
    from pymicroglia.figure_tables.event_labels import EVENT_LABELS, EVENT_TITLES, event_label
    from .exploration_lifecycle import review_details

    root, manifest, movie = recording(run, stem)
    _, observations, frames, _ = _observations(root, movie)
    rows, sources = [], []
    for table in ("lifecycle_events", "regime_transitions"):
        path = table_path(root, movie, table)
        if path:
            sources.append(str(path))
        for saved in read_rows(path):
            if saved.get('stem') and saved['stem'] != movie['stem']:
                continue
            identity = number(saved.get("identity"))
            frame = number(saved.get("frame_index", saved.get("to_frame_index")))
            if identity not in observations or frame is None or int(frame) != frame or not 0 <= frame < len(frames):
                continue
            identity, frame = int(identity), int(frame)
            event = saved.get("event") or f"{saved.get('from_regime', '')} → {saved.get('to_regime', '')}"
            partner = number(saved.get("partner_identity"))
            group = ("states" if table == "regime_transitions" else
                     "births" if event in {"born", "divided"} else
                     "merges" if event == "absorbed" else
                     "arrivals" if event in {"appeared", "entered_field"} else
                     "boundaries" if event in {"present_at_start", "present_at_end"} else "endings")
            row = {"identity": identity, "frame_index": frame, "end_frame": frame,
                   "event": event, "label": event_label(event), "group": group,
                   "status": saved.get("event_status", "status not recorded"),
                   "source": "lifecycle" if table == "lifecycle_events" else "states",
                   "partners": [int(partner)] if partner and int(partner) == partner else [],
                   "evidence": saved}
            row["id"] = _key(row)
            rows.append(row)
    timeline = load_track_timeline(root, movie["stem"], "identity")
    sources.extend(timeline["history_sources"])
    for cell in timeline["cells"]:
        for gap in cell["gaps"]:
            row = {"identity": cell["identity"], "frame_index": gap["start_frame"],
                   "end_frame": gap["end_frame"], "event": gap["reason"],
                   "label": "; ".join(gap["labels"]) or "Gap with no recorded explanation",
                   "group": "merges" if "same_host_merge_hiding" in gap["mechanisms"] else "gaps",
                   "status": "tracking assessment" if gap["mechanisms"] else "unexplained",
                   "source": "tracking", "partners": sorted(set(gap["hosts"] + gap["prior_owners"] + gap["next_owners"]) - {cell["identity"]}),
                   "evidence": gap}
            row["id"] = _key(row)
            rows.append(row)
    rows = sorted({r["id"]: r for r in rows}.values(), key=lambda r: (r["frame_index"], r["identity"], r["id"]))
    total = len(rows)
    rows = rows[:MAX_EVENTS]
    for row in rows:
        row.update(hours=frames[row["frame_index"]]["hours"],
                   source_frame=frames[row["frame_index"]]["source_frame_index"]+1,
                   source_end_frame=frames[row["end_frame"]]["source_frame_index"]+1)
    clips = _saved_clips(root, movie, artifacts)
    # Existing native lifecycle clips have deterministic filenames. No directory walk.
    legacy_films = root / 'films' / movie['stem']
    for row in rows if legacy_films.is_dir() else ():
        if row["source"] != "lifecycle":
            continue
        file = legacy_films / f"{row['identity']}_{row['event']}_{row['frame_index']}.mp4"
        if file.is_file():
            item = {"event_id": row["id"], "path": str(file), "name": file.name, "legacy": True}
            if artifacts:
                item["token"] = artifacts.select_files([file], root)[0]["token"]
            clips.append(item)
    tabs = [{"key": "all", "title": "All events"}]
    for name in [*EVENT_TITLES, *(n for n in EVENT_LABELS if n not in EVENT_TITLES)]:
        tabs.append({"key": "merges" if name == "absorbed" else name,
                     "title": EVENT_TITLES.get(name, name.replace('_', ' ').title()),
                     **({"group": "merges"} if name == "absorbed" else {"event": name})})
    tabs.extend({"key": key, "title": GROUPS[key], "group": key} for key in ("gaps", "states"))
    return {"run": str(root), "stem": movie["stem"], "recordings": [m["stem"] for m in manifest["movies"]],
            "rows": rows, "total": total, "truncated": total > len(rows), "groups": GROUPS, "tabs": tabs,
            "identities": sorted(observations), "frames": frames, "clips": clips,
            "sources": sources, "lifecycle_saved": table_path(root, movie, "lifecycle_events") is not None,
            "lifecycle_review": review_details(root, movie),
            "message": "" if rows else "No saved lifecycle events or tracking gap assessments in this recording.",
            "history_message": timeline.get("history_message", "")}


def event_window(data, event_id, event_hours=6):
    if not np.isfinite(event_hours) or not 0 <= event_hours <= 168:
        raise ValueError("Context must be between 0 and 168 hours on each side")
    row = next((r for r in data["rows"] if r["id"] == event_id), None)
    if row is None:
        raise ValueError("This event is not recorded in the selected recording")
    frames = data["frames"]
    a, b = frames[row["frame_index"]]["hours"], frames[row["end_frame"]]["hours"]
    if a is None or b is None:
        raise ValueError("This event has no saved time axis")
    window = [f["frame_index"] for f in frames if f["hours"] is not None and a-event_hours <= f["hours"] <= b+event_hours]
    return row, window[0], window[-1]


def load_event_frame(run, stem, event_id, frame, *, event_hours=6, cell_lut="red", crop_px=96):
    data = load_events(run, stem)
    event, first, last = event_window(data, event_id, event_hours)
    if not first <= frame <= last:
        raise ValueError("Requested frame is outside the selected event window")
    if not 16 <= crop_px <= 1024:
        raise ValueError("Crop size must be between 16 and 1024 pixels")
    root, _, movie = recording(run, stem)
    label_path, raw_path = (declared_source(root, movie, k) for k in ("labels", "raw"))
    labels = _labels(*signature(label_path))
    if len(labels) != len(data["frames"]):
        raise ValueError("The saved mask frame count disagrees with measurements")
    return _render_event_frame(data, event, frame, first, last, labels, raw_path, cell_lut, crop_px)


@lru_cache(maxsize=12)
def _painted_plane(path, size, modified, source, cell_lut):
    from auto_organotypic.render.luts import paint
    pixels = plane(Path(path), source)
    lo, hi = _contrast(path, size, modified)
    normalized = np.nan_to_num(np.clip((pixels.astype(float)-lo)/(hi-lo), 0, 1))
    return Image.fromarray(paint(normalized, cell_lut))


def _render_event_frame(data, event, frame, first, last, labels, raw_path, cell_lut, crop_px):
    from auto_organotypic.render.luts import paint

    own = labels[first:last+1] == event["identity"]
    _, yy, xx = np.nonzero(own)
    if not yy.size:
        observed = np.flatnonzero(np.any(labels == event['identity'], axis=(1, 2)))
        if not len(observed):
            raise ValueError("This identity has no saved mask to locate the event camera")
        before = observed[observed < first]
        reference = int(before[-1] if len(before) else observed[0])
        yy, xx = np.nonzero(labels[reference] == event['identity'])
    cy, cx = (int((yy.min()+yy.max())/2), int((xx.min()+xx.max())/2))
    half = max(crop_px//2, (int(yy.max()-yy.min())+1)//2+8, (int(xx.max()-xx.min())+1)//2+8)
    y0, y1, x0, x1 = cy-half, cy+half+1, cx-half, cx+half+1
    source = data["frames"][frame]["source_frame_index"]
    picture = (_painted_plane(*signature(raw_path), source, cell_lut) if raw_path
               else Image.fromarray(paint((labels[frame] > 0).astype(float), cell_lut)))
    if picture.size != (labels.shape[2], labels.shape[1]):
        raise ValueError("Raw and masks have different field dimensions")
    picture = picture.crop((x0, y0, x1, y1))
    draw = ImageDraw.Draw(picture)
    visible = []
    for identity in [event['identity'], *event['partners']]:
        cells = geometry(labels[frame], identity)
        if not cells:
            continue
        cell = cells[0]
        x_start, y_start, x_end, y_end = cell['bounds']
        if x_end <= x0 or x_start >= x1 or y_end <= y0 or y_start >= y1:
            continue
        visible.append(cell["identity"])
        colour = "#72ded3" if cell["identity"] == event["identity"] else "#efc77b"
        for path in cell["contours"]:
            points = [(x-x0, y-y0) for x,y in path]
            if len(points) > 1:
                draw.line(points, fill=colour, width=1)
        draw.text((cell["centre"][0]-x0+2, cell["centre"][1]-y0+2), str(cell["identity"]), fill=colour)
    out = BytesIO();picture.save(out, format="PNG")
    return {"image": "data:image/png;base64,"+base64.b64encode(out.getvalue()).decode(),
            "frame_index": frame, "source_frame": source+1, "hours": data["frames"][frame]["hours"],
            "present": event["identity"] in visible, "visible_identities": visible,
            "first_frame": first, "last_frame": last, "image_filter": "none", "event_id": event['id']}


def load_event_previews(run, stem, event_ids, *, cell_lut='red', crop_px=96):
    """Read a bounded group of native event frames without rerunning tracking."""
    if not event_ids or len(event_ids) > 32 or len(set(event_ids)) != len(event_ids):
        raise ValueError('Request between 1 and 32 distinct saved events')
    if not isinstance(crop_px, int) or not 16 <= crop_px <= 1024:
        raise ValueError('Crop size must be an integer between 16 and 1024 pixels')
    from auto_organotypic.render.luts import paint
    paint(np.zeros((1, 1)), cell_lut)
    data = load_events(run, stem)
    events = {row['id']: row for row in data['rows']}
    if set(event_ids) - events.keys():
        raise ValueError('This event is not recorded in the selected recording')
    root, _, movie = recording(run, stem)
    labels = _labels(*signature(declared_source(root, movie, 'labels')))
    if len(labels) != len(data['frames']):
        raise ValueError('The saved mask frame count disagrees with measurements')
    raw_path = declared_source(root, movie, 'raw')
    output = []
    for identifier in event_ids:
        row = events[identifier]
        try:
            output.append(_render_event_frame(data, row, row['frame_index'], row['frame_index'],
                                              row['end_frame'], labels, raw_path, cell_lut, crop_px))
        except (OSError, ValueError) as error:
            output.append({'event_id': identifier, 'error': str(error)})
    return {'run': data['run'], 'stem': data['stem'], 'previews': output}


def prepare_clip(run, stem, event_id, options=None):
    options = dict(options or {})
    unknown = set(options) - {"event_hours", "fps", "crop_px", "cell_lut"}
    if unknown:
        raise ValueError(f"Unknown event display settings: {sorted(unknown)}")
    options = {"event_hours": 6., "fps": 6., "crop_px": 96, "cell_lut": "red", **options}
    if not np.isfinite(options["fps"]) or not .5 <= options["fps"] <= 60:
        raise ValueError("Playback must be between 0.5 and 60 frames per second")
    if not isinstance(options["crop_px"], int) or not 16 <= options["crop_px"] <= 1024:
        raise ValueError("Crop size must be an integer between 16 and 1024 pixels")
    from auto_organotypic.render.luts import paint
    paint(np.zeros((1, 1)), options["cell_lut"])
    data = load_events(run, stem)
    event, first, last = event_window(data, event_id, options["event_hours"])
    root, _, movie = recording(run, stem)
    inputs = _inputs(root, movie)
    matched = next((c for c in data["clips"] if c.get("event_id") == event_id and c.get("options") == options), None)
    if matched:
        return {"reused": True, "clip": matched, "run": str(root), "stem": movie["stem"], "event_id": event_id}
    folder = _folder(root, movie["stem"]) / _key([event_id, options, inputs])
    if folder.exists():
        folder = folder.with_name(folder.name+"_"+uuid4().hex[:8])
    params = {"run": str(root), "stem": movie["stem"], "identities": [event["identity"]],
              "span": "event" if event["source"] == "lifecycle" else "recording",
              "events": [event["event"]] if event["source"] == "lifecycle" else [],
              "event_frames": [event["frame_index"]] if event["source"] == "lifecycle" else [],
              "frame_range": [first, last], "event_hours": options["event_hours"],
              "fps": options["fps"], "crop_px": options["crop_px"], "cell_lut": options["cell_lut"],
              "output_dir": str(folder), "display_options": {"image_filter": "none", "display_gamma": 1.,
              "display_black_percentile": 1., "display_gain": 1.,
              "context_caption": f"{event['label']} ({event['status']})"}}
    return {"params": params, "inputs": inputs, "options": options, "event_id": event_id,
            "event": event, "run": str(root), "stem": movie["stem"], "reused": False}


def record_clip(request, result):
    root, _, movie = recording(request["run"], request["stem"])
    if _inputs(root, movie) != request["inputs"]:
        raise ValueError("The event-video inputs changed while the clip was rendering")
    clips = result.get("clips") or []
    if len(clips) != 1 or not Path(clips[0]["path"]).is_file():
        raise ValueError("The native event-film action did not produce exactly one clip")
    file = Path(clips[0]["path"])
    entry = {"event_id": request["event_id"], "path": file.relative_to(root).as_posix(),
             "name": file.name, "options": request["options"], "frames": clips[0]["frames"]}
    path = _folder(root, movie["stem"]) / "clips.json"
    prior = json.loads(path.read_text()) if path.is_file() else {}
    entries = prior.get("clips", []) if prior.get("inputs") == request["inputs"] else []
    _atomic_json(path, {"schema_version": 1, "inputs": request["inputs"], "clips": [*entries, entry]})
    params = {**request["params"], "output_dir": "replayed"}
    script = file.parent / "reproduce.py"
    script.write_text("from pathlib import Path\nfrom pymicroglia.figure_tables.film_action import follow\n"
                      f"settings = {params!r}\nsettings['output_dir'] = str(Path(__file__).parent / 'replayed')\n"
                      "print(follow(**settings))\n", encoding="utf-8")
    return [str(file), str(script), str(path)]
