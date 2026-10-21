"""Durable progress for preparing saved rhythm results across recordings."""
from __future__ import annotations

import json

from .exploration_data import recording
from .exploration_rhythms import _atomic_json

ACTIVE = {"queued", "running"}


def batch_path(run, identifier):
    if (not isinstance(identifier, str) or len(identifier) != 32
            or any(char not in "0123456789abcdef" for char in identifier)):
        raise ValueError("Choose a saved rhythm batch.")
    root, _, _ = recording(run)
    return root / "AI_Exports" / "exploration_rhythm_batches" / f"{identifier}.json"


def summarize(batch):
    entries = batch["recordings"]
    counts = {state: sum(entry["state"] == state for entry in entries)
              for state in ("queued", "running", "completed", "failed", "interrupted")}
    batch["progress"] = {**counts, "total": len(entries),
                         "finished": counts["completed"] + counts["failed"]}
    batch["state"] = ("running" if counts["running"] else "queued" if counts["queued"]
                      else "interrupted" if counts["interrupted"]
                      else "failed" if counts["failed"] else "completed")
    return batch


def snapshot(row, jobs):
    batch = json.loads(json.dumps(row))
    for entry in batch["recordings"]:
        job = jobs.get(entry["job_id"])
        if job is not None:
            for key in ("state", "progress", "analysis_id", "error", "reused", "finished_at"):
                if key in job:
                    entry[key] = job[key]
        elif entry["state"] in ACTIVE:
            entry["state"] = "interrupted"
    return summarize(batch)


def save_batch(batch):
    from .exploration_comparison import _paths
    path = batch_path(batch["params"]["run"], batch["id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    root = path.parents[2]
    stored = _paths(batch, root, relative=True)
    stored["params"] = _paths(stored["params"], root, relative=True)
    stored["params"]["run"] = "."
    _atomic_json(path, stored)


def load_batch(run, identifier):
    from .exploration_comparison import _paths
    path = batch_path(run, identifier)
    if not path.is_file():
        raise ValueError("The saved rhythm batch was not found in this workspace.")
    batch = json.loads(path.read_text(encoding="utf-8"))
    if batch.get("schema_version") != 1:
        raise ValueError("Unsupported rhythm batch version.")
    # The current workspace location is authoritative after a folder move.
    root = path.parents[2]
    batch = _paths(batch, root)
    batch["params"] = _paths(batch["params"], root)
    batch["params"]["run"] = str(root)
    return snapshot(batch, {})
