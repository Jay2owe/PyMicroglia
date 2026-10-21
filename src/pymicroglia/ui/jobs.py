"""One-at-a-time recorded analysis jobs for the local workbench."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Callable
from uuid import uuid4
import json
import time
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobManager:
    def __init__(self, runner: Callable[..., dict] | None = None):
        if runner is None:
            from pymicroglia.run import run_recorded
            runner = run_recorded
        self.runner = runner
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pymicroglia-ui")
        self.jobs: dict[str, dict[str, Any]] = {}
        self.lock = RLock()

    def submit(self, action: str, params: dict, claim: str = "", request: str = "") -> dict:
        from pymicroglia.knowledge import describe, validate

        spec = describe(action)
        if not spec.get("ok") or spec.get("pending"):
            raise ValueError(spec.get("message") or f"Action {action!r} is unavailable")
        check = validate(action, params)
        if not check.get("ok") or check.get("unknown_params") or check.get("problems"):
            raise ValueError(check.get("message") or str(check.get("problems") or check))
        if spec.get("claim_required") and not claim.strip():
            raise ValueError("This analysis needs a sentence describing what it should show")
        identifier = uuid4().hex
        row = {"id": identifier, "action": action, "state": "queued", "submitted_at": _now(),
               "params": params, "claim": claim, "request": request, "record": None,
               "outputs": [], "error": None}
        with self.lock:
            self.jobs[identifier] = row
        self.pool.submit(self._run, identifier)
        return self.get(identifier)

    def submit_exploration_rhythms(self, run: str, stem: str, options: dict) -> dict:
        from .exploration_rhythms import validate_request
        prepared = validate_request(run, stem, options)
        return self._queue_exploration_rhythms(prepared, options)

    def _queue_exploration_rhythms(self, prepared, options) -> dict:
        root, _, movie, _, observations, _, config = prepared
        options = json.loads(json.dumps(options))
        with self.lock:
            existing = next((row for row in self.jobs.values()
                             if row.get("kind") == "exploration_rhythms"
                             and row["state"] in {"queued", "running"}
                             and row["params"]["run"] == str(root)
                             and row["params"]["stem"] == movie["stem"]
                             and row["params"]["options"] == options), None)
            if existing is not None:
                return dict(existing)
            identifier = uuid4().hex
            row = {"id": identifier, "kind": "exploration_rhythms", "action": "rhythm_analysis",
                   "state": "queued", "submitted_at": _now(), "error": None, "outputs": [],
                   "cells": len(observations),
                   "progress": {"stage": "queued", "completed": 0,
                                "total": len(observations) * len(config["metrics"]),
                                "completed_cells": 0, "total_cells": len(observations)},
                   "params": {"run": str(root), "stem": movie["stem"], "options": options}}
            self.jobs[identifier] = row
        self.pool.submit(self._exploration_rhythms, identifier)
        return self.get(identifier)

    def submit_exploration_rhythm_batch(self, run, recordings, options, *, _completed=None) -> dict:
        from .exploration_data import recording
        from .exploration_rhythms import validate_request
        from .exploration_rhythm_batches import save_batch, snapshot

        root, _, _ = recording(run)
        if not isinstance(recordings, list) or not recordings or len(recordings) > 256:
            raise ValueError("Choose between one and 256 loaded recordings.")
        prepared, seen = [], set()
        # Validate the complete batch before submitting any work.
        for source in recordings:
            if (not isinstance(source, dict) or not isinstance(source.get("run"), str)
                    or not isinstance(source.get("stem"), str)):
                raise ValueError("Each recording needs its saved run and recording name.")
            try:
                request = validate_request(source["run"], source["stem"], options)
            except (ValueError, OSError) as exc:
                raise ValueError(f"{source['stem']}: {exc}") from None
            key = (str(request[0]), request[2]["stem"])
            if key not in seen:
                seen.add(key)
                prepared.append(request)
        params = {"run": str(root), "options": json.loads(json.dumps(options)),
                  "recordings": [{"run": str(item[0]), "stem": item[2]["stem"]}
                                 for item in prepared]}
        retained = {}
        if _completed:
            from .exploration_rhythms import reusable_rhythm_result
            for request in prepared:
                key = (str(request[0]), request[2]["stem"])
                entry = _completed.get(key)
                if entry and reusable_rhythm_result(request[0], request[2], request[-1], entry.get("analysis_id")):
                    retained[key] = {**entry, "reused": True}
        with self.lock:
            existing = next((row for row in self.jobs.values()
                             if row.get("kind") == "exploration_rhythm_batch"
                             and row["params"] == params
                             and snapshot(row, self.jobs)["state"] in {"queued", "running"}), None)
            if existing:
                return snapshot(existing, self.jobs)
            identifier = uuid4().hex
            batch = {"schema_version": 1, "id": identifier, "kind": "exploration_rhythm_batch",
                     "action": "rhythm_analysis", "state": "queued", "submitted_at": _now(),
                     "params": params, "recordings": [], "outputs": [], "error": None}
            # Saving the plan first makes a queue interrupted during submission recoverable.
            batch["recordings"] = [retained.get((source["run"], source["stem"])) or
                                   {**source, "job_id": uuid4().hex, "state": "queued"}
                                   for source in params["recordings"]]
            save_batch(snapshot(batch, {}))
            for entry, request in zip(batch["recordings"], prepared):
                if entry["state"] == "completed":
                    continue
                job = self._queue_exploration_rhythms(request, options)
                entry["job_id"] = job["id"]
            self.jobs[identifier] = batch
            save_batch(snapshot(batch, self.jobs))
            return snapshot(batch, self.jobs)

    def rhythm_batch(self, run, identifier) -> dict:
        from .exploration_rhythm_batches import batch_path, load_batch, snapshot
        with self.lock:
            row = self.jobs.get(identifier)
            if row and row.get("kind") == "exploration_rhythm_batch":
                if batch_path(run, identifier) != batch_path(row["params"]["run"], identifier):
                    raise ValueError("This batch belongs to another comparison workspace.")
                return snapshot(row, self.jobs)
        return load_batch(run, identifier)

    def resume_rhythm_batch(self, run, identifier) -> dict:
        batch = self.rhythm_batch(run, identifier)
        completed = {(entry["run"], entry["stem"]): entry for entry in batch["recordings"]
                     if entry["state"] == "completed"}
        return self.submit_exploration_rhythm_batch(run, batch["params"]["recordings"],
                                                  batch["params"]["options"], _completed=completed)

    def _save_rhythm_batches(self, job_id):
        from .exploration_rhythm_batches import save_batch, snapshot
        with self.lock:
            for batch in self.jobs.values():
                if (batch.get("kind") == "exploration_rhythm_batch"
                        and any(entry["job_id"] == job_id for entry in batch["recordings"])):
                    save_batch(snapshot(batch, self.jobs))

    def submit_comparison(self, payload: dict) -> dict:
        from .exploration_comparison import validate_design
        validate_design(payload['design'])
        if not isinstance(payload.get('test'), dict):
            raise ValueError('Choose comparison test settings')
        identifier = uuid4().hex
        with self.lock:
            self.jobs[identifier] = dict(id=identifier, kind='group_comparison', action='group_comparison',
                state='queued', submitted_at=_now(), params=payload, outputs=[], error=None)
        self.pool.submit(self._comparison, identifier)
        return self.get(identifier)

    def _comparison(self, identifier):
        from .exploration_comparison import run_comparison
        with self.lock:
            row = self.jobs[identifier]
            row.update(state='running', started_at=_now())
        try:
            result = run_comparison(row['params'])
            with self.lock:
                row.update(state='completed', finished_at=_now(), result=result)
        except Exception as exc:
            with self.lock:
                row.update(state='failed', finished_at=_now(), error=f'{type(exc).__name__}: {exc}')

    def submit_event_clip(self, run: str, stem: str, event_id: str, options: dict | None = None) -> dict:
        from .events import prepare_clip
        request = prepare_clip(run, stem, event_id, options)
        with self.lock:
            existing = next((row for row in self.jobs.values()
                             if row.get("kind") == "event_clip" and row["state"] in {"queued", "running"}
                             and row.get("event_request", {}).get("event_id") == event_id
                             and row["event_request"]["run"] == request["run"]
                             and row["event_request"].get("options") == request.get("options")), None)
            if existing:
                return dict(existing)
            identifier = uuid4().hex
            row = {"id": identifier, "kind": "event_clip", "action": "follow",
                   "state": "completed" if request["reused"] else "queued", "submitted_at": _now(),
                   "params": request.get("params", {}), "event_request": request,
                   "claim": "", "request": "Display the selected saved event without changing tracking",
                   "record": None, "outputs": [request["clip"]["path"]] if request["reused"] else [],
                   "reused": request["reused"], "error": None,
                   "progress": {"stage": "reused" if request["reused"] else "queued"}}
            self.jobs[identifier] = row
        if not request["reused"]:
            self.pool.submit(self._run, identifier)
        return self.get(identifier)

    def submit_rhythm_audit(self, run: str, stem: str, options: dict) -> dict:
        from .exploration_audit import validate_request
        root, movie, _, config, _ = validate_request(run, stem, options)
        params = {"run": str(root), "stem": movie["stem"], "options": config}
        with self.lock:
            existing = next((row for row in self.jobs.values()
                             if row.get("kind") == "rhythm_audit" and row["state"] in {"queued", "running"}
                             and not row.get("cancel_requested") and row["params"] == params), None)
            if existing:
                return dict(existing)
            identifier = uuid4().hex
            row = {"id": identifier, "kind": "rhythm_audit", "action": "period_method_audit",
                   "state": "queued", "submitted_at": _now(), "params": params,
                   "error": None, "outputs": [], "cancel_requested": False,
                   "progress": {"stage": "queued", "completed": 0,
                                "total": len(config["detrend_methods"]) * len(config["period_methods"]) * len(config["input_filters"])}}
            self.jobs[identifier] = row
        self.pool.submit(self._rhythm_audit, identifier)
        return self.get(identifier)

    def cancel_rhythm_audit(self, identifier: str) -> dict:
        with self.lock:
            row = self.jobs[identifier]
            if row.get("kind") != "rhythm_audit":
                raise ValueError("This cancellation action is for rhythm audits.")
            if row["state"] in {"queued", "running"}:
                row["cancel_requested"] = True
        return self.get(identifier)

    def _rhythm_audit(self, identifier: str) -> None:
        from .exploration_audit import AuditCancelled, audit_files, run_audit
        started = time.monotonic()
        with self.lock:
            row = self.jobs[identifier]
            row.update(state="running", started_at=_now())
            params = dict(row["params"])
        def cancelled():
            with self.lock:
                return row["cancel_requested"]
        def progress(value):
            with self.lock:
                row["progress"] = {**value, "elapsed_seconds": time.monotonic() - started}
        try:
            result = run_audit(**params, progress=progress, cancelled=cancelled, reuse=True)
            _, files = audit_files(params["run"], params["stem"], result["audit_id"])
            with self.lock:
                row.update(state="completed", finished_at=_now(), audit_id=result["audit_id"],
                           reused=bool(result.get("reused")), outputs=[str(file) for file in files])
        except Exception as exc:
            with self.lock:
                state = "cancelled" if isinstance(exc, AuditCancelled) else "failed"
                row.update(state=state, finished_at=_now(), error=str(exc))
                row["progress"] = {**row["progress"], "stage": state}

    def _exploration_rhythms(self, identifier: str) -> None:
        from .exploration_rhythms import run_rhythm_analysis
        with self.lock:
            row = self.jobs[identifier]
            row.update(state="running", started_at=_now())
            params = dict(row["params"])
        started = time.monotonic()
        def progress(value):
            elapsed = time.monotonic() - started
            completed, total = value["completed"], value["total"]
            remaining = (elapsed / completed * (total - completed)
                         if completed >= 3 and total > completed else None)
            with self.lock:
                row["progress"] = {**value, "elapsed_seconds": elapsed,
                                   "remaining_seconds": remaining}
        try:
            result = run_rhythm_analysis(**params, analysis_id=identifier, progress=progress, reuse=True)
            folder = Path(result["source"]).parent
            with self.lock:
                row.update(state="completed", finished_at=_now(),
                           outputs=[str(folder / name) for name in ("result.json", "statistics.csv", "reproduce.py")],
                           rhythm_result=result["source"], analysis_id=result["analysis_id"], reused=bool(result.get("reused")))
                row["progress"] = {**row["progress"], "stage": "completed",
                                   "elapsed_seconds": time.monotonic() - started,
                                   "remaining_seconds": 0}
        except Exception as exc:
            with self.lock:
                row.update(state="failed", finished_at=_now(), error=f"{type(exc).__name__}: {exc}")
                row["progress"] = {**row["progress"], "stage": "failed",
                                   "elapsed_seconds": time.monotonic() - started,
                                   "remaining_seconds": None}
        finally:
            self._save_rhythm_batches(identifier)

    def submit_refinement(self, manifest_path: str, method: str = "single_frame",
                          options: dict | None = None, weights: str | None = None) -> dict:
        from pymicroglia.tracking.refinement_methods import describe

        manifest = Path(manifest_path).expanduser().resolve()
        if manifest.name != "handoff.json" or not manifest.is_file():
            raise ValueError("Choose a Motion measurement handoff receipt")
        handoff = json.loads(manifest.read_text(encoding="utf-8"))
        if handoff.get("schema") != "motion.measurement-handoff.v1":
            raise ValueError("This is not a Motion measurement handoff")
        movie = handoff.get("movie") or {}
        for key in ("labels", "raw"):
            file = Path(str(movie.get(key) or "")).resolve()
            if not file.is_file() or file.parent != manifest.parent:
                raise ValueError(f"Handoff {key} file is missing or outside its folder")
        if method == "off":
            raise ValueError("Off uses the exported Motion labels without a refinement job")
        if not isinstance(options, dict):
            raise ValueError("Refiner settings must be a JSON object")
        describe(method, options)
        if weights is not None and not Path(weights).is_file():
            raise ValueError("Selected refiner weights file was not found")
        identifier = uuid4().hex
        row = {"id": identifier, "kind": "refinement", "action": "refine_outlines",
               "state": "queued", "submitted_at": _now(), "method": method,
               "handoff": str(manifest), "error": None, "outputs": []}
        with self.lock:
            self.jobs[identifier] = row
        self.pool.submit(self._refine, identifier, handoff, manifest.parent, method, options, weights)
        return self.get(identifier)

    def submit_pipeline(self, name: str, params: dict, claim: str = "") -> dict:
        from .pipeline_contract import validate

        supplied = dict(params)
        from pymicroglia.pipelines import get
        import inspect
        if "claim" in inspect.signature(get(name).run).parameters:
            supplied["claim"] = claim
        check = validate(name, supplied)
        if not check["ok"]:
            raise ValueError(check["message"])
        identifier = uuid4().hex
        row = {"id": identifier, "kind": "pipeline", "action": name, "state": "queued",
               "submitted_at": _now(), "params": supplied, "error": None, "outputs": []}
        with self.lock:
            self.jobs[identifier] = row
        self.pool.submit(self._pipeline, identifier, name, supplied)
        return self.get(identifier)

    def _pipeline(self, identifier: str, name: str, params: dict) -> None:
        from pymicroglia.pipelines import get

        with self.lock:
            row = self.jobs[identifier]
            row["state"] = "running"
            row["started_at"] = _now()
        try:
            result = get(name).run(**params)
            outputs = result.get("outputs", []) if isinstance(result, dict) else []
            summary = ({"keys": sorted(map(str, result))} if isinstance(result, dict)
                       else {"type": type(result).__name__})
            with self.lock:
                row.update(state="completed", finished_at=_now(),
                           outputs=outputs if isinstance(outputs, list) else list(outputs.values())
                           if isinstance(outputs, dict) else [],
                             result_summary=summary,
                             measurement_run=result.get('measurement_run') if isinstance(result, dict) else None)
        except Exception as exc:
            with self.lock:
                row.update(state="failed", error=f"{type(exc).__name__}: {exc}", finished_at=_now())

    def _refine(self, identifier: str, handoff: dict, folder: Path, method: str,
                options: dict, weights: str | None) -> None:
        with self.lock:
            row = self.jobs[identifier]
            row["state"] = "running"
            row["started_at"] = _now()
        try:
            from pymicroglia.tracking.post_refine import refine_motion

            output = folder / f"refinement_{identifier[:12]}"
            movie = handoff["movie"]
            result = refine_motion(movie["raw"], movie["labels"],
                                   output_dir=output, method=method,
                                   weights=weights, method_options=options,
                                   um_per_px=handoff.get("microns_per_pixel"))
            params = dict(handoff.get("measure_params") or {})
            params["movies"] = [dict(movie, labels=result["outputs"]["refined_labels"],
                                     notes=f"Refined from Motion session {handoff['source']} by {method}")]
            with self.lock:
                row.update(state="completed", finished_at=_now(),
                           outputs=list(result["outputs"].values()),
                           changed_pixels=result["settings"]["changed_pixels"],
                           identity_presence=result["settings"]["identity_presence"],
                           measure_params=params, settings=result["outputs"]["settings"])
        except Exception as exc:
            with self.lock:
                row.update(state="failed", error=f"{type(exc).__name__}: {exc}", finished_at=_now())

    def _run(self, identifier: str) -> None:
        with self.lock:
            row = self.jobs[identifier]
            row["state"] = "running"
            row["started_at"] = _now()
            action, params, claim, request = row["action"], row["params"], row["claim"], row["request"]
        try:
            progress_token = None
            figure_token = None
            if action in {'cell_grid', 'cell_analysis_grid', 'all_cell_trace_grid'}:
                from pymicroglia.figure_tables.actions import _figure_progress
                def figure_progress(value):
                    with self.lock:
                        row['progress'] = {**value, 'stage': value.get('stage', 'analysing cells')}
                figure_token = _figure_progress.set(figure_progress)
            if row.get("kind") == "event_clip":
                from pymicroglia.figure_tables.film_action import _render_progress
                def progress(value):
                    with self.lock:
                        row["progress"] = value
                progress_token = _render_progress.set(progress)
            try:
                done = self.runner(action, claim=claim, request=request, entry="ui", **params)
            finally:
                if progress_token is not None:
                    _render_progress.reset(progress_token)
                if figure_token is not None:
                    _figure_progress.reset(figure_token)
            record = done.get("record") or {}
            result = done.get("result")
            outputs = record.get("outputs") or []
            if row.get("kind") == "event_clip":
                from .events import record_clip
                outputs = record_clip(row["event_request"], result)
            measurement_run = (result.get("run", {}).get("folder")
                               if action == "measure" and isinstance(result, dict) else None)
            with self.lock:
                row.update(state="completed", record=record, outputs=outputs,
                           recorded=bool(done.get("recorded")), measurement_run=measurement_run,
                           finished_at=_now())
        except Exception as exc:
            with self.lock:
                row.update(state="failed", error=f"{type(exc).__name__}: {exc}", finished_at=_now())

    def get(self, identifier: str) -> dict:
        with self.lock:
            if identifier not in self.jobs:
                raise KeyError(identifier)
            row = self.jobs[identifier]
            if row.get("kind") == "exploration_rhythm_batch":
                from .exploration_rhythm_batches import snapshot
                return snapshot(row, self.jobs)
            return dict(row)

    def list(self) -> list[dict]:
        with self.lock:
            return [self.get(row["id"]) for row in reversed(list(self.jobs.values()))]

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=False)
