"""Local HTTP application for the PyMicroglia desktop workbench."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlencode
from urllib.request import Request as UrlRequest, urlopen
from urllib.error import HTTPError, URLError
import asyncio
import json
import os
import threading
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .catalog import load_map
from .jobs import JobManager
from .results import ArtifactIndex, recent_runs, saved_record

STATIC = Path(__file__).with_name("static")


def _ping_motion(motion_url: str) -> None:
    try:
        with urlopen(motion_url.rstrip("/") + "/api/heartbeat", timeout=2):
            pass
    except Exception:  # noqa: BLE001 - Motion's heartbeat is optional
        pass


def create_app(*, auto_url: str | None = None, motion_url: str | None = None,
               jobs: JobManager | None = None) -> FastAPI:
    if motion_url:
        parsed = urlparse(motion_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Motion service must use a loopback HTTP address")
    app = FastAPI(title="PyMicroglia Workbench", docs_url=None, redoc_url=None)
    manager = jobs or JobManager()
    artifacts = ArtifactIndex()
    app.state.jobs = manager
    app.state.auto_url = auto_url
    app.state.motion_url = motion_url
    app.state.artifacts = artifacts
    app.state.last_heartbeat = None  # monotonic time of the page's last heartbeat

    @app.get("/api/heartbeat")
    def heartbeat() -> dict:
        """An open page calls this every 15 s; the browser launcher exits when it stops."""
        app.state.last_heartbeat = time.monotonic()
        if app.state.motion_url:
            # Best effort: keeps a separately launched motion-edit alive while this page is open.
            threading.Thread(target=_ping_motion, args=(app.state.motion_url,), daemon=True,
                             name="pymicroglia-motion-heartbeat").start()
        busy = getattr(manager, "busy", None)
        return {"ok": True, "busy": bool(busy()) if busy else False}

    @app.get("/api/config")
    def config() -> dict:
        from ..learned_mask import apply as masking
        try:
            configured = masking.weights().is_file()
        except FileNotFoundError:
            configured = False
        return {"auto_url": app.state.auto_url, "motion_url": app.state.motion_url,
                "mask_weights_configured": configured}

    @app.get("/api/catalogue")
    def catalogue() -> dict:
        return load_map()

    @app.get("/api/exploration/catalogue")
    def exploration_catalogue() -> dict:
        from .exploration_catalog import exploration_catalog
        return exploration_catalog()

    @app.get("/api/exploration")
    def exploration(run: str, stem: str | None = None, overview: bool = True) -> dict:
        from .exploration_data import load_exploration
        try:
            return load_exploration(run, stem, overview=overview)
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from None

    @app.get("/api/exploration/overview")
    def exploration_overview(run: str, stem: str) -> dict:
        from .exploration_data import load_overview
        try:
            return load_overview(run, stem)
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from None

    @app.get("/api/exploration/cell")
    def exploration_cell(run: str, stem: str, identity: int) -> dict:
        from .exploration_data import load_cell
        try:
            return load_cell(run, stem, identity)
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from None

    @app.get('/api/exploration/comparison-options')
    def comparison_options() -> dict:
        from .exploration_comparison import _colour
        from ..figure_tables.comparison import VIEWS, DEFAULTS
        from ..measure.contrasts import TESTS, CORRECTIONS
        from .. import workbench
        return dict(views=[dict(key=k, label=v) for k, v in VIEWS.items()], defaults=DEFAULTS,
                    tests=[dict(key=k, **v) for k, v in TESTS.items()], corrections=list(CORRECTIONS),
                    normalizations=[{**v, 'key': k} for k, v in workbench.NORMALIZATION_METHODS.items()],
                    detrends=list(workbench.DETREND_METHODS), filters=workbench.filter_methods(),
                    colours=[_colour(i) for i in range(6)])

    @app.get('/api/exploration/comparison-workspace')
    def comparison_workspace(run: str) -> dict:
        from .exploration_comparison import workspace
        try:
            return workspace(run)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from None

    @app.put('/api/exploration/comparison-workspace')
    def save_comparison_workspace(payload: dict[str, Any]) -> dict:
        from .exploration_comparison import save_workspace
        try:
            return save_workspace(payload['run'], payload['design'])
        except (ValueError, OSError, KeyError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from None

    @app.post('/api/exploration/comparison-design')
    def portable_comparison_design(payload: dict[str, Any]) -> dict:
        from .exploration_comparison import _paths, validate_design
        from .exploration_data import manifest
        try:
            root, _ = manifest(payload['run'])
            design = validate_design(_paths(payload['design'], root))
            return _paths(design, root, relative=True)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from None

    @app.post('/api/exploration/comparison-recordings')
    def add_comparison_recordings(payload: dict[str, Any]) -> dict:
        from .exploration_comparison import add_runs
        try:
            runs = payload['runs'] if 'runs' in payload else [payload['run']]
            return add_runs(payload['design'], runs)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from None

    @app.post('/api/exploration/comparison-selection')
    def comparison_selection(payload: dict[str, Any]) -> dict:
        from .exploration_comparison import selection_preview
        try:
            return selection_preview(payload)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from None

    @app.post('/api/exploration/comparison-assign')
    def comparison_assign(payload: dict[str, Any]) -> dict:
        from .exploration_comparison import assign_condition
        try:
            return assign_condition(payload)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from None

    @app.post('/api/exploration/comparison-preview')
    def comparison_preview(payload: dict[str, Any]) -> dict:
        from .exploration_comparison import render
        try:
            rendered = render(payload)
            return {**rendered, 'image': rendered['image'].decode('utf-8')}
        except (ValueError, OSError, KeyError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from None

    @app.post('/api/exploration/comparison-export')
    def comparison_export(payload: dict[str, Any]) -> Response:
        from .exploration_comparison import render
        try:
            format = payload.get('format', 'svg')
            rendered = render({k: v for k, v in payload.items() if k != 'format'}, format=format)
            return Response(rendered['image'], media_type='image/svg+xml' if format == 'svg' else 'image/png',
                            headers={'Content-Disposition': f'attachment; filename="comparison.{format}"'})
        except (ValueError, OSError, KeyError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from None

    @app.post('/api/exploration/comparison-jobs', status_code=202)
    def comparison_jobs(payload: dict[str, Any]) -> dict:
        try:
            return manager.submit_comparison(payload)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from None

    @app.post('/api/exploration/comparison-script')
    def comparison_script(payload: dict[str, Any]) -> Response:
        from .exploration_comparison import replay_script
        try:
            return Response(replay_script(payload), media_type='text/x-python',
                headers={'Content-Disposition':'attachment; filename="comparison.py"'})
        except (ValueError, OSError, KeyError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from None

    @app.get('/api/exploration/comparison-result')
    def comparison_result(run: str, result_id: str) -> dict:
        from .exploration_comparison import result
        try:
            return result(run, result_id)
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from None

    @app.get('/api/exploration/comparison-history')
    def comparison_history(run: str) -> dict:
        from .exploration_comparison import history
        try:
            return history(run)
        except (ValueError, OSError, KeyError) as exc:
            raise HTTPException(422, str(exc)) from None

    @app.get("/api/exploration/filter-options")
    def exploration_filter_options(run: str, stem: str) -> dict:
        from .exploration_filters import filter_catalogue
        try:
            return filter_catalogue(run, stem)
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from None

    @app.post("/api/exploration/filter-preview")
    def exploration_filter_preview(payload: dict[str, Any]) -> dict:
        from .exploration_filters import preview_filters
        try:
            return preview_filters(payload["run"], payload["stem"], payload["expression"],
                                   mode=payload.get("mode", "exclude"), unknown=payload.get("unknown", "keep"), analysis_ids=payload.get("analysis_ids"), decisions=payload.get("decisions"))
        except (ValueError, OSError, KeyError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from None

    @app.post("/api/exploration/filter-decisions")
    def exploration_filter_decisions(payload: dict[str, Any]) -> dict:
        from ..cell_display import import_decisions
        try:
            return import_decisions(payload["text"], payload["stem"], filename=payload.get("filename", "decisions.csv"))
        except (ValueError, KeyError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from None

    @app.get("/api/exploration/phases")
    def exploration_phases(run: str, stem: str, metric: str | None = None) -> dict:
        from .exploration_previews import load_phase_preview
        try:
            return load_phase_preview(run, stem, metric)
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from None

    @app.get("/api/exploration/rhythm-options")
    def exploration_rhythm_options(run: str, stem: str) -> dict:
        from .exploration_rhythms import rhythm_options
        try:
            return rhythm_options(run, stem)
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from None

    @app.get("/api/exploration/rhythms")
    def exploration_rhythms(run: str, stem: str, analysis_id: str | None = None) -> dict:
        from .exploration_data import recording
        from .exploration_results import saved_results
        try:
            root, _, movie = recording(run, stem)
            if analysis_id is not None:
                from .exploration_rhythms import load_attached_rhythms
                result = load_attached_rhythms(root, movie, analysis_id)
                if result is None:
                    raise ValueError("The selected saved rhythm analysis is unavailable or its inputs changed.")
                return result
            return saved_results(root, movie)["rhythms"]
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from None

    @app.post("/api/exploration/rhythm-jobs", status_code=202)
    def submit_exploration_rhythms(payload: dict[str, Any]) -> dict:
        if not isinstance(payload.get("run"), str) or not isinstance(payload.get("stem"), str):
            raise HTTPException(422, "Choose a saved recording for rhythm analysis.")
        try:
            return manager.submit_exploration_rhythms(payload["run"], payload["stem"], payload.get("options", {}))
        except (ValueError, OSError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from None

    @app.post("/api/exploration/rhythm-batches", status_code=202)
    def submit_exploration_rhythm_batch(payload: dict[str, Any]) -> dict:
        if not isinstance(payload.get("run"), str):
            raise HTTPException(422, "Choose the owning comparison workspace.")
        try:
            return manager.submit_exploration_rhythm_batch(
                payload["run"], payload.get("recordings"), payload.get("options", {}))
        except (ValueError, OSError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from None

    @app.get("/api/exploration/rhythm-batches/{identifier}")
    def exploration_rhythm_batch(identifier: str, run: str) -> dict:
        try:
            return manager.rhythm_batch(run, identifier)
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from None

    @app.post("/api/exploration/rhythm-batches/{identifier}/resume", status_code=202)
    def resume_exploration_rhythm_batch(identifier: str, payload: dict[str, Any]) -> dict:
        if not isinstance(payload.get("run"), str):
            raise HTTPException(422, "Choose the owning comparison workspace.")
        try:
            return manager.resume_rhythm_batch(payload["run"], identifier)
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from None

    @app.get("/api/exploration/rhythm-history")
    def exploration_rhythm_history(run: str, stem: str) -> dict:
        from .exploration_rhythms import rhythm_history
        try:
            return rhythm_history(run, stem)
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from None

    @app.get("/api/exploration/audit-options")
    def exploration_audit_options(run: str, stem: str) -> dict:
        from .exploration_audit import audit_options
        try:
            return audit_options(run, stem)
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from None

    @app.post("/api/exploration/audit-jobs", status_code=202)
    def exploration_audit_jobs(payload: dict[str, Any]) -> dict:
        try:
            return manager.submit_rhythm_audit(payload["run"], payload["stem"], payload["options"])
        except (ValueError, OSError, KeyError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from None

    @app.post("/api/exploration/audit-jobs/{identifier}/cancel")
    def exploration_cancel_audit(identifier: str) -> dict:
        try:
            return manager.cancel_rhythm_audit(identifier)
        except KeyError:
            raise HTTPException(404, "Unknown audit job") from None
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None

    @app.get("/api/exploration/audit-history")
    def exploration_audit_history(run: str, stem: str) -> dict:
        from .exploration_audit import audit_history
        try:
            return audit_history(run, stem)
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from None

    @app.get("/api/exploration/audit")
    def exploration_audit(run: str, stem: str, audit_id: str) -> dict:
        from .exploration_audit import audit_files, browser_result, load_audit
        try:
            result = load_audit(run, stem, audit_id)
            folder, files = audit_files(run, stem, audit_id, result=result)
            return {**browser_result(result), "artifacts": artifacts.select_files(files, folder)}
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from None

    @app.get("/api/exploration/rhythm-traces")
    def exploration_saved_traces(run: str, stem: str, metric: str, analysis_id: str, identities: str) -> dict:
        from .exploration_rhythms import load_rhythm_traces
        try:
            return load_rhythm_traces(run, stem, metric, analysis_id, identities)
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from None

    @app.get("/api/exploration/contacts")
    def exploration_contacts(run: str, stem: str, frame: int) -> dict:
        from .exploration_previews import load_contact_preview
        try:
            return load_contact_preview(run, stem, frame)
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from None

    @app.get("/api/exploration/tissue")
    def exploration_tissue(run: str, stem: str, layer: str = "first_coverage", metric: str | None = None, identities: str | None = None, analysis_id: str | None = None) -> dict:
        from .exploration_context import load_tissue_map
        try:
            return load_tissue_map(run, stem, layer, metric, identities=identities, analysis_id=analysis_id)
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from None

    @app.get("/api/exploration/timeline")
    def exploration_timeline(run: str, stem: str, order: str = "start") -> dict:
        from .exploration_context import load_track_timeline
        try:
            return load_track_timeline(run, stem, order)
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from None

    @app.get("/api/exploration/values")
    def exploration_values(run: str, stem: str, frame: int) -> dict:
        from .exploration_data import load_frame_values
        try:
            return load_frame_values(run, stem, frame)
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from None

    @app.get("/api/exploration/traces")
    def exploration_traces(run: str, stem: str, metric: str,
                           offset: int = 0, limit: int = 128, metrics: str = "") -> dict:
        from .exploration_data import load_trace_grid
        try:
            return load_trace_grid(run, stem, metric, offset, limit,
                                   metrics=metrics.split(",") if metrics else None)
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from None

    @app.get("/api/exploration/rhythm-fits")
    def exploration_rhythm_fits(run: str, stem: str, metric: str, analysis_id: str | None = None) -> dict:
        from .exploration_fits import load_rhythm_fits
        try:
            return load_rhythm_fits(run, stem, metric, analysis_id)
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from None

    @app.get("/api/exploration/barcode-order")
    def exploration_barcode_order(run: str, stem: str, metric: str,
                                  engine: str | None = None,
                                  time_mean: int = 1, scale: str = "cell", identities: str | None = None) -> dict:
        from .exploration_ordering import load_barcode_order
        try:
            options = {"time_mean": time_mean, "scale": scale}
            if identities is not None:
                options["identities"] = identities
            if engine is not None:
                options["engine"] = engine
            return load_barcode_order(run, stem, metric, **options)
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from None

    @app.get("/api/exploration/frame")
    def exploration_frame(run: str, stem: str, frame: int) -> dict:
        from .exploration_images import load_frame
        try:
            return load_frame(run, stem, frame)
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from None

    @app.get("/api/exploration/luts")
    def exploration_luts() -> dict:
        from .exploration_images import image_luts
        return image_luts()

    @app.get("/api/exploration/frame-image")
    def exploration_frame_image(run: str, stem: str, frame: int, preview: bool = False) -> dict:
        from .exploration_images import load_frame_image
        try:
            return load_frame_image(run, stem, frame, preview=preview)
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from None

    @app.get("/api/exploration/cell-images")
    def exploration_cell_images(run: str, stem: str, identity: int) -> dict:
        from .exploration_images import load_cell_images
        try:
            return load_cell_images(run, stem, identity)
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from None

    @app.get("/api/exploration/change")
    def exploration_change(run: str, stem: str, identity: int, frame: int) -> dict:
        from .exploration_images import load_change
        try:
            return load_change(run, stem, identity, frame)
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from None

    @app.get("/api/exploration/cell-frame")
    def exploration_cell_frame(run: str, stem: str, identity: int, frame: int) -> dict:
        from .exploration_images import load_cell_frame
        try:
            return load_cell_frame(run, stem, identity, frame)
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from None

    @app.get("/api/exploration/history")
    def exploration_history(run: str, stem: str, identity: int, frame: int = 0) -> dict:
        from .exploration_images import load_history
        try:
            return load_history(run, stem, identity, frame)
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from None

    @app.get("/api/events")
    def events(run: str, stem: str | None = None) -> dict:
        from .events import load_events
        try:
            return load_events(run, stem, artifacts)
        except (OSError, ValueError, KeyError) as exc:
            raise HTTPException(422, str(exc)) from None

    @app.get("/api/events/frame")
    def event_frame(run: str, stem: str, event_id: str, frame: int, event_hours: float = 6,
                    crop_px: int = 96, cell_lut: str = "red") -> dict:
        from .events import load_event_frame
        try:
            return load_event_frame(run, stem, event_id, frame, event_hours=event_hours,
                                    crop_px=crop_px, cell_lut=cell_lut)
        except (OSError, ValueError, KeyError) as exc:
            raise HTTPException(422, str(exc)) from None

    @app.get("/api/events/previews")
    def event_previews(run: str, stem: str, event_ids: str, crop_px: int = 96, cell_lut: str = "red") -> dict:
        from .events import load_event_previews
        try:
            return load_event_previews(run, stem, event_ids.split(','), crop_px=crop_px, cell_lut=cell_lut)
        except (OSError, ValueError, KeyError) as exc:
            raise HTTPException(422, str(exc)) from None

    @app.post("/api/events/clips", status_code=202)
    def event_clip(payload: dict[str, Any]) -> dict:
        try:
            if not isinstance(payload.get("options", {}), dict):
                raise ValueError("Event display settings must be an object")
            return manager.submit_event_clip(str(payload.get("run") or ""), str(payload.get("stem") or ""),
                                             str(payload.get("event_id") or ""), payload.get("options"))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise HTTPException(422, str(exc)) from None

    @app.get("/api/actions")
    def actions() -> dict:
        from pymicroglia.knowledge import describe
        return describe()

    @app.get("/api/actions/{name}")
    def action(name: str) -> dict:
        from pymicroglia.knowledge import describe
        data = describe(name)
        if not data.get("ok"):
            raise HTTPException(404, data.get("message", "Unknown action"))
        if name == 'measure':
            for parameter in data.get('params', []):
                if parameter['name'] == 'enabled_modules':
                    parameter['item_choices'] = parameter.get('choices', [])
        return data

    @app.get("/api/pipelines")
    def pipelines() -> list[dict]:
        from pymicroglia.pipelines import describe
        return describe()

    @app.get("/api/pipelines/{name}")
    def pipeline(name: str) -> dict:
        from .pipeline_contract import schema
        try:
            return schema(name)
        except (KeyError, ImportError):
            raise HTTPException(404, "Unknown or unavailable pipeline") from None

    @app.post("/api/pipelines/validate")
    def validate_pipeline(payload: dict[str, Any]) -> dict:
        from .pipeline_contract import validate
        name, params = payload.get("action"), payload.get("params")
        if not isinstance(name, str) or not isinstance(params, dict):
            raise HTTPException(422, "Choose a workflow and settings")
        try:
            return validate(name, params)
        except (KeyError, ImportError):
            raise HTTPException(404, "Unknown or unavailable pipeline") from None

    @app.post("/api/pipeline-jobs", status_code=202)
    def submit_pipeline(payload: dict[str, Any]) -> dict:
        name, params = payload.get("action"), payload.get("params")
        if not isinstance(name, str) or not isinstance(params, dict):
            raise HTTPException(422, "Choose a workflow and settings")
        try:
            return manager.submit_pipeline(name, params, str(payload.get("claim") or ""))
        except (KeyError, ImportError):
            raise HTTPException(404, "Unknown or unavailable pipeline") from None
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None

    @app.post("/api/validate")
    def validate(payload: dict[str, Any]) -> dict:
        from pymicroglia.knowledge import validate as check
        name = payload.get("action")
        params = payload.get("params", {})
        if not isinstance(name, str) or not isinstance(params, dict):
            raise HTTPException(422, "action and params are required")
        return check(name, params)

    @app.get("/api/jobs")
    def jobs_list() -> list[dict]:
        return manager.list()

    @app.get("/api/jobs/{identifier}")
    def job(identifier: str) -> dict:
        try:
            return manager.get(identifier)
        except KeyError:
            raise HTTPException(404, "Unknown job") from None

    @app.post("/api/jobs", status_code=202)
    def submit(payload: dict[str, Any]) -> dict:
        name, params = payload.get("action"), payload.get("params", {})
        if not isinstance(name, str) or not isinstance(params, dict):
            raise HTTPException(422, "action and params are required")
        try:
            return manager.submit(name, params, str(payload.get("claim") or ""),
                                  str(payload.get("request") or ""))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None

    @app.post("/api/refinements", status_code=202)
    def submit_refinement(payload: dict[str, Any]) -> dict:
        try:
            return manager.submit_refinement(
                str(payload.get("manifest") or ""),
                str(payload.get("method") or "single_frame"),
                payload.get("options") or {}, payload.get("weights") or None)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(422, str(exc)) from None

    @app.get("/api/runs")
    def runs(limit: int = 30) -> dict:
        return {"runs": recent_runs(limit)}

    @app.get("/api/runs/{run_id}")
    def run_result(run_id: str) -> dict:
        found = saved_record(run_id)
        if found is None:
            raise HTTPException(404, "Saved run not found")
        return {"result": found, "artifacts": artifacts.select_record(found.get("record") or {})}

    @app.post("/api/results/select")
    def select_results(payload: dict[str, Any]) -> dict:
        folder = payload.get("folder")
        if not isinstance(folder, str) or not folder.strip():
            raise HTTPException(422, "Choose a saved results folder")
        try:
            return artifacts.select_folder(folder)
        except (OSError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from None

    @app.get("/api/quicklook")
    def quicklook(run: str) -> dict:
        from .quicklook import load_quicklook

        try:
            return load_quicklook(run)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(422, str(exc)) from None

    @app.get("/api/availability/{figure}")
    def availability(figure: str, run: str, stem: str | None = None, item: str | None = None) -> dict:
        from pymicroglia.visualisation.figures import load, view_availability
        if figure not in load():
            raise HTTPException(404, "Unknown figure")
        if not Path(run).is_dir():
            raise HTTPException(422, "Choose a saved measurement run folder")
        try:
            return {"figure": figure, "views": view_availability(figure, run, stem=stem, item=item)}
        except (OSError, ValueError, KeyError) as exc:
            raise HTTPException(422, str(exc)) from None

    @app.get("/api/artifacts/{token}")
    def artifact(token: str):
        try:
            file, media = artifacts.get(token)
        except KeyError:
            raise HTTPException(404, "Artifact is not in the selected results") from None
        if file.suffix.lower() in {".tif", ".tiff"}:
            # TIFF stacks are never sent whole to an inline browser preview.
            return FileResponse(file, media_type="application/octet-stream", filename=file.name)
        return FileResponse(file, media_type=media)

    @app.api_route("/api/motion/{path:path}", methods=["GET", "POST"])
    async def motion_proxy(path: str, request: Request):
        if not motion_url:
            raise HTTPException(503, "Motion service is not connected")
        allowed = {"actions", "sources", "source", "frame", "action", "handoff"}
        if path not in allowed and not (path.startswith("actions/") and "/" not in path[8:]):
            raise HTTPException(404, "Unknown Motion endpoint")
        if request.method == "POST" and path not in {"action", "handoff"}:
            raise HTTPException(405, "This Motion endpoint is read only")
        query = request.url.query
        upstream = motion_url.rstrip("/") + "/api/" + path + ("?" + query if query else "")
        body = await request.body() if request.method == "POST" else None
        def fetch():
            req = UrlRequest(upstream, data=body, method=request.method,
                             headers={"Content-Type": "application/json"})
            try:
                with urlopen(req, timeout=120) as response:
                    return response.status, response.headers.get("Content-Type", "application/json"), response.read()
            except HTTPError as error:
                return error.code, error.headers.get("Content-Type", "application/json"), error.read()
        try:
            status, media, content = await asyncio.to_thread(fetch)
        except URLError as error:
            raise HTTPException(503, f"Motion service did not respond: {error.reason}") from None
        return Response(content=content, status_code=status, media_type=media)

    def review_call(function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (ValueError, OSError, KeyError) as exc:
            raise HTTPException(422, str(exc)) from None

    @app.get("/api/trace-review")
    def trace_review(home: str) -> dict:
        from .trace_review import list_variants
        return review_call(list_variants, home)

    @app.post("/api/trace-review/create")
    def trace_review_create(payload: dict[str, Any]) -> dict:
        from .trace_review import create_review, default_home, review_for_run
        run, stems = payload.get("run"), payload.get("stems")
        if not isinstance(run, str):
            raise HTTPException(422, "Choose a saved measurement run")
        if stems:
            home = payload.get("home") or str(review_call(default_home, run))
            review = review_call(create_review, home, [{"run": run, "stem": s} for s in stems], title=payload.get("title"))
        else:
            home, review = review_call(review_for_run, run, title=payload.get("title"))
            if payload.get("home"):
                home = payload["home"]
                review = review_call(create_review, home, review["recordings"], title=payload.get("title"))
        return {"home": str(Path(home).resolve()), "review": review}

    @app.get("/api/trace-review/grid")
    def trace_review_grid(home: str, variants: str, metrics: str | None = None) -> dict:
        from .trace_review import load_grid
        return review_call(load_grid, home, [v for v in variants.split(",") if v], metrics)

    @app.get("/api/trace-review/decisions")
    def trace_review_decisions(home: str) -> dict:
        from .trace_review import load_decisions
        return review_call(load_decisions, home)

    @app.put("/api/trace-review/decisions")
    def trace_review_save(payload: dict[str, Any]) -> dict:
        from .trace_review import save_decisions
        extra = {"preferred_variant": payload["preferred_variant"]} if "preferred_variant" in payload else {}
        result = review_call(save_decisions, str(payload.get("home") or ""), payload.get("changes") or [],
                             revision=payload.get("revision"), **extra)
        return result

    @app.post("/api/trace-review/handoff")
    def trace_review_handoff(payload: dict[str, Any]) -> dict:
        from .trace_review import handoff
        record = review_call(handoff, str(payload.get("home") or ""), note=str(payload.get("note") or ""),
                             variant=payload.get("variant"), metric=payload.get("metric"))
        return {key: record[key] for key in ("created_at", "counts", "variant", "decisions_revision")}

    @app.post("/api/trace-review/import")
    def trace_review_import(payload: dict[str, Any]) -> dict:
        from .trace_review import import_legacy
        return review_call(import_legacy, str(payload.get("home") or ""), str(payload.get("folder") or ""),
                           label=payload.get("label"))

    @app.get("/api/trace-review/crop")
    def trace_review_crop(run: str, stem: str, identity: int, frame: int, size: int = 72) -> dict:
        from .trace_review import cell_crop
        return review_call(cell_crop, run, stem, identity, frame, size=size)

    def intake_call(function, *args):
        try:
            return function(*args)
        except (OSError, ValueError, KeyError) as exc:
            raise HTTPException(422, str(exc)) from None

    @app.post("/api/intake/inspect")
    def intake_inspect(payload: dict[str, Any]) -> dict:
        from .intake import inspect
        paths = payload.get("paths")
        if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
            raise HTTPException(422, "Give the dropped file or folder paths")
        return intake_call(inspect, paths)

    @app.post("/api/intake/labels")
    def intake_labels(payload: dict[str, Any]) -> dict:
        from .intake import save_labels
        return intake_call(save_labels, str(payload.get("folder") or ""), payload.get("labels"))

    @app.post("/api/intake/run", status_code=202)
    def intake_run(payload: dict[str, Any]) -> dict:
        recordings = payload.get("recordings")
        if not isinstance(recordings, list):
            raise HTTPException(422, "Choose the recordings to analyse")
        return intake_call(manager.submit_intake, str(payload.get("folder") or ""),
                           str(payload.get("mode") or ""), recordings,
                           payload.get("settings"))

    @app.get("/api/intake/results")
    def intake_results(folder: str) -> dict:
        from .intake import results
        return {"results": intake_call(results, folder)}

    @app.post("/api/jobs/{identifier}/stop")
    def stop_job(identifier: str) -> dict:
        try:
            return manager.stop(identifier)
        except KeyError:
            raise HTTPException(404, "Unknown job") from None

    # Agent-correction review (port of Motion's correction reviewer): correction_review.py.
    from . import correction_review as _correction_review
    _correction_review.PREWARM = True

    def correction_call(function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (ValueError, OSError, KeyError, StopIteration, IndexError) as exc:
            raise HTTPException(422, str(exc) or "Unknown review item") from None

    @app.get("/api/correction-review")
    def correction_reviews(root: str) -> dict:
        from .correction_review import list_reviews
        return {"root": root, "reviews": correction_call(list_reviews, root)}

    @app.post("/api/correction-review/create")
    def correction_create(payload: dict[str, Any]) -> dict:
        from .correction_review import create_review
        return correction_call(create_review, str(payload.get("root") or ""), str(payload.get("source") or ""),
                               agent_session=payload.get("agent_session"), note=str(payload.get("note") or ""))

    @app.get("/api/correction-review/set")
    def correction_set(root: str, stem: str) -> dict:
        from .correction_review import load
        return correction_call(load, root, stem)

    @app.get("/api/correction-review/item")
    def correction_item(root: str, stem: str, key: str) -> dict:
        from .correction_review import item_view
        return correction_call(item_view, root, stem, key)

    @app.get("/api/correction-review/frame")
    def correction_frame(root: str, stem: str, key: str, frame: int, k: int = 3, prefetch: str = "",
                         channels: str = "", follow: str = "", every: bool = False) -> dict:
        """One frame of a join: Before / After / Raw / Changes (and shown channels) at zoom k, the map, the numbers.
        follow: cell numbers drawn in their own colours; every: colour every cell."""
        from .correction_review import frame_view
        return correction_call(frame_view, root, stem, key, frame, k, [p for p in prefetch.split(",") if p],
                               [c for c in channels.split(",") if c], [int(n) for n in follow.split(",") if n.strip()],
                               every)

    @app.get("/api/correction-review/cells")
    def correction_cells(root: str, stem: str) -> dict:
        """Every cell's frames and flagged events now (the reviewer's Gantt chart)."""
        from .correction_review import cells_view
        return correction_call(cells_view, root, stem)

    @app.get("/api/correction-review/tower")
    def correction_tower(root: str, stem: str, cells: str, lo: str = "", hi: str = "", style: str = "line",
                         preset: str = "cd68") -> dict:
        """The package tracking tower of the followed cells, as an image."""
        from .correction_review import tower_view
        return correction_call(tower_view, root, stem, [int(c) for c in cells.split(",") if c.strip()],
                               int(lo) if lo.strip() else None, int(hi) if hi.strip() else None, style, preset)

    @app.get("/api/correction-review/grow-preview")
    def correction_grow_preview(root: str, stem: str, frame: int, y: int, x: int, level: float = 0.4,
                                kind: str = "seed", y0: int = 0, y1: int = 0, x0: int = 0, x1: int = 0) -> dict:
        """Pixels Seed / Grow would take at a spot and grow level (nothing saved)."""
        from .correction_review import grow_preview
        return correction_call(grow_preview, root, stem, frame, y, x, level, kind, (y0, y1, x0, x1) if y1 > y0 else None)

    @app.get("/api/correction-review/pick")
    def correction_pick(root: str, stem: str, key: str, frame: int, y: int, x: int, panel: str = "Before") -> dict:
        from .correction_review import pick
        return correction_call(pick, root, stem, key, frame, y, x, panel)

    @app.post("/api/correction-review/decide")
    def correction_decide(payload: dict[str, Any]) -> dict:
        from .correction_review import decide
        return correction_call(decide, str(payload.get("root") or ""), str(payload.get("stem") or ""),
                               str(payload.get("key") or ""), payload.get("value"))

    @app.post("/api/correction-review/edit")
    def correction_edit(payload: dict[str, Any]) -> dict:
        from .correction_review import edit
        fields = ("action", "picked", "frame", "to_frame", "key", "other", "number", "piece_only", "points", "whole",
                  "level", "range_from", "range_to", "flip", "cells", "one_piece")
        return correction_call(edit, str(payload.get("root") or ""), str(payload.get("stem") or ""),
                               **{k: payload.get(k) for k in fields if k in payload})

    @app.get("/api/correction-review/marks")
    def correction_marks(root: str, stem: str) -> dict:
        """Jamie's own ticks and flags for a recording."""
        from .correction_review import marks
        return correction_call(marks, root, stem)

    @app.post("/api/correction-review/mark")
    def correction_mark(payload: dict[str, Any]) -> dict:
        """Tick / untick a cell, flag one at a frame, change a flag's note, remove a flag."""
        from .correction_review import mark
        return correction_call(mark, str(payload.get("root") or ""), str(payload.get("stem") or ""),
                               **{k: payload.get(k) for k in ("op", "cell", "frame", "id", "note") if k in payload})

    @app.post("/api/correction-review/split-preview")
    def correction_split_preview(payload: dict[str, Any]) -> dict:
        """Which side of a drawn split line keeps the number (nothing saved)."""
        from .correction_review import split_preview
        return correction_call(split_preview, str(payload.get("root") or ""), str(payload.get("stem") or ""),
                               int(payload.get("frame") or 0), payload.get("points") or [], payload.get("flip"),
                               payload.get("number"))

    @app.get("/api/correction-review/cells-in-box")
    def correction_cells_in_box(root: str, stem: str, frame: int, y0: int, y1: int, x0: int, x1: int) -> dict:
        from .correction_review import cells_in_box
        return {"cells": correction_call(cells_in_box, root, stem, frame, y0, y1, x0, x1)}

    @app.post("/api/correction-review/edit-batch")
    def correction_edit_batch(payload: dict[str, Any]) -> dict:
        """Queued edits, applied in order (each spot looked up again after the edits before it)."""
        from .correction_review import edit_batch
        return correction_call(edit_batch, str(payload.get("root") or ""), str(payload.get("stem") or ""),
                               payload.get("items") or [], payload.get("key"))

    @app.post("/api/correction-review/undo-edit")
    def correction_undo_edit(payload: dict[str, Any]) -> dict:
        from .correction_review import undo_edit
        return correction_call(undo_edit, str(payload.get("root") or ""), str(payload.get("stem") or ""),
                               int(payload.get("count") or 0))

    @app.post("/api/correction-review/note")
    def correction_note(payload: dict[str, Any]) -> dict:
        from .correction_review import add_note
        return correction_call(add_note, str(payload.get("root") or ""), str(payload.get("stem") or ""),
                               key=payload.get("key"), frame=int(payload.get("frame") or 0),
                               text=str(payload.get("text") or ""), tag=payload.get("tag"), picked=payload.get("picked"))

    @app.put("/api/correction-review/set")
    def correction_save(payload: dict[str, Any]) -> dict:
        from .correction_review import save
        return correction_call(save, str(payload.get("root") or ""), str(payload.get("stem") or ""),
                               decisions=payload.get("decisions"), notes=payload.get("notes"),
                               edits=payload.get("edits"))

    @app.post("/api/correction-review/focus")
    def correction_focus(payload: dict[str, Any]) -> dict:
        """What the page shows now (recording, join, frame, clicked cell), for the page's assistant."""
        keys = ("root", "stem", "key", "frame", "cell")
        focus = getattr(app.state, "correction_focus", None)
        if focus is not None:
            focus.clear()
            focus.update({k: payload[k] for k in keys if payload.get(k) not in (None, "")})
        return {"ok": True}

    @app.get("/api/correction-review/stamp")
    def correction_stamp(root: str, stem: str) -> dict:
        from .correction_review import stamp
        return {"stamp": correction_call(stamp, root, stem)}

    @app.post("/api/correction-review/save-truth")
    def correction_save_truth(payload: dict[str, Any]) -> dict:
        """Export to Motion (if changed) and save a dated ground-truth snapshot of the recording."""
        from .correction_truth import save_truth
        return correction_call(save_truth, str(payload.get("root") or ""), str(payload.get("stem") or ""),
                               app.state.motion_url, user=str(payload.get("user") or "jamie"))

    @app.post("/api/correction-review/render")
    def correction_render(payload: dict[str, Any]) -> dict:
        """Start the corrected movie (correction_outputs.render_movie); it runs in the background."""
        from .correction_outputs import render_movie
        return correction_call(render_movie, str(payload.get("root") or ""), str(payload.get("stem") or ""))

    @app.get("/api/correction-review/render-status")
    def correction_render_status(root: str, stem: str) -> dict:
        from .correction_outputs import render_status
        return correction_call(render_status, root, stem)

    @app.get("/api/correction-review/movie")
    def correction_movie(root: str, stem: str):
        from .correction_outputs import movie_file
        return FileResponse(correction_call(movie_file, root, stem), media_type="video/mp4")

    @app.get("/api/correction-review/traces")
    def correction_traces(root: str, stem: str) -> dict:
        from .correction_outputs import traces
        return correction_call(traces, root, stem)

    @app.post("/api/correction-review/export")
    def correction_export(payload: dict[str, Any]) -> dict:
        from .correction_review import export
        return correction_call(export, str(payload.get("root") or ""), str(payload.get("stem") or ""),
                               app.state.motion_url, user=str(payload.get("user") or "jamie"))

    if STATIC.is_dir():
        app.mount("/assets", StaticFiles(directory=STATIC), name="assets")

        @app.get("/correction-review")
        def correction_page():
            return FileResponse(STATIC / "correction-review.html")

        @app.get("/")
        def index():
            return FileResponse(STATIC / "index.html")

        @app.get("/review")
        def review_page():
            return FileResponse(STATIC / "trace-review.html")

    # Optional dependency stays out of the scientific package and is mounted only
    # after the app's routes, middleware and live job manager exist.
    from importlib.util import find_spec
    import os
    if os.environ.get("PYMICROGLIA_AI", "1") != "0" and find_spec("aiify") is not None:
        from pymicroglia.ai_assistant import install
        from .correction_agent import install as install_corrections
        install(app)
        install_corrections(app)
    return app
