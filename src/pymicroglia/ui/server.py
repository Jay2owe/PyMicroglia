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

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .catalog import load_map
from .jobs import JobManager
from .results import ArtifactIndex, recent_runs, saved_record

STATIC = Path(__file__).with_name("static")


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

    @app.get("/api/config")
    def config() -> dict:
        configured = os.environ.get("PYMICROGLIA_MASK_WEIGHTS", "")
        return {"auto_url": app.state.auto_url, "motion_url": app.state.motion_url,
                "mask_weights_configured": bool(configured and Path(configured).is_file())}

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

    if STATIC.is_dir():
        app.mount("/assets", StaticFiles(directory=STATIC), name="assets")

        @app.get("/")
        def index():
            return FileResponse(STATIC / "index.html")

    # Optional dependency stays out of the scientific package and is mounted only
    # after the app's routes, middleware and live job manager exist.
    from importlib.util import find_spec
    import os
    if os.environ.get("PYMICROGLIA_AI", "1") != "0" and find_spec("aiify") is not None:
        from pymicroglia.ai_assistant import install
        install(app)
    return app
