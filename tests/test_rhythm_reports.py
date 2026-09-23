"""Saved selections remain complete through reports, grids and optional images."""
from pymicroglia._results import read_document

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pymicroglia import workbench as circadian
from pymicroglia.pipelines import parse
from pymicroglia.pipelines._contracts import Settings, StepSpec, content_id
from pymicroglia.pipelines.rhythm.discovery import resolve_request, run_request
from pymicroglia.pipelines._runner import ExecutionContext
from pymicroglia.pipelines._screening import file_hash


def make_screen(tmp_path, monkeypatch, images=False, transform=None):
    import tifffile
    tmp_path.mkdir(parents=True, exist_ok=True)
    rows = []
    for movie, identity, left, right in (("a", 1, 10, 80), ("a", 2, 20, 50), ("b", 1, 30, 60)):
        for hour in range(72):
            rows.append(dict(stem=movie, identity=identity, hours=float(hour), frame_index=hour,
                imagej_frame=hour + 1, source_imagej_frame=hour + 2,
                signal=left + hour / 100 if hour not in (24, 25) else np.nan, other=right + hour / 100, empty=np.nan))
    frame = pd.DataFrame(rows)
    if transform is not None: frame = transform(frame)
    tables = {"cell_frame": frame, "cell_summary": frame[["stem", "identity"]].drop_duplicates()}
    paths = {name: tmp_path / (name + ".csv") for name in tables}
    for name, table in tables.items(): table.to_csv(paths[name], index=False)
    movies = []
    for movie in ("a", "b"):
        record = {"stem": movie}
        if images:
            labels = np.zeros((72, 16, 16), dtype=np.uint16)
            labels[:, 2:5, 3:6] = 1
            labels[:, 10:13, 9:12] = 2
            raw = np.arange(73 * 16 * 16, dtype=np.float32).reshape(73, 16, 16)
            if movie == "b": raw += 1_000_000
            inputs = {}
            for kind, values in (("labels", labels), ("raw", raw)):
                path = tmp_path / f"{movie}-{kind}.tif"
                tifffile.imwrite(path, values, photometric="minisblack")
                inputs[kind] = {"path": str(path), "sha256": file_hash(path)}
            record["provenance"] = {"inputs": inputs}
        movies.append(record)
    (tmp_path / "manifest.json").write_text(json.dumps({"movies": movies}), encoding="utf-8")
    def estimate(hours, values, params, method, **kwargs):
        value = values[0]
        return {"method": method, "status": "failed" if 50 <= value < 70 and method == "f" else "ok",
            "period_hours": 40. if value >= 70 else 12., "p_value": .001 if value < 30 or value >= 70 else .8,
            "native_series": {"fitted_fixture": {"x": [0., 1., 2.], "y": [1., 2., 1.], "x_unit": "hours", "y_unit": "fixture units"}},
            "native_result": {}, "components": [], "diagnostics": {},
            "display_processed_trace": {"hours": list(hours), "values": list(values), "value_unit": "fixture units"}}
    monkeypatch.setattr(circadian, "estimate_one", estimate)
    request = parse([{"pipeline": "rhythm-discovery", "test_measurements": ["signal", "other", "empty"],
        "analysis_options": {"fit_method": "mesa", "significance_method": "f", "detrend": "none"}}])[0]
    resolved = resolve_request(request, source_run=file_hash(tmp_path / "manifest.json"), tables=tables, input_hashes={k: file_hash(p) for k, p in paths.items()})
    execution = run_request(resolved, paths, tmp_path / "pipeline", only=("rhythm-screen",))
    assert execution.successful
    return resolved, paths, execution.results["rhythm-screen"]


def render_context(tmp_path, resolved, paths, saved, appearance=None):
    out = tmp_path / "render/renders/evidence/presentation/invocation"
    out.mkdir(parents=True, exist_ok=True)
    return ExecutionContext(StepSpec("selected-cell-evidence", "selected-cell-evidence", ("rhythm-screen",), kind="render"),
        resolved, Settings(), paths, {"rhythm-screen": saved}, None, out, "illustrative-evidence",
        Settings({"evidence": appearance or {"grid_cells_per_page": 1, "report_measurements_per_page": 1}}), content_id(appearance))


def test_every_cell_metric_and_page_uses_only_saved_results(tmp_path, monkeypatch):
    from pymicroglia.pipelines.rhythm.reports import produce
    resolved, paths, saved = make_screen(tmp_path / "run", monkeypatch)
    from pymicroglia.pipelines import _saved_figures
    captured = []
    def save(page, output, name, **kwargs):
        slug = 'rhythm-cell-report' if name.startswith('rhythm-cell-report') else 'rhythm-trace-grid'
        captured.append((slug, page))
    monkeypatch.setattr(_saved_figures, 'save_page', save)
    def forbidden(*a, **k): pytest.fail("saved evidence renderer attempted scientific analysis")
    for name in ("estimate_one", "estimate_grouped_rhythms", "filter_rhythm_trace", "adjust_pvalues"):
        monkeypatch.setattr(circadian, name, forbidden)
    context = render_context(tmp_path, resolved, paths, saved)
    result = produce(context)
    assert result.status == "completed"
    manifest = read_document(context.output / "evidence_manifest.json")
    assert manifest["logical_report_count"] == 2
    reports = [r for slug, r in captured if slug == "rhythm-cell-report"]
    grids = [r for slug, r in captured if slug == "rhythm-trace-grid"]
    assert len(reports) == 3 and len(grids) == 3
    assert all(len(r.auxiliary["status.csv"]) == 3 for r in reports)
    assert {(r.auxiliary["statistics.csv"].iloc[0].identity, r.auxiliary["statistics.csv"].iloc[0].measurement) for r in grids} == {(1, "signal"), (2, "signal"), (1, "other")}
    unresolved = next(r for r in reports if r.auxiliary["statistics.csv"].iloc[0].measurement == "other")
    assert unresolved.auxiliary["statistics.csv"].iloc[0].display_state == "significant-unresolved"
    assert "native" not in set(unresolved.figure_data.view)
    signal = next(r for r in reports if r.auxiliary["statistics.csv"].iloc[0].measurement == "signal")
    assert set(signal.figure_data.view) == {"raw", "detrended", "native"}
    raw = signal.figure_data[signal.figure_data.view.eq("raw")]
    assert raw.loc[raw.hours.isin([24, 25]), "value"].isna().all() and raw.hours.tolist() == list(range(72))
    assert all(json.loads(r.auxiliary["images.csv"].iloc[0].image_json)["status"] == "unavailable" for r in reports)
    assert next(m for m in manifest["measurements"] if m["measurement"] == "empty")["pages"] == 0
    initial_id = saved.outcome.scientific_id
    context2 = render_context(tmp_path / "second", resolved, paths, saved, {"grid_cells_per_page": 12, "trace_view": "detrended"})
    produce(context2)
    assert saved.outcome.scientific_id == initial_id
    manifest2 = read_document(context2.output / "evidence_manifest.json")
    assert manifest2["selected_cells"] == manifest["selected_cells"]
    assert len([p for p in manifest2["pages"] if p["kind"] == "grids"]) == 2


def test_snapshots_keep_movie_cell_and_source_frame_alignment(tmp_path, monkeypatch):
    from pymicroglia.pipelines.rhythm.images import prepare, IMAGE_DEFAULTS
    resolved, paths, saved = make_screen(tmp_path / "run", monkeypatch, images=True)
    ctx = render_context(tmp_path, resolved, paths, saved)
    cells = [{"source_run": resolved.inputs.source_run, "movie": m, "identity": i} for m, i in (("a", 1), ("a", 2), ("b", 1))]
    archive, inventory, _ = prepare(ctx, cells, {**IMAGE_DEFAULTS, "image_hours": [0, 71]}, ctx.output)
    records = read_document(inventory)["cells"]
    assert all(r["status"] == "available" for r in records)
    with np.load(archive, allow_pickle=False) as arrays:
        for row in records:
            assert [t["source_frame_index"] for t in row["tiles"]] == [1, 72]
            assert [t["hours"] for t in row["tiles"]] == [0, 71]
            top, _, left, _ = row["crop_box"]
            expected = 256 + top * 16 + left + (1_000_000 if row["movie"] == "b" else 0)
            assert arrays[row["archive_key"] + "_raw"][0, 0, 0] == expected
            assert arrays[row["archive_key"] + "_mask"].sum() == 18
            assert row["settings"]["display_only"]
    # A changed source cannot silently supply an image for an existing result.
    with (tmp_path / "run/a-raw.tif").open("ab") as stream: stream.write(b"changed")
    _, inventory, _ = prepare(ctx, cells, IMAGE_DEFAULTS, ctx.output / "changed")
    records = read_document(inventory)["cells"]
    assert [r["status"] for r in records] == ["unavailable", "unavailable", "available"]


def test_failed_estimator_retains_saved_significance_diagnostic(tmp_path, monkeypatch):
    from tests.test_pipeline_screening import inputs
    from pymicroglia.pipelines._screening import screen
    resolved, paths = inputs(tmp_path)
    def estimate(hours, values, params, method, **kwargs):
        if method == "mesa": return {"status": "failed", "method": method}
        return {"method": method, "status": "ok", "p_value": .001, "period_hours": 12.,
            "display_processed_trace": {"hours": list(hours), "values": list(values), "value_unit": "fixture units"}}
    monkeypatch.setattr(circadian, "estimate_one", estimate)
    saved = screen(resolved, paths, tmp_path / "screen")
    selected = saved.display_inputs[saved.display_inputs.identity.eq(7)]
    assert selected.processing_source_method.eq("f").all()
    assert selected.processed_trace.map(bool).all()
    assert saved.results[saved.results.identity.eq(7)].significant.all()


def test_flat_export_consumes_one_page_at_a_time(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from pymicroglia.figure_tables.prepared import PreparedPage, Drawing
    from pymicroglia.pipelines._saved_figures import save_page
    from pymicroglia.visualisation import panels
    saved = []
    def save(figure, path, **kwargs):
        assert len(plt.get_fignums()) == 1
        assert kwargs['table'] == {'value': [len(saved)]}
        assert kwargs['bundle'] is False
        saved.append(path.name)
    monkeypatch.setattr(panels, 'save', save)
    def draw(value, *, canvas):
        assert len(plt.get_fignums()) == 1
        ax = canvas.subplots()
        ax.plot([value])
        return canvas, (ax,)
    for i in range(3):
        assert not plt.get_fignums(), 'previous page remains open while the next is built'
        page = PreparedPage(Drawing(draw, (i,)), pd.DataFrame({'value': [i]}))
        save_page(page, tmp_path, f'page-{i}', sources=[], settings={}, claim='Synthetic streaming test')
    assert saved == ['page-0', 'page-1', 'page-2'] and not plt.get_fignums()
    assert not list(tmp_path.glob('*.zip')) and not (tmp_path / 'plot.py').exists()


def test_registered_report_reopens_pinned_snapshots(tmp_path, monkeypatch):
    from pymicroglia.pipelines.rhythm.reports import produce
    import matplotlib.pyplot as plt
    resolved, paths, saved = make_screen(tmp_path / "run", monkeypatch, images=True)
    from pymicroglia.pipelines import _saved_figures
    from pymicroglia.visualisation.figures import get_figure
    from pymicroglia.figure_tables.pipeline_inputs import SavedInputs
    from pymicroglia.figure_tables.rhythm_traces import build
    contexts = []
    def save(page, output, name, *, sources, settings, **kwargs):
        contexts.append((name, settings))
    monkeypatch.setattr(_saved_figures, 'save_page', save)
    context = render_context(tmp_path, resolved, paths, saved)
    produce(context)
    name, settings = contexts[0]
    spec = get_figure('rhythm-cell-report')
    report = SavedInputs(run=tmp_path / 'run', spec=spec, item=name,
                         options=settings, binding=settings['binding'])
    def forbidden(*a, **k): pytest.fail('reopening saved snapshots recomputed science')
    for method in ('estimate_one', 'estimate_grouped_rhythms', 'filter_rhythm_trace', 'adjust_pvalues'):
        monkeypatch.setattr(circadian, method, forbidden)
    rebuilt = build(report, 'reports')
    assert json.loads(rebuilt.auxiliary['images.csv'].iloc[0].image_json)['status'] == 'available'
    with (context.output / 'cell_tiles.npz').open('ab') as stream: stream.write(b'changed')
    with pytest.raises(ValueError, match='image assets are missing or changed'):
        build(report, 'reports')
