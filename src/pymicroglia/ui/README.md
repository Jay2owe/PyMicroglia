# PyMicroglia workbench UI

Local FastAPI app (`server.py`) serving the workbench (`/`) and the trace
reviewer (`/review`).

## Analyse recordings — drop in, label, run everything

The first page (`intake.py`, `static/intake.{js,css}`). Drop a TIFF stack or a
folder on the window (desktop window: real paths via
`auto_organotypic.desktop_drop`; browser: paste a path), edit each recording's
name, condition, treatment, sample and notes, then run `auto_microglia` as a
`kind="intake"` job with chain-stage progress and Stop. Labels live in the
folder's `recording_labels.json`. Drops on the Auto-Organotypic tab are
forwarded to that page instead. Routes: `/api/intake/{inspect,labels,run,results}`,
`/api/jobs/{id}/stop`. Tests: `tests/test_intake.py`.

## Trace review — the one canonical way to review cell traces

Status: working draft, verified on synthetic recordings and one imported real
legacy grid (95_A3); not yet used on a live review.

Jamie reviews every cell's trace and rhythm fit in one grid, marks cells
include / exclude / representative (star), and hands the decisions off.
Agents produce analyses and read the handoff; they never write Jamie's marks.

```
agent: run_rhythm_analysis (label, window)  ──► AI_Exports/exploration_rhythms/<rec>/<id>/
agent: import_legacy (old figure folder)     ──► <review>/imported/<id>/
                       │
Jamie: /review grid ── sort, filter, overlay, crop video ──► decisions.json (+ decisions_log.jsonl)
                       │ Hand off
agent: read_handoff / approved_cells  ◄── handoff.json + included_cells.csv
```

- **Variant** = saved analyses sharing settings (method, test, detrend,
  normalisation, period range, time window, label). Variants on the same time
  window can be overlaid, as can several metrics; different windows are only
  switched between, never overlaid.
- **Decisions** are bound to each recording's `cell_frame.csv` hash. If
  measurements change after a handoff, `read_handoff` lists the recording as
  stale and `approved_cells` refuses to return cells.
- **Agent suggestions** (`suggest`) show as a dashed "→in/→out" tag; Jamie
  accepts them in bulk or ignores them.

### Registered agent actions

`trace_review_create` (run → review), `trace_review_variant` (one rhythm
variant over every recording; needs a claim saying why), `trace_review_import`,
`trace_review_suggest`, `trace_review_handoff`. Defined in
`tools/regenerate_catalogue.py` (`_TRACE_REVIEW_ACTIONS`); regenerate
`data/actions.json` after changing them.

### Commands

```bash
python -m pymicroglia.ui.trace_review open <run-or-review-folder>   # start and open the reviewer
python -m pymicroglia.ui.trace_review create <run> [--stem S ...] [--home DIR]
python -m pymicroglia.ui.trace_review variants <review>
python -m pymicroglia.ui.trace_review import <review> <legacy_figure_folder> [--label L]
python -m pymicroglia.ui.trace_review suggest <review> suggestions.csv --agent NAME   # cell,status,reason
python -m pymicroglia.ui.trace_review handoff <review>
```

New variants: `run_rhythm_analysis(run, stem, {..., "label": "24-96 h", "window_start_hours": 24, "window_end_hours": 96})`
in `exploration_rhythms.py`, or the existing rhythm-batch endpoint. Do not make
trace grids any other way for review.

Default review folder for a run: `<run>/AI_Exports/trace_review/`.

Code: `trace_review.py`, `static/trace-review.{html,js,css}`; tests:
`tests/test_trace_review.py`.
