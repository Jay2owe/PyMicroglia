# Motion parity fixture

What `Motion/analysis` produced on one synthetic tracked movie, frozen before
the port into PyMicroglia began. Every later port stage runs
`tests/test_motion_parity.py` and points `compare_run` at the folder its
ported code writes; an empty list of differences is parity. Stage 01 of
`Auto-Organotypic/docs/consolidation/motion-port/` made it.

## Provenance

| what | value |
|---|---|
| Motion commit | `df035b0dc1ea2a9a94ae10e1b390cd1100b6b227` (working tree with uncommitted in-flight changes; the working tree is the version being ported, and `environment.motion_status_short` in `expected.json` lists what was modified) |
| Circadian Workbench | `0.8.2` (`circadian_workbench.__version__` as installed; the Motion checkout only floors it, and stage 05 bumps it deliberately) |
| frozen on | 2026-09-21, Windows 11, Python 3.12.10 |
| numpy / pandas / scipy / matplotlib | see `environment` in `expected.json` |

## What is in here

| path | what |
|---|---|
| `inputs/` | the synthetic movie, written by `tests/fixtures.py::tracked_movie(seed=0)`: labels, raw, unclaimed, evidence, provenance, a valid mask, one extra channel with a shifts table, one object set, two side tables |
| `config.json` | the Motion analysis configuration: every module enabled, two windows, one paired contrast, one metric group, the shared circadian defaults (2–48 h search, alpha 0.05, 24 observations, 3 cycles, `lomb` for both the estimator and the test, `robust_linear` detrend); every input pinned by SHA-256 |
| `run/` | the run's CSV and JSON files (tables, `statistics.csv`, manifests, summaries; never the stacks or figures), so a later stage can diff a whole table rather than three rows of it. Manifest input paths are relative to this fixture; machine-specific paths in embedded Workbench environment records were replaced with `<fixture-root>` before publication. Numerical values and scientific settings are unchanged. The affected table hash in `expected.json` was updated. `pooled/tables/` is not copied: with one movie every pooled table is byte-identical to the per-movie one, which `expected.json` records by hash |
| `expected.json` | the frozen record, sections below |
| `freeze.py` | the one-shot that wrote `expected.json` from finished output folders |

`expected.json` sections:

- `measure`: for every CSV the run wrote, its SHA-256, row count, column
  names and first and last three rows as written (nine significant figures);
  the whole `manifest.json`; the count of tables the manifest names.
- `figures`: one entry per builder (83: 71 result figures, 12 review
  figures), with the SHA-256 of every plotted-data CSV in its bundle, or the
  refusal text when the figure declined this movie. The SVG is not frozen:
  fonts and matplotlib versions move it.
- `pipelines`: the six result-dependent demos plus the state demo. For each:
  every step's status from its `execution-result.json` records, every
  artefact's hash, the `verification.json` the demo wrote, and every CSV it
  produced (hash, rows, columns, head, tail).
- `extra.states`: the tables the `states` step wrote on the run.
- `commands`: what `pool`, `window`, `contrasts`, `videos` and `cluster` did.
- `counts`: 83 figures, 12 review, 71 results, 24 registered modules, and
  the number of measured tables.
- `environment`: versions and the Motion commit.

Absolute paths inside records are scrubbed to `<scratch>`, `<pymicroglia>`,
`<motion>` or `<home>` so the record reads the same on any machine.

## How it was made

Everything ran from the Motion checkout, with `--out` folders in a scratch
directory, and the config in this folder. Relative paths in `config.json` are
resolved against the config file itself.

```bash
cd "<Motion>"
python -m analysis doctor --config "<PyMicroglia>/tests/fixtures/motion_parity/config.json"
python -m analysis run    --config "<PyMicroglia>/tests/fixtures/motion_parity/config.json" --out <scratch>/run
# run already pools, windows and tests; the standalone commands refuse the
# immutable run (recorded under commands.on_the_finished_run) and were also
# run on a stripped copy to confirm they agree (commands.on_a_stripped_copy)
python -m analysis videos <scratch>/run --identity 1 --dry-run
python -m analysis videos <scratch>/run --identity 1
python -m analysis states  <scratch>/run --out <scratch>/states  --options analysis/states.example.json
python -m analysis cluster <scratch>/run --out <scratch>/cluster --options analysis/clustering.example.json
# every builder, one subprocess each, as analysis/figures/build_all.py does,
# with stdout/stderr captured per builder into a log freeze.py reads
python -m analysis.pipelines.rhythm_demo        <scratch>/demos/rhythm --verify
python -m analysis.pipelines.relationship_demo  <scratch>/demos/relationship --verify
python -m analysis.pipelines.behaviour_demo     <scratch>/demos/behaviour
python -m analysis.pipelines.coordination_demo  --out <scratch>/demos/coordination
python -m analysis.pipelines.intervention_demo  --out <scratch>/demos/intervention
python -m analysis.state_demo <scratch>/demos/state_source --out <scratch>/demos/state
python -m analysis.pipelines.audit_demo         <scratch>/demos/audit --verify
```

Then, from PyMicroglia:

```bash
python tests/fixtures/motion_parity/freeze.py --run <scratch>/run \
    --figure-log <scratch>/figure_log.json \
    --pipeline rhythm-discovery=<scratch>/demos/rhythm \
    --pipeline method-audit=<scratch>/demos/audit \
    --pipeline measurement-relationships=<scratch>/demos/relationship \
    --pipeline behaviour-states=<scratch>/demos/behaviour \
    --pipeline spatial-coordination=<scratch>/demos/coordination \
    --pipeline intervention-response=<scratch>/demos/intervention \
    --pipeline state-analysis=<scratch>/demos/state \
    --extra states=<scratch>/states \
    --commands <scratch>/commands.json --counts <scratch>/counts.json \
    --environment <scratch>/environment.json \
    --scrub scratch=<scratch> --scrub pymicroglia=<PyMicroglia> \
    --scrub motion=<Motion> --scrub "home=C:\Users\<you>" \
    --copy-tables
python -m pytest tests/test_motion_parity.py
```

A stage that deliberately changes an output says which files and which
numbers in its commit message and refreshes the record with the same
commands in the same commit.

## What the synthetic movie cannot exercise

Recorded as the expected outcome, not dropped:

- 35 saved-display figures (55 to 86, and the review audit figures 50 to
  54) refuse with `saved pipeline inputs require a materialized figure item`:
  they draw saved pipeline results, and the measurement run holds none. Their
  drawing is exercised by the six demos, whose figure artefacts are frozen
  under `pipelines`.
- `cell-report-card` refuses because its default identity (44) is not one of
  the three cells; setting `figures.cell-report-card.options.identity` in the
  config would draw it, and a stage that wants that figure in the net may do
  so and refresh the record.
- `second-verse` refuses: no cell passed the primary rhythm test with two
  sufficiently observed cycles in a 24 h recording.
- `contact-ledger` refuses: the three cells never touch, so `contacts.csv`
  has no rows and is not written.
- `neighbour-coordination` refuses: too few sufficiently observed cells.
- `cell-lifecycle-events` fails drawing (`Axis limits cannot be NaN or Inf`):
  every event is a first-frame arrival or a last-frame ending.
- `period-method-audit` fails (`Unknown format code 'g' for object of type
  'str'`) and `audit-focused` fails (`unknown figure 'period-method-audit'`):
  both are Motion's own behaviour on this movie at this commit, recorded as
  found and not fixed here (stage 11 retires Motion; stage 08 ports these
  figures).
- `cluster` refuses: three cells is below its ten-cell training floor.
- `videos` with no `--identity` finds no cell to film: no cell carries a
  lifecycle event other than being present at the first and last frame.
- The `history` module writes nothing: the movie has no tracker decision
  folder, which is the ordinary case for outlines that did not come from the
  tracker.
- `contrasts` on `area_px_median` and `turnover_index_median` across the
  two windows would refuse (every paired difference is zero, because the
  cells are fixed-size squares), so the contrast tests `corrected_mean_median`
  and `integrated_density_median`, which do move.

## Two properties of Motion worth knowing before comparing

- Run twice, the run is byte-identical (checked at freeze time: every CSV
  matched across two runs).
- The standalone `window` and `contrasts` commands read `cell_frame.csv`
  back from disk at nine significant figures, while `run` windows the
  in-memory table at full precision. The windowed medians, IQRs and the
  contrast's confidence bound therefore differ in the ninth digit between
  the two routes (`background_median_iqr` 0.124997437 against 0.124997432).
  The frozen numbers are `run`'s. A port that computes windows from the
  written table will show this in `cell_summary_windowed.csv`,
  `frame_summary_windowed.csv`, `window_change.csv` and `statistics.csv`,
  and nowhere else.
- Pipeline demo tables that carry absolute paths in cells (the linked-index
  and source tables) are frozen by hash and scrubbed sample rows; a port
  that runs them in a new folder will need to scrub before comparing.

## Original source retirement

On 2026-09-22 the original `Motion/analysis` source folder was retired locally
after the complete port verification. Motion now contains three compatibility
files forwarding the legacy commands to PyMicroglia. No retirement commit or
public release has been made. The original 1,285 files are retained in
`development/motion-analysis-backup/analysis-20260922-061053.zip`, relative to
the PyMicroglia root, with every file checked against the SHA-256 inventory in
`development/motion-analysis-backup/manifest.json`. The historical fixture
and its original expected outcomes above remain the reference. The live
replacement is verified by `development/motion-retirement-verification.json`.
