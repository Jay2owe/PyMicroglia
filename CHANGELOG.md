# Changelog

## 0.3.0 - 2026-09-23

- Add a post-Motion eligibility stage to `auto_microglia`. It defaults to
  excluding identities with internal gaps over four hours or at least 50%
  missing data from analysis only, while independent configurable views let
  users apply the same exclusions to review videos and still images. Motion
  labels and identity numbers remain unchanged.
- Add configurable tracked-cell video and still actions over original photons,
  using the accepted per-identity outside outlines. Width, opacity, colours,
  contrast, display-only smoothing, timestamps and playback are selectable.
- Require Auto-Organotypic 0.7.2 for frame-specific outline rendering.
- Keep the `rhythm` extra installable alongside the required Circadian
  Workbench 0.9 series.
- Package tracked-cell measurements, state analysis, clustering, cell-following
  videos and 84 figure actions inside PyMicroglia. The automated chain runs a
  frozen Motion tracker through its declared target; Motion retains its active
  tracking and review workflow.
- Add six result-dependent workflows with explicit claims, saved replay and
  scientific/display cache separation: rhythms, method audits, measurement
  relationships, accepted states, spatial coordination and interventions.
- Route all shared statistics through Circadian Workbench 0.9; preserve the
  configured estimator, significance test, sufficiency rules and provenance.
  Unknown rhythms have no default daily folding or common-period assumption.
- Give saved figures named views and move their data preparation outside the
  drawing layer. Save tables and figures through one artefact ledger per folder;
  keep metadata documents in the shared workings directory.
- Keep the original recording-manifest byte identity when adding figure plans,
  so saved replay and seeded scientific calculations retain their original inputs.
- Use explicit public parameter names where Motion meanings differed from existing
  actions: `state_method`, `cell_count`, `ring_count` and
  `transition_normalisation`. Historical Motion plot/film arguments are translated
  by the compatibility command; measurement choices and numerical operations remain unchanged.
- Include searchable package guidance, generated action descriptions and context
  files. Add the `states` optional dependencies and preserve legacy table names
  when reading typed saved pipeline results.


- Depend on `Auto-Organotypic>=0.7,<0.8`, and take that release's conventions
  registry as the one place a run's colours live. `auto_microglia` writes
  the chain's resolved `conventions.json` beside its own record and draws its
  cell traces under it; `phase_green_red` declares the green and red its name
  promises to the registry, writes it in the run folder and draws its four
  movies from it; `cry1_dluc_photon` keeps its hand-composed purple and
  records that it is explicit. The trace panel's first colour is the
  registry's inside a run and the house dLuc outside one, as before.
  `trace_tables.detrend` calls `auto_organotypic.baselines` by name and keeps
  only the panel's own shading margin; `normalise` stays, with the reason
  written down. Nothing this package draws changes outside a run of the
  chain, and its frozen panel fixtures hold.

- Put the learned mask **inside** Auto-Organotypic's chain rather than in a loop
  after it. That package now publishes `register_stage`, and
  `auto_microglia.register()` — called on import — adds `cell_masks` after
  `split`, owned by PyMicroglia. It is a stage there in every sense: addressable
  as `stages=("cell_masks",)` and `since="cell_masks"`, dropped by `skip=`,
  configured with `cell_masks_options` checked against `cell_masks.mask_run`'s
  live signature before any stage runs, reported in the chain's own record with
  this package named as its owner, announced through `on_progress`, and given
  its own row in the `what_would_run` staleness grid — the one step in this
  pipeline that costs minutes, so the one worth being able to ask about before
  paying for it. One run, one record: the run used to be split across the
  chain's record and a manifest written here, which nobody who was not present
  could read as a single run.
- Keep a plain Auto-Organotypic run unchanged on a machine with PyMicroglia
  installed: the stage registers as `opt_in=True`, and naming its options is how
  it is asked for — that package's own rule for its exports and its review. So
  `cell_masks=False` now adds *nothing* to the call rather than adding a skip,
  and what reaches the chain is its own default run.
- Read the still mask's labels back from the file the record names instead of
  passing an array between two functions in one process. A run record names a
  file and never contains one, and a label image per recording inside a JSON is
  how a run record becomes the largest thing in the output tree.

- Add the `auto_microglia` pipeline: Auto-Organotypic's whole chain — instrument
  pull, broad crop, RIPR registration, trim, split — with microglia defaults, and
  then the learned cell mask, cell traces and the handoff to Motion. It calls
  `auto_organotypic.pipeline.run_pipeline` and passes every keyword through
  rather than restating its parameter block, so a stage, parameter or default
  that package gains arrives here with no edit. `auto_microglia.differences()` is
  the complete list of what it decides differently and the tests assert it is
  complete.
- Default `outline`, `trace` and `spatial` off in that pipeline: these recordings
  often contain no SCN, and an outline of something that is not there would
  orient and crop every stage below it around a mistake. Off is a default, not a
  removal — `outline=True` restores the SCN half whole at Auto-Organotypic's own
  settings, `region_traces` and `spatial` follow it unless given, and asking for
  either without it is refused at the keyword rather than several stages later on
  a missing file. Skipping it loses no trace — `region_trace.run` takes
  `labels=`, so the maintained trace, control and rhythm engine runs on cell
  labels instead of outline lobes.
- Enforce the inverse stage order there: in `auto_microglia` the learned mask is
  what finds the cells, so a measurement logged before it is refused, where in
  `dluc_single_cell` the mask is a branch off a finished measurement and the
  opposite is refused. The accepted seedless detector is not used in the new
  pipeline — it needs a 14 hour window and keeps only cells that hold still —
  and is untouched where it can be used.
- Write a recipe beside every learned mask and reuse the mask when it matches,
  so re-running a folder does not pay for the network pass again — minutes per
  recording, against seconds for everything else in the run. Keyed on the
  *pictures* rather than on the recording: a stack prepared with a different
  window, or without the cosmic-ray step, is a different set of pictures from the
  same file, and a key naming only the file would hand back a mask of something
  else. `force=True` (the chain's own word) or `reuse=False` masks again.
- Report progress from the mask step in the shape Auto-Organotypic's own stages
  use, so a watcher told about every other stage is told about the slowest one
  too instead of seeing a run that appears to have stopped.
- Leave `mask_window_h` out rather than forward it as `None`: the window's
  default is the duration the weights were trained on and lives next to them, so
  passing `None` asked that function to treat "not specified" as a number.
- Refuse an `auto_microglia` run whose `NOT_OURS` names a stage the installed
  Auto-Organotypic no longer has. `skip` is filtered against the stage list and
  an unknown name in it is ignored, so a rename upstream would have left every
  recording outlined, oriented and cropped around a two-lobe structure these
  recordings do not contain — a full run of confident, wrong results, with no
  error. `AO_STAGES` and `CLAIMED` are the other two copied lists and each has a
  test comparing it against the live package; nothing else in the module names
  anything of that package's, so everything else follows it with no edit.
- Rename the chain's `review` stage to `chain_review` inside `auto_microglia`,
  the one keyword of Auto-Organotypic's that cannot pass through under its own
  name: `review` is a `pymicroglia.review.Review` in every pipeline of this
  package, so `review=True` bound to that and raised `'bool' object has no
  attribute 'as_records'` instead of asking for the scorecard.
  `auto_microglia.CLAIMED` is the whole translation layer and a test asserts it
  stays whole, so the next colliding name fails a test rather than a run.
- Add `cell_masks.mask_still`: one label image from the mean of every frame, so a
  cell is one region for the whole recording, which is what a trace needs and
  what a per-frame mask cannot give.
- Add `pipelines.motion_handoff`: writes `motion/motion_inputs.json` in the
  Motion project's own `config.json` vocabulary, pinning each registered stack
  where it already is, plus its masks, by path and SHA-256. The stacks Motion
  computes itself are listed under `still_missing` rather than written empty.
  Motion is not installed, so the stage reports `pending` with the sentence that
  would make it ready instead of raising mid-run.
- Add `pymicroglia.learned_mask`: a mask for a single exposure, for the
  recordings the accepted seedless detector cannot serve because it needs a
  14 hour window. Normalise, a small U-net, then grow each seed out into its
  processes. Weights are not shipped; `PYMICROGLIA_MASK_WEIGHTS` names the run,
  so a mask always traces back to the round that produced it.
- Warn rather than refuse when a recording's pixel size is not one the weights
  were trained on. The failure is otherwise silent: a 2x mismatch costs a third
  to a half of the cells and the mask still looks reasonable.
- Add the `mask` extra (torch, scikit-image), imported inside the functions that
  need it so a machine without them still imports `pymicroglia`.
- Add `learned_mask=True` to the `dluc_single_cell` pipeline: the same mask as
  an opt-in branch off a finished run, written as probability, labels and mask,
  one stack each. It runs after the measurement and changes no number the run
  reports; `check_stage_order` refuses a run shaped the other way round. Two
  runs that differ only in whether they asked for it share a run folder, since
  they are the same analysis.
- Add `controls.read_findings`, and make the instrumental control work again.
  Auto-Organotypic stopped returning `instrumental_rhythm_detected`, `passes` and
  `dluc_clean` on 2026-09-14 because a per-recording pass mark cannot tell a
  filled well's own glow from the incubator. `run_controls` was still
  subscripting two of those keys and raised `KeyError` on every real call, and
  `dluc_single_cell` read the other two through `.get` with defaults, so every
  run reported no instrumental cycle and a clean bioluminescence channel
  whatever the control had seen. The judgement is now made once, from the
  readings, at the threshold the control was run at.
- Add `development/`, where an analysis is proved before any of it becomes a
  module, with the single-frame mask rounds as its first occupant.

## 0.2.0 - 2026-08-26

- Move longitudinal image reading, registration, cosmic-ray correction,
  filtering, rhythm analysis and the keyed artefact store to Auto-Organotypic while
  retaining PyMicroglia's public adapters.
- Save SVG, PDF, PNG, JPEG, TIFF, WebP, AVIF and HEIF figure variants through
  ReproFig with exact plotted tables, source provenance and configurable
  resolution.
- Add opt-in proof policies, controlled promotion and canonical publication
  workbooks while preserving the ordinary one-call figure workflow.
