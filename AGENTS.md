## Output ownership and placement

Treat the current checkout as your desk, not the filing cabinet. A requested
output does not automatically belong in the repository you are currently
working in. Before creating a new code file, document, figure, test, report or
generated artefact, read the sibling consolidation map at
`../../Auto-Organotypic/docs/consolidation/00_overview.md` when present, then
write the output to the package that owns it.

- **Auto-Organotypic:** automation, registration, outlining, batch running and
  SCN-specific image work; cross-package consolidation plans go in
  `docs/consolidation/`.
- **Circadian Workbench:** circadian statistics and circadian figures.
- **ClockCytePy:** ClockCytePy spatial analysis and spatial maps.
- **PyMicroglia:** microglia-specific segmentation, decoys and single-cell
  traces.

Keep integration plans in Auto-Organotypic and package-specific code, tests and
outputs in their owning package. Do not leave an output in the current
repository merely because it is the working directory.

When building condition-assignment controls, keep batch cell IDs, recording checkboxes and previewed recording-name regular expressions behind one Set condition action.

When preparing a comparison for review, load every requested recording into one persisted comparison workspace and verify combined plot counts in that window.

When recordings are loaded together, expose every source in the top Recording dropdown and preserve the combined workspace when switching.

When explaining a running analysis, verify its engine against the package's documented default, identify any custom method explicitly, answer only the operation asked about, and keep the job running unless Jamie asks to stop.

## graphify

For exploratory single-cell rhythm analyses, use unadjusted per-trace p-values and do not apply multiple-comparison correction unless Jamie explicitly requests it.

When testing rhythm recovery from simulated images, give the fitter only outlines available after the tested detection procedure; explicitly label any fit using an earlier source outline as a control with additional shape information and separate it from the practical comparison.

For bioluminescence outlining simulations, use the user's selected PyMicroglia chain of U-Net, Motion tracking and single-frame U-Net refinement, with its recorded model and live package defaults unless the user selects another recipe.

When changing the simulation backend or runtime, preserve the existing comparison page layout and section order unless Jamie explicitly requests a redesign.

When comparing rhythm recovery from simulated cell images, retain all requested cells and show tracking quality per cell; separate results for a tracking group defined from nonrhythmic calibration images, never from which period method succeeds.

Before rendering or tracking a continuous-emission simulation, reject missing source shapes; missing recorded tracking outlines must never silently switch off simulated cells, and reconstructed or generated shapes must be declared simulation assumptions.

This project has a graphify knowledge graph at graphify-out/.

Rules:
- Before answering architecture or codebase questions, read graphify-out/GRAPH_REPORT.md for god nodes and community structure
- If graphify-out/wiki/index.md exists, navigate it instead of reading raw files
- Refresh the Graphify graph only when requested or when an explicit release check requires a current graph.
