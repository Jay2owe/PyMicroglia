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

## graphify

This project has a graphify knowledge graph at graphify-out/.

Rules:
- Before answering architecture or codebase questions, read graphify-out/GRAPH_REPORT.md for god nodes and community structure
- If graphify-out/wiki/index.md exists, navigate it instead of reading raw files
- Refresh the Graphify graph only when requested or when an explicit release check requires a current graph.
