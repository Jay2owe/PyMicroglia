# Microglial data-only figure renderers

These panels render tables prepared by PyMicroglia; scientific measurement and
rhythm estimation belong upstream. Use the package adapters to prepare inputs.

`panels/all_cell_traces.py` preserves original acquisition hours, missing values
and saved evidence. Its optional `grid_layout_inches` setting accepts positive
`width`, `row_height`, `top` and `bottom` dimensions, keeping long-grid margins
independent of the number of rows. Omitting it retains the established layout.
No layout option changes measurement values or fit evidence.

Focused validation: `python -m pytest tests/test_trace_grid_layout.py -q` from
the package root. Default geometry, every plotted sample and invalid dimensions
are checked. The corrected-CD68 source-bound figure choices live separately in
Jamie/Admin/LSR/Poster plot choices/Corrected CD68 traces; they remain draft.

<!-- agent-work-trail:begin -->
## Current work
Status: draft (does not imply scientific approval).
- deliverable: [README.md](README.md)
- support: [all_cell_traces.py](panels/all_cell_traces.py)

## Recent changes
- 2026-10-24: Optional physical margins compact long trace sheets; default renderer geometry and data remain unchanged
<!-- agent-work-trail:end -->
