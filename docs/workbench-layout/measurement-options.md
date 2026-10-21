# Recommended cell measurement options

Implemented measurement picker grouping, 2026-10-03. The loaded 95_A3 measurement run enabled
only `intensity`. Its cell-frame table therefore contains reporter measurements
and no area, shape, movement, branching or neighbour values. The current
selectors already discover numeric columns saved in the cell-frame table.

Keep the configurable trace and trace grid; barcodes support requested comparison
copies in the same section. Group the measurement
choices within their selectors; retain per-plot aggregation, temporal smoothing,
overlay/stack and units settings. A missing module is a measurement dependency,
not a reason to invent a trace or rerun tracking.

## First additions: values that change with time

| Group | Recommended choices | Existing native columns |
|---|---|---|
| Size | Area, perimeter, equivalent diameter | `area_px`, calibrated `area_um2`; `perimeter_px`, calibrated `perimeter_um`; `equivalent_diameter_px`, calibrated `equivalent_diameter_um` |
| Shape | Circularity, solidity, elongation, outline complexity | `circularity`, `solidity`, `aspect_ratio`, `ramification_index` |
| Processes | Branch count, total/longest branch length, number of tips | `branch_count`, `total_branch_px`, `longest_branch_px`, `skeleton_endpoints`; supplementary `skeleton_px` / calibrated `skeleton_length_um` |
| Footprint change | Turnover, gained/lost area, net area change, extension bias | `turnover_index`, `gained_px`, `lost_px`, `area_change_px`, `extension_bias` |
| Movement | Speed, consecutive-frame centroid/body steps, displacement from start | `speed`, `step_px_gapless`, `soma_step_px_gapless`, `displacement_from_first_px` |
| Exploration | Territory reached so far, newly explored area, revisited fraction | `cumulative_unique_px`, `new_px`, `revisit_fraction` |
| Neighbours | Nearest outline/centroid distance, cells within the recorded radius, local density | `nearest_edge_px`, `nearest_neighbour_px`, `neighbours_within_radius`, `local_density` |

First priority: area, footprint turnover, speed, branch count, total branch
length and nearest outline distance. These complement the existing reporter
traces and frame-linked outline views.

Useful immediate comparisons in the existing trace: total light + area;
mean light + area; turnover + body movement; branch length + gained/lost area;
signal + nearest-neighbour distance. Different units use separate lanes or an
explicit normalised overlay. Gapless steps preserve the native missing-frame
semantics.

## Optional specialised measurements

| Group | Choices | Existing source / integration requirement |
|---|---|---|
| Finer branching | Median branch length, free-tip branches, branch order, branch points | Morphology cell-frame summaries; prefer `branch_count` (junction clusters contracted) over legacy `skeleton_branches` |
| Radial structure | Maximum ring crossings, radius of maximum crossings, radial reach | Sholl cell-frame summaries; preserve the choice of cell-scaled versus fixed-width rings |
| Direction | Turning angle, step direction | `walk` cell-frame table |
| Territory use | Share of eventual territory covered, cell area relative to territory | `territory_shape_frame` |
| Additional reporters | Mean/total/corrected signal, punctate fraction, saturation for each named channel | `cell_channels`; choose channel and aggregation within the existing plots; requires a table adapter |
| Contacts | Shared boundary or contact status with a chosen partner | `contacts_frame`; retain the recorded dilation and partner identity; requires a pair-to-selected-cell adapter |
| Tissue objects | Distance to an object outline, overlap fraction, touching status | `cell_objects`; choose object set; requires that input and a table adapter |

Keep flags available beside cells and in filtering: touching the field edge,
disconnected pieces, fragments, usable shape ratios and finite measurement
coverage.

## Whole-recording cell summaries

Total path length, net displacement, path straightness, directional persistence,
eventual territory area, core territory fraction, shared-territory fraction,
time in contact and observation coverage belong in cell summaries and cohort
comparisons. These have one value per cell or cell pair, rather than an observed
value at each time point.

The measurement registry also declares lag profiles, recurrence curves, branch
rows and spatial maps. Those retain their existing specialised visual homes.

## Accepted interface

The measurement browsers share six tabs: Signal, Size & shape, Branches,
Movement & change, Territory, and Neighbours & contacts. Switching tabs preserves
selection. A selected-measurements strip spans all groups; unmeasured choices stay
visible, disabled and labelled "Not measured yet". Traces, barcodes, comparisons
and rhythm settings use the same measurement catalogue. Each plot's metric stays
local. Browsing choices starts no analysis.

The interface reads existing saved values. Existing saved masks and tracking
identities can supply missing modules through an explicit measurement job;
adding a picker option does not calculate it or rerun tracking.
