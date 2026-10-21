# Research: where the PyMicroglia workbench tabs should divide

Status: research and the agreed synthetic-demo layout, 29 September 2026. The user approved generating the single-page layout described below. The [interactive demo](demo/index.html) is a separate prototype, not a change to the production UI. The agreed nine plot types are in [plot-types.md](plot-types.md).

## What exists now

- The top row has three full-window workspaces: Auto-Organotypic, Motion tracks, and PyMicroglia extras. Each has different operations and its own sidebar. This matches the user's earlier decision to place Motion tracks second. See [index.html](../../src/pymicroglia/ui/static/index.html).
- PyMicroglia extras currently has nine scientific pages. The largest are Signal and rhythms (19 figure entries, 51 views), States and traits (14 entries, 40 views), and Neighbours and coordination (12 entries, 35 views). The source inventory has 88 figure entries and 228 named views in total. See [capability_map.json](../../src/pymicroglia/ui/capability_map.json).
- The automatic charts in [quicklook.js](../../src/pymicroglia/ui/static/quicklook.js) repeat the same time-series and distribution forms on multiple scientific pages. This is the precise duplication the proposed one-instance-per-plot-type rule would remove.
- A separate saved [cell review](../../src/pymicroglia/visualisation/review_assets/review.html) already uses one cell and frame selection to coordinate original image, mask changes, rotatable outline history, signal and area traces, and a cell-by-time barcode. Its [controller](../../src/pymicroglia/visualisation/review_assets/review.js) is the strongest local proof of the linked interaction Jamie wants.
- The current automatic preview API reads a limited selection of saved measurement rows: up to 32 cells, 450 displayed points per trace and 256 MiB of source tables. It is appropriate for previews, but a full recording-wide interactive barcode or high-resolution scrubbing would need a deliberate data-loading design. See [quicklook.py](../../src/pymicroglia/ui/quicklook.py).

## Evidence that affects the split

| Source | Finding | Implication for PyMicroglia |
| --- | --- | --- |
| [GOV.UK Design System: Tabs](https://design-system.service.gov.uk/components/tabs/) | Tabs suit clearly related, mutually exclusive content when one section is more useful first. They hinder comparisons that require switching back and forth, and too many tabs weaken labels. | Keep the broad product workspaces in the top row. Keep complementary PyMicroglia plots together on one page; put the most useful group first. |
| [W3C: Tabs pattern](https://www.w3.org/WAI/ARIA/apg/patterns/tabs/) | A tab set displays one panel at a time and has a defined keyboard and focus model. | Use real tabs only for mutually exclusive workspaces. Use controls on the plot itself for its metrics and layers. If the top row is implemented as tabs, provide the tab roles, selection state and arrow-key behaviour. |
| [Baldonado, Woodruff and Kuchinsky, 2000](https://www.cs.loyola.edu/~lawrie/CS774/S06/research/p110-wang_baldonado.pdf) | Coordinated views are justified when they reveal complementary information; similar views should be merged, and their coordination should be visible and consistent. | One barcode can switch measurement. The original cell image and morphology history deserve separate adjacent views because they answer different questions. A shared cell/frame highlight should make their link obvious. |
| [Shneiderman, 1996](https://hci.stanford.edu/courses/cs448b/papers/shneiderman96eyes.pdf) | Visual exploration commonly starts with an overview, then selection and filtering, then detail. | Open on recording context and let a cell/barcode selection lead directly to its image and traces. Keep detailed fit and method evidence available from that selection. |
| [napari viewer documentation](https://napari.org/stable/getting_started/viewer.html) | Scientific image exploration uses one persistent canvas, selectable layers and dimension sliders. | Treat outlines, mask-change colours, paths and reporters as layers of a single image viewer; use one frame control for the whole PyMicroglia workspace. |

The implications in the right column are design inferences from those sources and the local application. They are not a claim that a published guideline prescribes PyMicroglia's exact tab names.

## Corrected layout proposal: one home per plot type

The saved `review.html` is the model: one page contains distinct, complementary views, each with its own controls and one shared cell/time selection.

Keep the three already accepted top-level workspaces: Auto-Organotypic, Motion tracks, PyMicroglia extras. The recommended PyMicroglia layout uses headings and optional sidebar jump links down a single visual page. A jump link scrolls to the one existing plot; it does not create another plot or switch to a duplicate panel.

| Order on the PyMicroglia page | Complementary plots placed together | Unique plot types used here |
| --- | --- | --- |
| 1. Follow a cell through time | Original image and outline beside rotatable outline history; time-series, barcode and event marks aligned by frame; phase clock near the rhythm trace | Image/movie viewer; rotatable outline history; time-series plot; matrix/barcode; event timeline; phase clock |
| 2. See the tissue and neighbours | Spatial coverage or rhythm layer beside supported cell-pair or network view | Tissue map; cell relationship view |
| 3. Compare cells and groups | One configurable summary for distributions and declared contrasts | Distribution and group comparison view |

Each of the nine agreed plot types appears in **exactly one row**. There is one barcode on the page. Changing its selector from light to area or turnover changes that barcode in place. It never appears a second time under Tissue, Signal, or Cell shape. Similarly, the one trace plot may change metrics or show aligned lanes; there is no second trace plot in an Interventions section.

The first row preserves the working pattern in `review.html` rather than disassembling it. The additions are distinct plot types, such as an event timeline and phase clock, only if they add information that the existing views cannot show clearly. The large number of saved figure functions are configurations, sources or exports for these canonical views; the UI does not need a gallery of repeated chart shapes.

## Linked interaction and metric controls

1. A recording, selected cell or cell pair, and frame/time form the shared selection. Clicks on the image, barcode or trace move that selection. Each plot visibly marks the shared selection and reports when that cell has no observation at that frame.
2. Each plot owns its metric controls. The trace may show total light while the barcode shows footprint turnover. Selecting a scientific measurement changes only the plot whose control was used.
3. Spatial aggregation and temporal smoothing have separate labels. “Total light” sums pixels inside the saved mask; “mean light” is per pixel. “Three-frame mean” smooths the chosen metric over time. Options appear only where saved data supports them.
4. Different units in the one time-series plot use aligned lanes or explicit normalisation, with units shown. Multiple cells may use that same canvas with colour and selection.
5. Pipelines remain deliberate on-demand actions in the sidebar. A plot selector or time slider does not start a long analysis.

## Review before production integration

1. Inspect the [single-page demo](demo/index.html) for spacing, control placement and linked behavior. The three product workspaces have separate sidebars; the PyMicroglia plots each appear once.
2. Do “matrix or barcode” and “distribution and group comparison” each cover truly one visual type? Before implementing, assign all 228 saved figure views to the nine types and mark genuine exceptions rather than forcing dissimilar figures into one control.

Keep this prototype separate from the production interface until feedback on the demonstrated layout and scope is incorporated. The reusable design guidance has also been installed as the application-neutral `app-development` skill.
