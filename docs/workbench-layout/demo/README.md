# Coordinated workbench demo

Open [index.html](index.html) directly in a browser. It uses only the adjacent local CSS and JavaScript, with no server, network assets or analysis backend.

This prototype implements the agreed arrangement: a single scrollable PyMicroglia extras page with nine unique plot families, complementary groups, local metric controls, one shared cell/time selection and three full-window product workspaces. Auto-Organotypic and Motion tracks illustrate their own workspace shells; they are not replacements for the installed applications.

## Try the interactions

- Drag the shared frame slider or play the recording; the image, white outline-history slice, trace cursor, barcode cursor and context views move together.
- Click a barcode row/time to select both. Click the trace to select time, or a tissue-map point, relationship node, phase marker or distribution point to select a cell.
- Change the barcode metric among light, area, speed and footprint turnover. Its plot remains in one place and the trace keeps its own metric.
- Switch light aggregation between total in the mask and mean per pixel. Temporal smoothing is a separate control: raw, three-frame mean or seven-frame mean.
- Drag the outline history to rotate it, scroll to zoom, or use its arrow keys. Height represents recording time, not tissue depth.
- Open image settings for base/refined outline overlays and an optional fading trail with a configurable length. Trails start off.
- Inspect cell 7 at frames 82–93 (one-based display) or cell 19 at frames 169–174 to see missing observations preserved across views.
- Switch the top product tabs; selection and plot settings persist. Pipelines open a clearly labelled, unconnected action drawer.

## Synthetic data contract

`demo.js` deterministically creates 24 cells with 240 observations at 30-minute intervals, covering five days. It includes two simulated groups, known 24-hour light oscillations, moving polar outlines, three declared observation gaps and a simulated intervention at 60 hours.

Refined outlines come from the deterministic geometry at each frame; base outlines average that geometry across seven neighbouring frames. Both use the same fixed identity. Geometric area is calculated from the polygon; total light is simulated per-pixel mean light multiplied by that area. The displayed photon-like pixels are a deterministic illustration, not the pixel source of those measurements. Footprint turnover is a simulated measurement, not a measured polygon intersection. Phases are known simulation values; no period-fitting algorithm or significance test is run.

Temporal means use only adjacent observed frames within the same continuous observation segment. A missing frame stays missing. Display settings do not mutate the frozen observations.

## Checks

`verify_demo.py` connects to the dedicated browser-use Chrome automation profile at `127.0.0.1:9333`. Open the demo through `browser-use-auto`, then run the script to exercise the linked interactions, metric isolation, gaps, keyboard tabs, playback and responsive layout. It writes [verification.json](verification.json) and two browser screenshots for layout inspection. These screenshots document the UI; they are not scientific figures.

The source and agreed plot inventory remain in [plot-types.md](../plot-types.md). The navigation evidence is in [tab-research.md](../tab-research.md). The reusable, application-neutral workflow is installed separately as the `app-development` skill.
