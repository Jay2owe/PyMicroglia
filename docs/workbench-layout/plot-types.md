# PyMicroglia workbench: agreed plot list

Status: the plot list, no-duplication rule and single-page grouping were agreed for a synthetic demo. The working [demo](demo/index.html) is available for feedback before production integration; exact controls and full capability coverage remain reviewable.

## Rule

Each base plot type has one fixed home and one interactive instance in the entire PyMicroglia extras workspace. Its own controls change the metric, layer, comparison or saved analysis shown there. No other section or tab renders that plot type again, even with different data. A selected recording, cell and frame link different, complementary plot types together. Existing saved analyses and figures remain available as sources or exports; this rule concerns how the workbench presents them.

## Plot types

| Plot type | Possible content and controls |
| --- | --- |
| Image or movie viewer | Original frame or cell crop; outline, identity and mask-change layers; shared frame control |
| Time-series plot | Light, area, footprint turnover, speed and other measurements; aligned lanes for metrics with different units; temporal smoothing and cell selection |
| Matrix or barcode | One matrix view with selectable measurement, rows, columns and colour scale; cell-by-time barcode is one mode |
| Rotatable outline history | A cell's saved outlines through time, with the selected frame highlighted |
| Tissue map | Coverage, period, phase, territory and path layers on tissue coordinates |
| Phase clock | Phase and cycle timing from saved rhythm results |
| Distribution and group comparison view | Distributions and declared group contrasts, selected by analysis |
| Event timeline | Cell states, contacts and intervention events on recording time |
| Cell relationship view | Selected cell pairs, contacts and supported networks |

The first concrete example is the barcode: one cell-by-time matrix can switch among total light, mean light, area, footprint turnover and other available measurements. Spatial aggregation (total versus per-pixel mean light) and temporal display smoothing (raw versus three- or seven-frame mean) are different controls. A measurement is offered only when the selected run contains it.

## Existing source inventory

The current source of truth for available methods is [`src/pymicroglia/ui/capability_map.json`](../../src/pymicroglia/ui/capability_map.json). At the time this list was saved it recorded 88 figures, 228 figure views, 138 actions, 24 measurement modules and 11 pipelines. This document is a proposed interface vocabulary, not a claim that every saved figure already has a live interactive equivalent.
