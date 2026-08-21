# Trace panel reference

What `trace_panel_figure.py` produced, kept so the port can be checked against
it without ever running it. No test in this package executes an engine; every
parity test compares against output an engine produced earlier, and this folder
is that output for stage 09.

## Provenance

Produced on 2026-08-19 by
`Protocols/Analysis/trace_panel_figure/trace_panel_figure.py`, unmodified, from

    Experiments/Tmem119-CreERT2/Tests/Cry1-DIO-dLuc/AI_Exports/
      MCG_04_595_pipeline_improved_2026-08-12/traces_24h.csv

which is that pipeline run's own trace output for `MCG_04 - 1 - 595`
(Cry1-DIO-dLuc, Tmem119-CreERT2), written 2026-08-12.

| File | What it is |
| --- | --- |
| `traces_24h.csv` | The input. SHA256 `e6adbd94…0057296a`, 250 rows, 49 columns. |
| `engine_full_plotted.csv` | Every value the engine drew with default settings — twelve panels, one per cell. |
| `engine_full_provenance.json` | The engine's own provenance record for that figure: per trace the standard deviation, the normalisation denominator and the shaded edge width. |
| `engine_small.png` | A two-panel figure at explicit settings, kept for a pixel comparison. 828 × 299. |
| `engine_small_plotted.csv` | Every value drawn in that figure. |
| `engine_small_provenance.json` | Its provenance record, including the settings it was drawn under. |

The command behind the small figure, which is also the one the test reproduces:

```
trace_panel_figure.py --csv traces_24h.csv \
  --panel "cell_1_processed" \
  --panel "cells 1 and 4 = cell_1_processed[cell 1] + cell_4_processed[cell 4]" \
  --time-start 96 --time-end 168 --xtick-interval 12 --vline-interval 24 \
  --width 6.0 --panel-height 1.4 --dpi 80
```

## Two things to know when reading these

**The paths inside the JSON records are the engine's, verbatim.** They point at
the temporary folder the reference was generated in, which no longer exists.
That is what provenance looks like when it is kept rather than tidied; the tests
read the numbers, not the paths.

**The full figure was cut down to a small one for the pixel test on purpose.**
The default twelve-panel PNG is 1.0 MB, which is not a size to keep in a test
fixture. The numerical comparison is done on the full twelve panels, where it is
strongest, and the pixel comparison on the small one, where it is cheap. Both
came out exact: every plotted value bit-identical, and 0 of 247,572 pixels
different.

## Why the standard deviations are worth checking on their own

The twelve annotations the full figure prints — 6.7, 8.8, 8.4, 15.6, 19.6, 20.9,
52.3, 40.0, 33.1, 43.7, 49.2, 4.7 % — are each computed over the *trustworthy
interior* of a trace, excluding the shaded ends and ignoring the display
smoothing. Getting the shaded width wrong, or smoothing before measuring, moves
them without moving the lines visibly. They are the cheapest signal that the
port kept the whole chain and not just the drawing.
