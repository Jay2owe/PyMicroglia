# Cosmic-ray removal: the engine's own output, kept as the reference

Five files, copied byte for byte from a run of
`Protocols/Analysis/microglia_cosmic_ray_removal.py` on 2026-08-16. Nothing
here was produced by PyMicroglia, and **no test runs an engine** — the
comparison is against output the engine wrote earlier, which is the only kind
of parity claim that survives the engine being absent.

| File | What it is |
| --- | --- |
| `synthetic_registered.ome.tif` | The input. 9 frames, 2 channels, 32x32, one 6000-count spike at frame 5, y 15, x 18. |
| `engine_cleaned.tif` | What the engine wrote. SHA-256 `dea1fcc6…`, recorded in `engine_run.json` as `output_sha256`. |
| `engine_replacement_mask.tif` | Which pixels it replaced, 0 or 255. |
| `engine_events.csv` | The one connected event, with its peak and its replacement value. |
| `engine_run.json` | The settings, and the noise the engine measured: centre −2.0, sigma 5.9304. |

The originals sit in
`Experiments/Tmem119-CreERT2/Tests/Cry1-DIO-dLuc/AI_Exports/_cosmic_standalone_smoke/`
and are not touched. They are 77 KB in total here, which is why this gate does
not need a real acquisition: it is a small file with a known right answer,
rather than a large one with an approximate one.

## Why the METHOD_VERSION differs

`engine_run.json` says `2026-08-16-temporal-neighbour-k12-g2`; the package says
`2026-08-16-matched-line-neighbour-blend`. Matched-line repair was added to the
engine after this run. It changes nothing here — the spike is a compact point,
no trail qualifies, and the port reproduces the older run exactly — which is
the useful property: the line repair only ever *adds* to what the point
detector found.

## Regenerating

Only if the engine's method changes and this reference should follow it. Run
the engine yourself, by hand, and copy the five files across:

```powershell
python "Protocols\Analysis\microglia_cosmic_ray_removal.py" `
    tests\fixtures\cosmic_ray_reference\synthetic_registered.ome.tif `
    --output-dir <somewhere> --signal-channel 2 --mask-growth-px 2 `
    --sample-frames 9 --no-preview
```

Do not regenerate it from PyMicroglia. A reference the package produced is a
test that the package agrees with itself.
