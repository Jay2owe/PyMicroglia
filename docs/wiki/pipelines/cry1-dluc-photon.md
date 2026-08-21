# Cry1-dLuc Photon Pipeline

## Summary

`cry1_dluc_photon` registers a multichannel Cry1-dLuc time-lapse on transmitted
light and creates photon-aware videos and quality-control products.

## Scientific Boundary

The registered TIFF stores raw transformed counts and is the scientific output.
Variance stabilization, display percentiles, legends, timestamps, and movies
are display products and must not be used for quantitative measurement.

## Command

```powershell
pymicroglia run cry1_dluc_photon `
  source="C:\path\to\cry1-dluc.ome.tif" `
  videos=True `
  --claim "register the Cry1-dLuc recording and inspect photon dynamics"
```

The input must be a time/channel/y/x stack with at least three channels. The
pipeline checks the expected channel order before processing.

## Main Parameters

Use `pymicroglia describe cry1_dluc_photon` for the full live list.

| Parameter | Default | Meaning |
|---|---:|---|
| `source` | required | Input multichannel TIFF or OME-TIFF. |
| `fps` | `4.0` | Video frame rate. |
| `crf` | `18` | MP4 quality setting; lower values make larger, higher-quality files. |
| `compression_level` | `4` | TIFF compression level. |
| `anscombe_offset` | `0.375` | Anscombe transform constant; not a cosmetic tuning parameter. |
| `videos` | `true` | Render display products after the registered TIFF. |
| `if_exists` | `"version"` | Existing run-folder policy. |
| `reuse` | `true` | Reuse an exact existing run or upstream result. |

## Outputs

- Registered raw-count TIFF.
- Registration shifts and residual checks.
- Photon-aware display frames and quality-control panels.
- Videos when `videos=True`.
- `manifest.json`, `REPORT.md`, and run provenance.

## Python

```python
from pymicroglia.pipelines.cry1_dluc_photon import run

manifest = run(
    r"C:\path\to\cry1-dluc.ome.tif",
    videos=False,
    claim="produce a registered raw-count stack for quantitative review",
)
```

`videos=False` stops after the scientific registered stack and its review
artefacts.

## See Also

- [Pipeline index](README.md)
- [Analysis flow](../concepts/analysis-flow.md)
- [Action index](../actions/README.md)
