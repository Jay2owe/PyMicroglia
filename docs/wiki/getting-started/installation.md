# Installation

## Goal

Install PyMicroglia and only the optional features required by your workflow.

## Core Package

```powershell
python -m pip install PyMicroglia
```

The core install provides lazy TIFF access, registration, filtering, the action
registry, the artefact store, and result objects.

## Optional Feature Bundles

The names inside square brackets are optional feature bundles:

| Extra | Adds | Use it for |
|---|---|---|
| `kit` | `analysis-kit` | Run records, searchable history, notebooks, and shared style metadata. |
| `figure` | Matplotlib and `analysis-kit` | Quality-control and publication figures. |
| `seg` | scikit-image | Cell segmentation. |
| `video` | FFmpeg support through imageio | MP4 exports. |
| `rhythm` | Circadian Workbench | Periodograms, cosinor fits, and rhythm significance. |
| `test` | pytest | Local development and test runs. |

For the complete single-cell dLuc pipeline:

```powershell
python -m pip install "PyMicroglia[kit,figure,seg,video,rhythm]"
```

For local development from a clone:

```powershell
git clone https://github.com/Jay2owe/PyMicroglia.git
cd PyMicroglia
python -m pip install -e ".[kit,figure,seg,video,rhythm,test]"
```

## Check It Worked

```powershell
pymicroglia doctor
```

Read these fields in the JSON response:

- `interpreter`: the Python installation answering the command.
- `analysis_kit`: present when run recording is available.
- `circadian_workbench`: present when rhythm analysis is available.
- `video_export_available`: whether video encoding can run.
- `pending_actions`: should be `0` in a complete installation.
- `complaints`: actionable setup problems.

Fiji is optional. It is needed only for drawing a manual outline; the health
report may show `hand_roi_available: false` while the rest of the package is
healthy.

## Next

- [First analysis](first-analysis.md)
- [Doctor troubleshooting](../troubleshooting/doctor.md)
