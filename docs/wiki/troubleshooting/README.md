# Troubleshooting

Start with:

```powershell
pymicroglia doctor
pymicroglia discover
```

| Symptom | First check | Route |
|---|---|---|
| Command not found | `python -m pip show PyMicroglia` | Confirm installation and the active environment. |
| Wrong package versions | `pymicroglia doctor` | Use the reported interpreter for installation. |
| Run history unavailable | `analysis_kit` in doctor output | Install the `kit` extra. |
| Rhythm action unavailable | `circadian_workbench` in doctor output | Install the `rhythm` extra. |
| Figure or segmentation import error | Optional feature bundle | Install `figure` or `seg`. |
| Video export unavailable | `video_export_available` | Install the `video` extra and check FFmpeg support. |
| Manual ROI unavailable | `imagej` block in doctor output | Start Fiji and its AI Assistant command server. |
| Action says parameter is unknown | `pymicroglia describe ACTION` | Use the live parameter name and type. |
| Pipeline completed but result is blocked | `REPORT.md` | Resolve the named evidence, question, or remedy. |
| Expected reuse did not occur | Artefact sidecars and `store.explain` | Compare source, parameter, method, and upstream keys. |
| Store access downloads large files | `store_dehydrated` | Make the Tier B store available offline or use a local store path. |

## Safe Diagnostic Order

1. Run `doctor`.
2. Run `discover` when action binding is involved.
3. Run `describe` and `validate` before retrying a long action.
4. Read `REPORT.md` and `manifest.json` for a completed pipeline.
5. Inspect the exact artefact key before disabling reuse.

Avoid recursive searches across synced microscopy folders. PyMicroglia's scan
command deliberately indexes one results folder at a time.

## See Also

- [Read doctor output](doctor.md)
- [Using actions](../actions/using-actions.md)
- [Artefact store](../concepts/artefact-store.md)
