# Read Doctor Output

`pymicroglia doctor` reports which Python is answering and whether optional
features and storage are ready.

| Field | Meaning | Healthy state |
|---|---|---|
| `ok` | Overall control-layer health | `true` |
| `interpreter` | Exact Python executable running PyMicroglia | The environment you intended |
| `analysis_kit` | Run-recording dependency | Version number when records are required |
| `circadian_workbench` | Rhythm-analysis dependency | Version number when rhythm analysis is required |
| `video_export_available` | Video encoder availability | `true` for video workflows |
| `rhythm_analysis_available` | Circadian analysis availability | `true` for `test_rhythm` and single-cell rhythm tests |
| `imagej.ok` | Fiji command connection | Required only for a manual ROI |
| `actions` | Registered actions | `26` for version 0.1.0 |
| `pending_actions` | Declared actions without a target | `0` |
| `store_root` | Tier B pixel-cache location | The intended project or local cache |
| `store_dehydrated` | Online-only stored arrays | `0` |
| `complaints` | Actionable setup problems | Empty |
| `fix` | Suggested repair | `null` when healthy |

## Install Into the Reported Interpreter

When `pymicroglia` and `pip` refer to different environments, copy the path
from `interpreter` and invoke it directly:

```powershell
& "C:\path\reported\by\doctor\python.exe" -m pip install --upgrade PyMicroglia
```

## Optional Dependencies

```powershell
python -m pip install "PyMicroglia[kit,figure,seg,video,rhythm]"
```

Re-run `doctor` afterwards. Do not install a missing optional feature unless the
workflow actually needs it.

## Fiji

If `imagej.ok` is false, follow the `reason` field. Typically:

1. Start Fiji.
2. Open **Plugins > AI Assistant**.
3. Enable the TCP command server.
4. Re-run `pymicroglia doctor`.

Nothing except the hand-drawn ROI route requires this connection.

## Store Complaints

If `store_dehydrated` is greater than zero, make the configured store available
offline or set `PYMICROGLIA_STORE` to a local folder. The complaint is about
performance and reliable memory mapping; Tier A results remain separate.

## See Also

- [Installation](../getting-started/installation.md)
- [Artefact store](../concepts/artefact-store.md)
