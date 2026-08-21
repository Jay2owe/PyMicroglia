# Documentation Standard

Use these shapes when extending the wiki.

## Action Page

````markdown
# action_name

## Summary
What the action does and when to use it.

## Scientific boundary
What may be measured from the output, what is display only, and any refusal.

## Command
```powershell
pymicroglia run action_name source="C:\path\to\recording.ome.tif"
```

## Parameters
| Parameter | Type | Default | Meaning |

## Returns
Python values returned immediately.

## Saved outputs
Files and stored artefacts written by the action.

## Review
What a person must inspect before using the result.

## See also
Related actions, concepts, and workflows.
````

## Pipeline Page

Every pipeline page must state:

- Expected input shape and channel assumptions.
- Stages in execution order.
- Instrumental or negative controls.
- Display-only outputs versus measurement outputs.
- Output folder collision behavior.
- Review blockers and what resolves them.
- A minimal command and a Python example.

## Parameter Wording

Keep the main table focused on what each parameter controls. Put controlled
values in a separate option table and mark the default.

| Option | Behavior |
|---|---|
| `"version"` (default) | Keep the existing run and create a new version. |
| `"overwrite"` | Replace the existing run folder. |
| `"error"` | Refuse if the run folder exists. |
| `"skip"` | Return the existing manifest without recomputing. |

## Output Wording

- **Returns** are Python values received by the caller.
- **Saved outputs** are files written to disk.
- **Tier A artefacts** are permanent derived results.
- **Tier B arrays** are rebuildable cached pixels.
- **Decisions** are human choices reused independently of method versions.

## Example Requirements

- Use public imports and commands.
- Use `C:\path\to\...` placeholders.
- State when a command can read a large image stack or run for hours.
- Give a read-only inspection command before an expensive command.
- Include a claim for actions that draw a scientific conclusion.
