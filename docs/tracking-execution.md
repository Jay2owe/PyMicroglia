# Tracking processor controls

Pass execution resources separately from scientific tracking settings:

```python
from pymicroglia.tracking import run

result = run("motion_inputs.json", "tracking", tracker_options={
    "bounded_memory": True,
    "resume": True,
    "execution": {
        "workers": 4,
        "cpu_budget": 4,
        "memory_gb": 8,
        "stage_workers": {"detection": 2, "identity_handoff": 4},
    },
})
```

`detection` distributes independent frames over threads. `identity_handoff`
distributes read-only candidate discovery over processes, sharing the label movie
through a read-only mapped file. Results are joined in their original order and
receive the same proposal identifiers. Applying corrections remains sequential
because later corrections depend on earlier label edits.

Counts are capped by the run's processor budget, available processors and useful
tasks. The handoff stage also limits workers using a conservative table-memory
estimate. `memory_gb` is a planning budget, not an operating-system memory limit.
Omitting `execution` preserves the existing tracker defaults.

These are the two supported stage overrides; other stages keep their existing
execution. Unknown stage names and invalid counts fail validation. The run's
`engine/execution.json` records settings and implementation hashes.

Run independent recordings in separate processes and separate tracking folders.
The engine's import lock and per-folder ownership lock remain in place. Across
recordings, budget the sum of simultaneously active processors and peak memory;
also limit numerical-library threads inside each worker to avoid nested pools.

On Windows, launch scripts under an `if __name__ == "__main__":` guard. Keep the
package source importable in spawned processes. A running serial launcher must
exit before its positions are handed to a concurrent coordinator.

The focused checks are `tests/test_tracking_execution.py`; they compare proposal
tables exactly against the frozen serial producer, including eligible and tied
cases, shuffled rows, empty inputs and memory fallback. Processor counts are a
capability, not a promised linear speedup; compare complete stage calls,
including process startup and result merging, on the intended recording.
