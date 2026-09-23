# Cell behaviour states

Validate shared state structure before describing assignments, profiles, timelines, occupancy, transitions, bouts and independent sample effects. No supported states is an explicit outcome.

The public action is `behaviour_states`; its request declares `pipeline: "cell-behaviour-states"`.
Pass a validated request object or JSON path as `pipeline_request`, with the saved
measurement `run` and a human-written `claim`. Multiple recordings require
pooled tables. `only` selects steps with their prerequisites.

Inspect `pymicroglia describe behaviour_states` for current arguments. Request fields
are validated by the workflow before execution. Scientific and presentation
settings are separate. Reuse requires matching code and unmodified evidence.

Open the generated `index.html` for linked outcomes, figures, and exclusions.
Changing presentation reads the exact saved results; it does not silently rerun
scientific calculations. Keep the shared ledgers with copied evidence.
