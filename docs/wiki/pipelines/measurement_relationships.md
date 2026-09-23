# Measurement relationships

Declare within-cell, between-cell and lag questions separately. Retain real observation clocks, sample aggregation, unavailable timing and the complete saved lag grid.

The public action is `measurement_relationships`; its request declares `pipeline: "measurement-relationships"`.
Pass a validated request object or JSON path as `pipeline_request`, with the saved
measurement `run` and a human-written `claim`. Multiple recordings require
pooled tables. `only` selects steps with their prerequisites.

Inspect `pymicroglia describe measurement_relationships` for current arguments. Request fields
are validated by the workflow before execution. Scientific and presentation
settings are separate. Reuse requires matching code and unmodified evidence.

Open the generated `index.html` for linked outcomes, figures, and exclusions.
Changing presentation reads the exact saved results; it does not silently rerun
scientific calculations. Keep the shared ledgers with copied evidence.
