# Rhythm discovery

Screen requested cell measurements with separately selected period estimation and significance. Reports retain untestable cells, unsupported periods, groups and compatible timing.

The public action is `rhythm_discovery`; its request declares `pipeline: "rhythm-discovery"`.
Pass a validated request object or JSON path as `pipeline_request`, with the saved
measurement `run` and a human-written `claim`. Multiple recordings require
pooled tables. `only` selects steps with their prerequisites.

Inspect `pymicroglia describe rhythm_discovery` for current arguments. Request fields
are validated by the workflow before execution. Scientific and presentation
settings are separate. Reuse requires matching code and unmodified evidence.

Open the generated `index.html` for linked outcomes, figures, and exclusions.
Changing presentation reads the exact saved results; it does not silently rerun
scientific calculations. Keep the shared ledgers with copied evidence.
