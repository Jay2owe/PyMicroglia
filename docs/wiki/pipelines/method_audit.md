# Method audit

Declare benchmark designs, candidate methods and confirmation policy. Development, selection and independent confirmation remain separate; export records the selected saved decisions.

The public action is `method_audit`; its request declares `pipeline: "method-selection-audit"`.
Pass a validated request object or JSON path as `pipeline_request`, with the saved
measurement `run` and a human-written `claim`. Multiple recordings require
pooled tables. `only` selects steps with their prerequisites.

Inspect `pymicroglia describe method_audit` for current arguments. Request fields
are validated by the workflow before execution. Scientific and presentation
settings are separate. Reuse requires matching code and unmodified evidence.

Open the generated `index.html` for linked outcomes, figures, and exclusions.
Changing presentation reads the exact saved results; it does not silently rerun
scientific calculations. Keep the shared ledgers with copied evidence.
