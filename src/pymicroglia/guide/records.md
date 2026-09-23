# Files, provenance and replay

Visible folders contain tables, figures and reports. Machine documents live in
the shared Auto-Organotypic workings folder, currently `.auto-organotypic`.
One shared `artefacts.json` ledger describes each output folder.
Keep ledgers with their files when copying or archiving results.

Scientific workflow tables use CSV with JSON-encoded cells, preserving nested
native diagnostics, exact numbers, empty text and missing values. Column types
live in the ledger. Read them with
`pymicroglia.pipelines._screening.read_table(path)`.
Verified fingerprints cover bytes and type contracts.

Completion documents hold each outcome once. Execution records reference those
outcomes and record each invocation. Legacy JSON tables and inline execution
records remain readable without rewriting. Figures bind exact completions and
reject altered evidence or selections. Report copies retain table contracts
and relative navigation.
