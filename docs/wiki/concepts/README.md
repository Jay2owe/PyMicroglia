# Concepts

These pages explain the rules shared by every PyMicroglia workflow.

| Page | Question it answers |
|---|---|
| [Analysis flow](analysis-flow.md) | In what order are pixels transformed, measured, and displayed? |
| [Artefact store](artefact-store.md) | What is permanent, what is cached, and when is work reused? |

## Core Rules

- Register before cosmic-ray removal.
- Measure unsmoothed data; display smoothing is a separate branch.
- Store derived evidence and parameters, not unnecessary copies of raw pixels.
- Reuse only exact matches for source, parameters, method version, and upstream
  artefacts.
- Refuse ambiguous upstream results instead of guessing.
- Run instrumental controls before accepting a circadian claim.
