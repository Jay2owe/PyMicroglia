# Stale video parity imports
**Date**: 2026-08-26
**Files changed**: `tests/test_video_parity.py`
**Guard**: `test_video_adapter_does_not_import_removed_private_modules`, `test_red_only_translates_to_the_consolidated_renderer`

## What went wrong
PyMicroglia's video implementation moved to Auto-Organotypic and four legacy export
functions were consolidated into `stack_to_video`. The old parity test still
imported deleted private modules through `pymicroglia.video`, so the entire test
suite failed during collection before any tests could run.

## The broken pattern
```python
from pymicroglia.video import annotate, encode, exports, luts, render
# annotate, exports, luts and render no longer belong to the video package.
```

## The fix
The regression tests now check the supported boundary directly: PyMicroglia's
public generic renderer is the installed Auto-Organotypic function, and established
action names translate their arguments into that renderer without importing
the deleted private modules.

## Why it matters
A stale test must not make PyMicroglia untestable after an implementation moves
to its owning package. Deep renderer parity belongs to the Auto-Organotypic suite;
PyMicroglia guards only its public delegation address.
