# FFT component test module import
**Date**: 2026-09-23
**Files changed**: `src/pymicroglia/workbench.py`, `tests/test_workbench_seam.py`
**Guard**: `test_fft_component_gateway_does_not_require_package_attribute` in `tests/test_workbench_seam.py`

## What went wrong
The all-cell trace grid reached its fitted-component significance test, then raised `AttributeError` before it could report a result. The Circadian Workbench component-test module was installed but was not exposed as an attribute on the package object. PyMicroglia looked up that absent attribute instead of importing the module directly.

## The broken pattern
```python
return cw.component_significance.test_components(...)  # package attribute may not exist
```

## The fix
Import the component-test function from its module at the PyMicroglia Workbench gateway, then call that imported function.
```python
from circadian_workbench.component_significance import test_components as _test_components
return _test_components(...)
```

## Why it matters
Without the direct import, a valid FFT-NLLS fit can fail at the final significance step, preventing every selected cell grid from being regenerated.
