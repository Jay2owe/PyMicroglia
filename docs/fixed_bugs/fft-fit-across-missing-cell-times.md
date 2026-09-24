# FFT–NLLS fit drawn across missing cell-times

The saved cell-trace grid previously evaluated the actual summed FFT–NLLS fit
at time points where a cell outline could not be matched. In long gaps,
multiple fitted components could interfere to create large, unsupported
dotted peaks. The displayed curve now omits those times while leaving the
fitted model, measured points and component tests unchanged.

`tests/test_all_cell_trace_grid.py::test_fft_nlls_period_style_uses_the_actual_multicomponent_fit`
checks both an initial gap and an internal missing frame, and verifies that
the displayed finite values remain the actual multicomponent fit.
