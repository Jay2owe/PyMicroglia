"""Compatibility names for statistics owned by Circadian Workbench."""
from pymicroglia import workbench
REFERENCES = workbench.sample_contrasts.REFERENCES
UnsupportedContrast = workbench.cw.WorkbenchUnavailableError
policy = workbench.sample_contrasts.independent_sample_policy
mean_difference = workbench.sample_contrasts.mean_difference
compare = workbench.sample_contrasts.independent_sample_contrast
