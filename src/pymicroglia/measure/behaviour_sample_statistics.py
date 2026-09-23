"""Compatibility names for statistics owned by Circadian Workbench."""
from pymicroglia import workbench
METRICS = ('occupancy_observed', 'occupancy_assigned', 'unknown_fraction', 'switch_rate', 'transition_probability')
REFERENCES = workbench.sample_contrasts.REFERENCES
UnsupportedComparison = workbench.cw.WorkbenchUnavailableError
def policy(settings):
    return workbench.sample_contrasts.state_comparison_policy(settings,metrics=METRICS)
difference = workbench.sample_contrasts.mean_difference
def compare(reference, target, settings):
    return workbench.sample_contrasts.state_comparison(reference, target, settings, metrics=METRICS)
