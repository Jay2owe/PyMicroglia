"""Compatibility names for statistics owned by Circadian Workbench."""
from pymicroglia import workbench
SPATIAL_NULL = workbench.spatial_permutation.SPATIAL_NULL
SOURCES = workbench.spatial_permutation.SOURCES
def _call(function,*args,**kwargs):
    try:
        return function(*args,**kwargs)
    except workbench.cw.WorkbenchUnavailableError as error:
        from ..pipelines._runner import Unavailable
        raise Unavailable(str(error)) from error


def validate_spatial(question):
    return _call(workbench.spatial_permutation.validate_spatial,question)
pair_difference = workbench.spatial_permutation.pair_difference
distance_effect = workbench.spatial_permutation.distance_effect
def spatial_test(*args,**kwargs):
    return _call(workbench.spatial_permutation.spatial_test,*args,**kwargs)
