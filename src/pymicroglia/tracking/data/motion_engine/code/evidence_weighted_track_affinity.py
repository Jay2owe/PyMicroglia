"""Scale remembered signal-name affinity by present physical evidence."""
import numpy as np


def continuity_fraction(state, observation, support):
    """Overlap or paired departure/arrival, in dimensionless body fractions.

    The motion evidence is already calibrated upstream. A remembered name
    supplies no movement evidence of its own. A gap without measured motion
    can retain only the overlap that is actually present.
    """
    a=state.bounds
    b=observation.get('bounds')
    if a is None or b is None:
        intersection=int(np.count_nonzero(state.mask & observation['mask']))
    else:
        y0,y1=max(a[0],b[0]),min(a[1],b[1])
        x0,x1=max(a[2],b[2]),min(a[3],b[3])
        intersection=(int(np.count_nonzero(state.mask[y0:y1,x0:x1] & observation['mask'][y0:y1,x0:x1]))
                      if y0<y1 and x0<x1 else 0)
    overlap=intersection/max(float(state.area)+float(observation['area'])-intersection,1.)
    movement=min(float(support['source_loss_coverage']),float(support['destination_gain_coverage']))
    return max(overlap,movement)
