"""Soma displacement for a temporary expansion which separates into bodies."""
import numpy as np
from observed_soma_reservation import _inside


class SeparatingOutlineMotion:
    def __init__(self, cores, labels, direction):
        self.cores, self.labels, self.direction = cores, labels, direction
        self.areas = [np.bincount(frame.ravel()) for frame in labels]

    def update(self, state, observation, target_t, prior_states):
        steps=max(abs(target_t-state.t),1)
        position=observation['position'].copy()
        velocity=.65*(position-state.position)/steps+.35*state.velocity
        next_t=target_t+self.direction
        if steps!=1 or not 0<=next_t<len(self.labels) or observation['area']<=state.area:
            return position,velocity,None
        earlier=[(key,y,x) for key,y,x in self.cores.get(state.t,[]) if _inside(state.mask,y,x)]
        if len(earlier)!=1:
            return position,velocity,None
        key,old_y,old_x=earlier[0]
        now=[(y,x) for track,y,x in self.cores.get(target_t,[]) if track==key]
        later=[(y,x) for track,y,x in self.cores.get(next_t,[]) if track==key]
        if len(now)!=1 or len(later)!=1 or not _inside(observation['mask'],*now[0]):
            return position,velocity,None
        y,x=later[0]
        if not _inside(observation['mask'],y,x):
            return position,velocity,None
        resident=int(self.labels[next_t,y,x])
        children=np.unique(self.labels[next_t][observation['mask']]);children=children[children>0]
        if resident<=0 or len(children)<2 or np.any(self.areas[next_t][children]>=observation['area']):
            return position,velocity,None
        # Resident and another independently segmented child must both overlap
        # the expanded outline. No pixel distance or area threshold is fitted.
        if resident not in children:
            return position,velocity,None
        delta=np.array(now[0],float)-np.array([old_y,old_x],float)
        measured=state.position+delta
        return measured,.65*delta+.35*state.velocity,dict(raw_track=int(key),
            measured_displacement=float(np.linalg.norm(delta)),
            outline_displacement=float(np.linalg.norm(position-state.position)),
            measured_position_y=float(measured[0]),measured_position_x=float(measured[1]),
            separating_children=len(children),evidence='observed_expansion_and_separation')
