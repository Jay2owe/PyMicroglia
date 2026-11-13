"""Avoid reconnection pairs outside the unchanged elapsed-frame interval."""
import ast
from contextlib import contextmanager
import inspect
import numpy as np


@contextmanager
def spatial_reconnections(module):
    original=module.reconnection_hypotheses
    tree=ast.parse(inspect.getsource(original))
    loops=[node for node in ast.walk(tree) if isinstance(node,ast.For)
           and isinstance(node.target,ast.Name) and node.target.id=='after_id'
           and ast.unparse(node.iter)=='track_ids']
    rows=[node for node in ast.walk(tree) if isinstance(node,ast.Assign)
          and len(node.targets)==1 and isinstance(node.targets[0],ast.Name)
          and node.targets[0].id=='startpoint' and ast.unparse(node.value)=='after.iloc[0]']
    assert len(loops)==len(rows)==1,'Reconnection source changed; equivalence review required'
    loops[0].iter=ast.parse('_eligible_after(grouped, track_ids, end_frame, maximum_gap, endpoint, maximum_distance, base_distance, gap_distance)',mode='eval').body
    rows[0].value=ast.parse('_start_row(after)',mode='eval').body
    cache={};row_cache={}
    def start_row(frame):
        key=id(frame)
        if key not in row_cache:row_cache[key]=(frame,frame.iloc[0])
        assert row_cache[key][0] is frame
        return row_cache[key][1]
    def eligible(grouped,track_ids,end_frame,maximum_gap,endpoint,maximum_distance,base_distance,gap_distance):
        if cache.get('grouped') is not grouped:
            starts=np.array([int(start_row(grouped[key]).frame) for key in track_ids],dtype=object)
            cache.update(grouped=grouped,starts=starts,xy=np.array([(float(start_row(grouped[key]).x),float(start_row(grouped[key]).y)) for key in track_ids]))
        starts=cache['starts']
        selected=(starts>=end_frame+1)&(starts<=end_frame+1+maximum_gap)
        xy=cache['xy'];px,py=float(endpoint.x),float(endpoint.y)
        if len(xy) and np.isfinite(xy).all() and np.isfinite([px,py,maximum_distance,base_distance,gap_distance]).all():
            gaps=np.array([int(start)-end_frame for start in starts],dtype=float)
            limits=np.nextafter(np.minimum(maximum_distance,base_distance+gap_distance*gaps),np.inf)
            selected &= (np.abs(xy[:,0]-px)<=limits)&(np.abs(xy[:,1]-py)<=limits)
        return (track_ids[int(i)] for i in np.flatnonzero(selected))
    namespace=dict(vars(module));namespace.update(_eligible_after=eligible,_start_row=start_row)
    exec(compile(ast.fix_missing_locations(tree),original.__code__.co_filename+'::bounded_reconnections','exec'),namespace)
    module.reconnection_hypotheses=namespace['reconnection_hypotheses']
    try:yield
    finally:module.reconnection_hypotheses=original
