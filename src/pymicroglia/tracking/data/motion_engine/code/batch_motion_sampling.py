"""Experimental exact batching of existing seven-point motion queries.

All linking costs, candidate order and motion sample positions remain unchanged.
The batches contain the existing geometric candidates, never selected identities.
"""
import ast
from contextlib import contextmanager
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from physical_tracking_speed import candidate_indices


class MotionBatch:
    def __init__(self, fallback):
        self.fallback=fallback
        self.weights=np.linspace(0.,1.,7)
        self.previous=1-self.weights
        self.plane=None
        self.maxima=None
        self.prepared=None

    def prepare(self,plane,x0,y0,detections):
        self.prepared=None
        if plane is None or not detections:
            return
        if plane.ndim!=2 or plane.dtype.kind not in 'biuf' or (plane.dtype.kind=='f' and plane.dtype.itemsize not in (4,8)):
            return
        coordinates=[x0,y0]+[v for row in detections for v in (row['x'],row['y'])]
        if not all(isinstance(v,(int,float,np.integer,np.floating))
                   and not (isinstance(v,np.floating) and v.dtype.itemsize>8)
                   for v in coordinates):
            return
        try:
            targets=np.asarray([(row['x'],row['y']) for row in detections],dtype=np.float64)
            start=np.asarray([x0,y0],dtype=np.float64)
            if not np.isfinite(targets).all() or not np.isfinite(start).all():
                return
        except (TypeError,ValueError,OverflowError):
            return
        if self.plane is not plane:
            self.plane=plane;self.maxima=None
            if not np.isfinite(plane).all():
                return
            fill=np.iinfo(plane.dtype).min if plane.dtype.kind in 'iu' else (-np.inf if plane.dtype.kind=='f' else 0)
            self.maxima=ndi.maximum_filter(np.pad(plane,1,mode='constant',constant_values=fill),size=3,mode='constant',cval=fill)
        if self.maxima is None:
            return
        x=np.rint(self.previous[None,:]*x0+self.weights[None,:]*targets[:,0,None])
        y=np.rint(self.previous[None,:]*y0+self.weights[None,:]*targets[:,1,None])
        valid=(x>=-1)&(x<=plane.shape[1])&(y>=-1)&(y<=plane.shape[0])
        samples=np.full(x.shape,-np.inf,dtype=np.float64)
        samples[valid]=self.maxima[y[valid].astype(np.intp)+1,x[valid].astype(np.intp)+1]
        values=np.where(valid.any(axis=1),samples.max(axis=1),0.)
        values=np.clip(values/65535.,0.,1.)
        self.prepared=(plane,x0,y0,{(float(x),float(y)):float(value) for (x,y),value in zip(targets,values)})

    def support(self,plane,x0,y0,x1,y1):
        if self.prepared is not None:
            saved_plane,saved_x,saved_y,values=self.prepared
            try:
                if plane is saved_plane and x0==saved_x and y0==saved_y:
                    return values[(x1,y1)]
            except (KeyError,TypeError,ValueError):
                pass
        return self.fallback(plane,x0,y0,x1,y1)


@contextmanager
def batched_motion_linker(module):
    """Replace only the compiled linker while retaining all surrounding contexts."""
    original=module.link_observations
    tree=ast.parse(Path(module.__file__).read_text())
    function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='link_observations')
    matches=[]
    for node in ast.walk(function):
        if isinstance(node,ast.For) and ast.unparse(node.target)=='(di, detection)' and ast.unparse(node.iter)=='enumerate(detections)':
            if any(isinstance(v,ast.Name) and v.id=='endpoint_distance' for v in ast.walk(node)):
                matches.append(node)
    # ast.unparse tuple parentheses vary across Python versions; inspect the AST itself.
    if not matches:
        matches=[node for node in ast.walk(function) if isinstance(node,ast.For)
            and isinstance(node.target,ast.Tuple) and [v.id for v in node.target.elts]==['di','detection']
            and ast.unparse(node.iter)=='enumerate(detections)'
            and any(isinstance(v,ast.Name) and v.id=='endpoint_distance' for v in ast.walk(node))]
    assert len(matches)==1,'Linker structure changed; inspect the new execution path'
    matches[0].iter=ast.parse('_prepared_candidates(detections,px,py,maximum_distance,motion_plane,last)',mode='eval').body
    motion=MotionBatch(module._motion_support_plane);cache={}
    def prepare(detections,px,py,distance,plane,last):
        pairs=list(candidate_indices(detections,px,py,distance,cache))
        motion.prepare(plane,last['x'],last['y'],[row for _,row in pairs])
        return pairs
    namespace=dict(vars(module));namespace.update(_prepared_candidates=prepare,_motion_support_plane=motion.support)
    compiled=ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]))
    exec(compile(compiled,str(module.__file__)+'::batch_motion','exec'),namespace)
    module.link_observations=namespace['link_observations']
    try:
        yield
    finally:
        module.link_observations=original
