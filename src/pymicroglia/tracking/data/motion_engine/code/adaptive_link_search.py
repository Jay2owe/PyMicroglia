"""Conservative search bound from the linker's existing movement budget.

The observed motion score is in [0, 1]. A field maximum bounds every sampled
score, so candidates outside the largest possible allowance cannot be assigned.
The original per-pair checks and assignment objective still run unchanged.
"""
import ast
from functools import lru_cache
from contextlib import contextmanager
from pathlib import Path
import numpy as np
from physical_tracking_speed import candidate_indices


def search_radius(maximum,allowed,extension,plane,cache):
    if not np.isfinite([maximum,allowed,extension]).all():
        return maximum
    if plane is None:
        upper=0.
    else:
        if cache.get('plane') is not plane:
            cache.clear();cache['plane']=plane;cache['upper']=None
            if plane.dtype.kind in 'biuf' and np.isfinite(plane).all():
                peak=float(np.max(plane)) if plane.size else 0.
                cache['upper']=float(np.clip(peak/65535.,0.,1.))
        upper=cache['upper']
        if upper is None:
            return maximum
    return min(maximum,allowed+max(extension,0.)*upper)


@contextmanager
def adaptive_link_search(module):
    original=module.link_observations
    tree=ast.parse(Path(module.__file__).read_text())
    function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='link_observations')
    matches=[node for node in ast.walk(function) if isinstance(node,ast.For)
        and isinstance(node.target,ast.Tuple) and [v.id for v in node.target.elts]==['di','detection']
        and ast.unparse(node.iter)=='enumerate(detections)'
        and any(isinstance(v,ast.Name) and v.id=='endpoint_distance' for v in ast.walk(node))]
    assert len(matches)==1,'Linker structure changed; review search-bound integration'
    matches[0].iter=ast.parse('_within_budget(detections,px,py,maximum_distance,allowed,motion_extension,motion_plane)',mode='eval').body
    detection_cache={};plane_cache={}
    patch_source = module._patch
    patch_state = {}
    @lru_cache(maxsize=8192)
    def cached_patch(frame_index, x, y):
        return patch_source(patch_state['raw'][frame_index], x, y)
    def reset_patches(raw):
        cached_patch.cache_clear()
        patch_state.clear()
        patch_state['raw'] = raw
    patch_calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Name) and node.func.id == '_patch']
    assert len(patch_calls) == 2, 'Appearance calls changed; equivalence review required'
    for call in patch_calls:
        assert len(call.args) == 3 and not call.keywords
        image = call.args[0]
        assert isinstance(image, ast.Subscript) and ast.unparse(image.value) == 'raw'
        call.func = ast.Name(id='_cached_patch', ctx=ast.Load())
        call.args[0] = image.slice
    function.body.insert(0, ast.parse('_reset_patches(raw)').body[0])
    def within(detections,px,py,maximum,allowed,extension,plane):
        bound=search_radius(maximum,allowed,extension,plane,plane_cache)
        return candidate_indices(detections,px,py,bound,detection_cache)
    namespace=dict(vars(module));namespace['_within_budget']=within
    namespace['_cached_patch']=cached_patch
    namespace['_reset_patches']=reset_patches
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),
                 str(module.__file__)+'::adaptive_search','exec'),namespace)
    module.link_observations=namespace['link_observations']
    try:
        yield
    finally:
        module.link_observations=original
