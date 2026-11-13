"""Relative contrast for visibility between observed cell positions."""
import ast
from contextlib import contextmanager
import inspect
import numpy as np
from relative_detection_visibility import relative_visibility


def latent_visibility(peak,weak,strong):
    if not np.isfinite([peak,weak,strong]).all():
        return float(np.clip((peak-weak)/max(strong-weak,1e-6),0.,1.))
    return relative_visibility(peak,weak,strong)


@contextmanager
def relative_latent_visibility(module):
    original=module.expand_track_points;tree=ast.parse(inspect.getsource(original))
    matches=[node for node in ast.walk(tree) if isinstance(node,ast.Assign)
             and len(node.targets)==1 and isinstance(node.targets[0],ast.Name)
             and node.targets[0].id=='visibility' and isinstance(node.value,ast.Call)
             and ast.unparse(node.value.func)=='float']
    assert len(matches)==1,'Latent visibility source changed; equivalence review required'
    matches[0].value=ast.parse('_relative_visibility(evidence_value, weak, strong_threshold)',mode='eval').body
    namespace=dict(vars(module));namespace['_relative_visibility']=latent_visibility
    exec(compile(ast.fix_missing_locations(tree),original.__code__.co_filename+'::relative_latent_visibility','exec'),namespace)
    module.expand_track_points=namespace['expand_track_points']
    module.expand_track_points._motion_relative_latent_visibility=True
    try:yield
    finally:module.expand_track_points=original
