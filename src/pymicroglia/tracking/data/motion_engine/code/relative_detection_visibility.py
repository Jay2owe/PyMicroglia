"""Experimental scale-free visibility ratio; native equivalence required before use."""
import ast
from contextlib import contextmanager
import inspect
import numpy as np


def relative_visibility(peak,weak,strong):
    if not np.isfinite([peak,weak,strong]).all():
        return float(np.clip((peak-weak)/max(strong-weak,np.finfo(np.float32).eps),0.,1.))
    span=strong-weak
    if span>0:return float(np.clip((peak-weak)/span,0.,1.))
    return float(peak>weak)


@contextmanager
def relative_detection_visibility(module, *, reuse_core_structure=False):
    original=module.detect_frame;tree=ast.parse(inspect.getsource(original))
    matches=[node for node in ast.walk(tree) if isinstance(node,ast.Assign)
             and len(node.targets)==1 and isinstance(node.targets[0],ast.Name)
             and node.targets[0].id=='visibility']
    assert len(matches)==1,'Detector source changed; equivalence review required'
    matches[0].value=ast.parse('_relative_visibility(peak, weak, strong)',mode='eval').body
    namespace=dict(vars(module));namespace['_relative_visibility']=relative_visibility
    if reuse_core_structure:
        calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call)
               and ast.unparse(node.func)=='ndi.label' and len(node.args)==1
               and ast.unparse(node.args[0])=='core' and not node.keywords]
        assert len(calls)==1,'Core connectivity call changed; equivalence review required'
        calls[0].keywords=[ast.keyword(arg='structure',value=ast.Name(id='_core_structure',ctx=ast.Load()))]
        structure=module.ndi.generate_binary_structure(2,1);structure.flags.writeable=False
        namespace['_core_structure']=structure
    exec(compile(ast.fix_missing_locations(tree),original.__code__.co_filename+'::relative_visibility','exec'),namespace)
    module.detect_frame=namespace['detect_frame']
    module.detect_frame._motion_relative_visibility=True
    module.detect_frame._motion_reused_core_structure=reuse_core_structure
    try:yield
    finally:module.detect_frame=original
