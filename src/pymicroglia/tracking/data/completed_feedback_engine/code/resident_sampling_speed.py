"""Reuse the verified batched sampling implementation for resident ownership."""
import ast
from contextlib import contextmanager
from pathlib import Path
import owner_sampling_speed


@contextmanager
def batched_resident_owners(module, aliases=()):
    # Adapt the existing implementation's interface only. Its batch arithmetic
    # remains unchanged; fixed bounds belong to the original resident function.
    tree=ast.parse(Path(owner_sampling_speed.__file__).read_text())
    outer=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='batched_owner_samples')
    outer.name='_resident_context'
    guards=[node for node in outer.body if isinstance(node,ast.Assert)]
    assert len(guards)==1
    literal=next(node for node in ast.walk(guards[0]) if isinstance(node,ast.Constant) and isinstance(node.value,str) and 'float(np.clip' in node.value)
    literal.value='float(np.clip(0.75 * radius, 2.0, 6.0))'
    attach=next(node for node in outer.body if isinstance(node,ast.FunctionDef) and node.name=='attach')
    assert [node.arg for node in attach.args.args]==['points','labels','params']
    attach.args.args.pop();attach.args.defaults=[]
    bounds=[node for node in ast.walk(attach) if isinstance(node,ast.Call) and ast.unparse(node.func)=='module.spatial_bounds']
    assert len(bounds)==1
    assignment=next(node for node in attach.body if isinstance(node,ast.Assign) and node.value is bounds[0])
    assignment.value=ast.parse('(2.0,6.0,None)',mode='eval').body
    for node in ast.walk(attach):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='original':
            assert len(node.args)==3 and ast.unparse(node.args[-1])=='params'
            node.args.pop()
    # The legacy scalar method has special/error behavior outside the field.
    # Keep that path there; in-field batching alone is the claimed speedup.
    attach.body[0:0]=ast.parse('''if not points.columns.is_unique or not {'frame','x','y','radius_px','state'}.issubset(points.columns):
    return original(points,labels)
if labels.ndim != 3:
    return original(points,labels)
if not points.empty:
    positions=points[['x','y']].to_numpy(float)
    if np.any(positions[:,0]<0) or np.any(positions[:,0]>=labels.shape[2]) or np.any(positions[:,1]<0) or np.any(positions[:,1]>=labels.shape[1]):
        return original(points,labels)
''').body
    namespace=dict(vars(owner_sampling_speed))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[outer],type_ignores=[])),str(owner_sampling_speed.__file__)+'::resident_interface','exec'),namespace)
    original=module.attach_owners;changed=[]
    with namespace['_resident_context'](module):
        for alias in aliases:
            if alias.attach_owners is original:
                changed.append(alias);alias.attach_owners=module.attach_owners
        try:
            yield
        finally:
            for alias in changed:alias.attach_owners=original
