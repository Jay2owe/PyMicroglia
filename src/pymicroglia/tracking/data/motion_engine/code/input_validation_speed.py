"""Read dimensions without decoding already hash-verified input pixels."""
from contextlib import contextmanager
import tifffile


@contextmanager
def header_only_validation(pipeline):
    original=pipeline._verify_pins
    # The verification function uses loaded arrays only for shape/ndim.
    # Clone its globals instead of changing the module loader shared by jobs.
    import ast,inspect,types
    tree=ast.parse(inspect.getsource(original))
    arrays={'raw','lag','neutral','trails','ages'}
    attributes=[node for node in ast.walk(tree) if isinstance(node,ast.Attribute)
                and isinstance(node.value,ast.Name) and node.value.id in arrays]
    assert all(node.attr in ('shape','ndim') for node in attributes)
    calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call)
           and isinstance(node.func,ast.Name) and node.func.id=='load_stack']
    assert len(calls)==5,'Input validation changed; review required'
    class Shape:
        def __init__(self,path):
            with tifffile.TiffFile(path) as image:
                self.shape=tuple(image.series[0].shape)
            self.ndim=len(self.shape)
        def __len__(self):return self.shape[0]
    namespace=dict(original.__globals__);namespace['load_stack']=Shape
    candidate=types.FunctionType(original.__code__,namespace,original.__name__,
                                 original.__defaults__,original.__closure__)
    pipeline._verify_pins=candidate
    try:yield
    finally:pipeline._verify_pins=original
