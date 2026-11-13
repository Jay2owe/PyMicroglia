"""Build the unchanged endpoint table once per lifetime discovery call."""
import ast
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from types import FunctionType


@contextmanager
def cached_lifetime_endpoints(module):
    original = module.discover
    path = Path(module.__file__)
    tree = ast.parse(path.read_text(encoding='utf8'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == '_endpoint_compatibility')
    assignments = [n for n in function.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'endpoints' for t in n.targets)
                   and isinstance(n.value, ast.Call)
                   and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == 'agg']
    assert len(assignments) == 1, 'Endpoint aggregation changed; review required'
    expression = compile(ast.Expression(assignments[0].value), str(path), 'eval')
    assignments[0].value = ast.Call(func=ast.Name(id='_cached_endpoint_table', ctx=ast.Load()),
                                  args=[ast.Name(id='all_points', ctx=ast.Load())], keywords=[])
    compiled = compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                       str(path)+'::cached_endpoint_table', 'exec')
    statistics = []

    @wraps(original)
    def discover(*args, **kwargs):
        tables = {}
        counts = dict(endpoint_calls=0, endpoint_table_builds=0)
        namespace = dict(original.__globals__)

        def table(points):
            counts['endpoint_calls'] += 1
            key = id(points)
            if key not in tables:
                tables[key] = (points, eval(expression, namespace, {'all_points': points}))
                counts['endpoint_table_builds'] += 1
            saved, value = tables[key]
            assert saved is points
            return value

        namespace['_cached_endpoint_table'] = table
        exec(compiled, namespace)
        local = FunctionType(original.__code__, namespace, original.__name__,
                             original.__defaults__, original.__closure__)
        local.__kwdefaults__ = original.__kwdefaults__
        try:
            return local(*args, **kwargs)
        finally:
            tables.clear()
            counts['retained_tables_after_call'] = 0
            statistics.append(counts)

    module.discover = discover
    try:
        yield statistics
    finally:
        module.discover = original
