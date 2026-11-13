"""Reuse row tuple types while retaining owner-run sorting, values and boundaries."""
import ast
from collections import namedtuple
from contextlib import contextmanager
import inspect


@contextmanager
def cached_owner_row_types(module):
    original=module._runs;tree=ast.parse(inspect.getsource(original));types={}
    matches=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='itertuples']
    assert len(matches)==1,'Owner-run row iteration changed; equivalence review required'
    call=matches[0];assert not call.args and len(call.keywords)==1 and call.keywords[0].arg=='index' and isinstance(call.keywords[0].value,ast.Constant) and call.keywords[0].value.value is False
    frame=call.func.value;call.func=ast.Name(id='_cached_tuples',ctx=ast.Load());call.args=[frame];call.keywords=[]
    def tuples(frame):
        fields=tuple(frame.columns)
        if fields not in types:types[fields]=namedtuple('Pandas',fields,rename=True)
        make=types[fields]._make
        return map(make, zip(*(column for _, column in frame.items())))
    namespace=dict(vars(module));namespace['_cached_tuples']=tuples
    exec(compile(ast.fix_missing_locations(tree),original.__code__.co_filename+'::cached_row_types','exec'),namespace)
    module._runs=namespace['_runs']
    try:yield
    finally:module._runs=original
