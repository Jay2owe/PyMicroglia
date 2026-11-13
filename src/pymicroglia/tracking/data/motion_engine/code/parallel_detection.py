"""Run independent frame detections concurrently, preserving exact frame order."""
import ast
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import inspect
import os


def detect_frames(raw,params,detect_frame,workers=None):
    count=max(1,min(4,os.cpu_count() or 1)) if workers is None else max(1,int(workers))
    def detect(item):
        index,frame=item
        return detect_frame(frame,index,params)
    if count==1:
        yield from map(detect,enumerate(raw))
    else:
        with ThreadPoolExecutor(max_workers=count) as executor:
            yield from executor.map(detect,enumerate(raw))


@contextmanager
def parallel_detection(history):
    original=history._owner_consensus
    tree=ast.parse(inspect.getsource(original))
    matches=[node for node in ast.walk(tree) if isinstance(node,ast.For)
             and ast.unparse(node.target)=='(frame_index, frame)'
             and ast.unparse(node.iter)=='enumerate(raw)']
    assert len(matches)==1,'Physical detector loop changed; equivalence review required'
    loop=matches[0]
    assignments=[node for node in loop.body if isinstance(node,ast.Assign)
                 and ast.unparse(node.targets[0])=='(rows, threshold)'
                 and isinstance(node.value,ast.Call)
                 and ast.unparse(node.value.func)=='raw_physical_hypotheses.detect_frame']
    assert len(assignments)==1,'Detector assignment changed; equivalence review required'
    loop.body.remove(assignments[0]);loop.target=ast.parse('rows, threshold',mode='eval').body
    loop.target.ctx=ast.Store()
    for item in loop.target.elts:item.ctx=ast.Store()
    loop.iter=ast.parse('_parallel_detections(raw, raw_params)',mode='eval').body
    namespace=dict(vars(history))
    namespace['_parallel_detections']=lambda raw,params:detect_frames(raw,params,history.raw_physical_hypotheses.detect_frame)
    exec(compile(ast.fix_missing_locations(tree),original.__code__.co_filename+'::parallel_detection','exec'),namespace)
    history._owner_consensus=namespace[original.__name__]
    try:yield
    finally:history._owner_consensus=original
