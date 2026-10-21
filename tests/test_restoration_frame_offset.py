"""The actual frozen frame-clock check must accept full recordings from zero."""
import ast
from contextlib import contextmanager
from importlib import resources
from types import SimpleNamespace
import zipfile

import pytest

from pymicroglia.tracking.completed_evidence import _restoration_context


@pytest.mark.parametrize('offset',[0,2,5])
def test_actual_frozen_restoration_receives_source_frame_offset(offset):
    archive=resources.files('pymicroglia.tracking').joinpath('data/completed_feedback_engine.zip')
    with zipfile.ZipFile(archive.open('rb')) as z:
        source=z.read('frozen/final/r214_a003/measured_restoration_continuity_a002.py').decode()
    tree=ast.parse(source)
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='measured_restoration_continuity')
    recorded=[]
    class Continuity:
        def __init__(self,*args):pass
        def propose(self,previous,current,frame,proposals):return proposals
        def apply(self,original,*args):return original(*args)
        def record(self,current,result,frame,actions):recorded.append(frame)
    namespace=dict(contextmanager=contextmanager,CoupledRestoredContinuity=Continuity,geometry_by_frame=lambda _:None)
    exec(compile(ast.Module(body=[function],type_ignores=[]),'actual_frozen_context','exec'),namespace)
    overlap=lambda *args:[]
    apply=lambda current,*args:(current,[])
    ownership=SimpleNamespace(overlap_proposals=overlap,apply_proposals=apply)
    memory=SimpleNamespace(measured_restoration_continuity=namespace['measured_restoration_continuity'])
    with _restoration_context(memory,ownership,None,[],offset):
        ownership.overlap_proposals(None,None,None,None,None)
        ownership.apply_proposals(None,[],'R',None,offset+2)
    assert recorded==[1]
    assert ownership.overlap_proposals is overlap and ownership.apply_proposals is apply
