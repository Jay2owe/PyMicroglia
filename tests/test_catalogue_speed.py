"""Catalogue enumeration shares verification without retaining stale metadata."""
import pytest


def test_catalogue_capture_is_per_request_and_verifies_completion(monkeypatch):
    from pymicroglia import knowledge, workbench
    from copy import deepcopy
    calls=[]
    original=workbench.cw.run_records.software_identity
    def counted(*args,**kwargs):
        calls.append(1)
        return original(*args,**kwargs)
    monkeypatch.setattr(workbench.cw.run_records,'software_identity',counted)
    monkeypatch.setattr(knowledge.REGISTRY,'names',lambda:['one','two','three','four'])
    monkeypatch.setattr(knowledge,'_entry',lambda name:{'name':name,'methods':workbench.cw.call('period_methods').data})
    first=knowledge.describe()
    first_calls=len(calls)
    assert first_calls==2  # capture plus fresh completion verification
    reference=deepcopy(first)
    first['actions'][0]['methods']['methods'].clear()
    second=knowledge.describe()
    assert second==reference and len(calls)==2*first_calls
    assert workbench._ANALYSIS_CAPTURE.get() is None


def test_catalogue_rejects_changed_environment_and_resets_scope(monkeypatch):
    from pymicroglia import knowledge, workbench
    monkeypatch.setattr(knowledge.REGISTRY,'names',lambda:['one'])
    monkeypatch.setattr(knowledge,'_entry',lambda name:{'name':name})
    def changed(expected):
        raise ValueError('environment changed')
    monkeypatch.setattr(workbench,'verify_rhythm_environment',changed)
    with pytest.raises(ValueError,match='environment changed'):
        knowledge.describe()
    assert workbench._ANALYSIS_CAPTURE.get() is None
