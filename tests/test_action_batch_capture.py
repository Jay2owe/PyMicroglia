"""Public actions share provenance work without accepting changed environments."""
from copy import deepcopy
import pytest


@pytest.fixture
def isolated(monkeypatch,tmp_path):
    monkeypatch.setenv('PYMICROGLIA_INDEX',str(tmp_path/'index'))
    monkeypatch.setenv('PYMICROGLIA_STORE',str(tmp_path/'store'))
    import pymicroglia.run as runner
    monkeypatch.setattr(runner,'validate',lambda *a,**k:{})
    return runner


def test_batch_reuses_identity_but_each_child_keeps_its_record(isolated,monkeypatch):
    from pymicroglia import workbench
    records=workbench.cw.run_records;real=records.software_identity;calls=[]
    def identity():
        value=real();calls.append(value);return value
    monkeypatch.setattr(records,'software_identity',identity)
    def action():return [workbench.cw.call('period_methods').run_record for _ in range(5)]
    monkeypatch.setattr(isolated.REGISTRY,'resolve',lambda name:action)
    completed=isolated.run_action('null_channel_phase_test',claim='Software boundary verification')
    assert len(calls)==2
    assert len(completed)==5
    assert all(row['environment']==calls[0] for row in completed)
    assert len({id(row) for row in completed})==5
    assert workbench._ANALYSIS_CAPTURE.get() is None


def test_changed_engine_is_refused_before_action_returns(isolated,monkeypatch):
    from pymicroglia import workbench
    records=workbench.cw.run_records;real=records.software_identity;calls=[]
    def identity():
        value=deepcopy(real());calls.append(None)
        if len(calls)>1:
            group=next(iter(value));key=next(iter(value[group]));value[group][key]=str(value[group][key])+' changed'
        return value
    monkeypatch.setattr(records,'software_identity',identity)
    monkeypatch.setattr(isolated.REGISTRY,'resolve',lambda name:lambda:workbench.cw.call('period_methods'))
    with pytest.raises(ValueError,match='environment differs'):
        isolated.run_action('null_channel_phase_test',claim='Software boundary verification')
    assert workbench._ANALYSIS_CAPTURE.get() is None


def test_failed_batch_cannot_leave_a_capture_in_the_next_action(isolated,monkeypatch):
    from pymicroglia import workbench
    def failure():
        assert workbench._ANALYSIS_CAPTURE.get() is not None
        raise RuntimeError('failed operation')
    monkeypatch.setattr(isolated.REGISTRY,'resolve',lambda name:failure)
    with pytest.raises(RuntimeError,match='failed operation'):
        isolated.run_action('null_channel_phase_test',claim='Software boundary verification')
    assert workbench._ANALYSIS_CAPTURE.get() is None


def test_simple_image_action_does_not_open_rhythm_capture(isolated,monkeypatch):
    from pymicroglia import workbench
    def forbidden():raise AssertionError('Image action must not initialize the rhythm engine')
    monkeypatch.setattr(workbench,'analysis_session',forbidden)
    monkeypatch.setattr(isolated.REGISTRY,'resolve',lambda name:lambda:42)
    assert isolated.run_action('display_filter')==42
