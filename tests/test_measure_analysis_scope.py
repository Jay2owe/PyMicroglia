"""Measurement shares engine setup only when its resolved config asks for rhythms."""
import importlib
import json
import pytest


@pytest.mark.parametrize('from_file',[False,True])
def test_measurement_config_shares_identity_and_keeps_each_record(tmp_path,monkeypatch,from_file):
    api=importlib.import_module('pymicroglia.measure')
    engine=importlib.import_module('pymicroglia.measure.run')
    from pymicroglia import workbench
    real=workbench.cw.run_records.software_identity;identities=[]
    def identity():
        row=real();identities.append(row);return row
    monkeypatch.setattr(workbench.cw.run_records,'software_identity',identity)
    def measurement(config,*args,**kwargs):
        assert config.enabled_modules==['rhythms']
        return [workbench.cw.call('period_methods').run_record for _ in range(5)]
    monkeypatch.setattr(engine,'run',measurement)
    params={'movies':[{'stem':'one','raw':'raw.tif','labels':'labels.tif'}],
            'enabled_modules':['rhythms'],'frame_interval_min':30}
    if from_file:
        source=tmp_path/'analysis.json';source.write_text(json.dumps(params))
        params={'analysis_config':source}
    records=api.measure(output_dir=tmp_path/'out',**params)
    assert len(identities)==2
    assert len(records)==5 and len({id(r) for r in records})==5
    assert all(r['environment']==identities[0] for r in records)
    assert workbench._ANALYSIS_CAPTURE.get() is None


def test_measurement_without_rhythms_does_not_open_engine_capture(tmp_path,monkeypatch):
    api=importlib.import_module('pymicroglia.measure')
    engine=importlib.import_module('pymicroglia.measure.run')
    from pymicroglia import workbench
    def forbidden():raise AssertionError('No rhythm module requested')
    monkeypatch.setattr(workbench,'analysis_session',forbidden)
    monkeypatch.setattr(engine,'run',lambda *args,**kwargs:42)
    assert api.measure([{'stem':'one','raw':'raw.tif','labels':'labels.tif'}],
        output_dir=tmp_path/'out',enabled_modules=['morphology'],frame_interval_min=30)==42
