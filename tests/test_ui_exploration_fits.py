"""A dashed overlay evaluates stored models and cannot silently fit a new one."""
import csv
import hashlib
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from pymicroglia.ui.exploration_data import recording, table_path
from pymicroglia.ui.exploration_fits import load_rhythm_fits
from pymicroglia.ui.exploration_rhythms import _folder
from pymicroglia.ui.jobs import JobManager
from pymicroglia.ui.server import create_app
from test_ui_exploration_data import saved_run


@pytest.fixture
def model_run(saved_run):
    root, _, movie = recording(saved_run, 'one')
    source = table_path(root, movie, 'cell_frame')
    folder = _folder(root, 'one') / ('a' * 32)
    folder.mkdir(parents=True)
    diagnostics = {'status': 'ok', 'mesor': 0,
                   'phase_zero_timestamp': '2000-01-01T03:00:00'}
    components = [{'period_hours': 24, 'amplitude': 2, 'phase_hours': 6},
                  {'period_hours': 8, 'amplitude': 1, 'phase_hours': 2}]
    row = {'identity': 19, 'metric': 'area_px', 'estimate_status': 'ok',
           'estimation_method': 'fft_nlls', 'estimate_diagnostics_json': json.dumps(diagnostics),
           'estimate_components_json': json.dumps(components),
           'applied_settings_json': json.dumps({'detrend': 'robust_linear'})}
    with (folder/'statistics.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(row))
        writer.writeheader();writer.writerow(row)
    result = {'source':str(folder/'result.json'), 'analysis_id':'a'*32, 'metrics':['area_px'],
              'rows':[], 'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'movie_sha256':hashlib.sha256(json.dumps(movie,sort_keys=True).encode()).hexdigest(),
              'statistics_sha256':hashlib.sha256((folder/'statistics.csv').read_bytes()).hexdigest()}
    (folder/'result.json').write_text(json.dumps(result))
    (folder.parent/'latest.json').write_text(json.dumps({'result':('a'*32)+'/result.json'}))
    return root, folder, source


def test_exact_multicomponent_model_restores_units_clock_and_gaps_without_refitting(model_run, monkeypatch):
    from pymicroglia import workbench
    root, folder, source = model_run
    before = {path:path.read_bytes() for path in root.rglob('*') if path.is_file()}
    def forbid_fit(*args,**kwargs):
        raise AssertionError('Displaying a saved model must not fit or test a rhythm')
    monkeypatch.setattr(workbench,'estimate_one',forbid_fit)
    monkeypatch.setattr(workbench,'detrend_trace',forbid_fit)
    monkeypatch.setattr(workbench.cw.analysis,'_fit_damped_oscillation',forbid_fit)
    result = load_rhythm_fits(root,'one','area_px')
    curve = result['rows'][0]
    assert curve['identity']==19 and 3 not in curve['indices']
    hours = (np.array(curve['indices'])+5)/2
    expected = 20 + 2*np.cos(2*np.pi*(hours-3-6)/24) + np.cos(2*np.pi*(hours-3-2)/8)
    np.testing.assert_allclose(curve['values'],expected,atol=1e-10)
    assert before == {path:path.read_bytes() for path in root.rglob('*') if path.is_file()}
    # Older saved analyses predate the evidence fingerprint but remain readable.
    path=folder/'result.json';saved=json.loads(path.read_text());saved.pop('statistics_sha256')
    path.write_text(json.dumps(saved))
    assert load_rhythm_fits(root,'one','area_px')['rows']==result['rows']


def test_missing_unmodelled_or_changed_evidence_is_not_replaced(model_run):
    root, folder, source = model_run
    assert load_rhythm_fits(root,'one','integrated_density')['rows']==[]
    with pytest.raises(ValueError,match='not available'):
        load_rhythm_fits(root,'one','missing')
    path=folder/'statistics.csv';original=path.read_bytes();path.write_bytes(original+b'\n')
    with pytest.raises(ValueError,match='evidence changed'):
        load_rhythm_fits(root,'one','area_px')
    path.write_bytes(original)
    source.write_bytes(source.read_bytes()+b'\n')
    assert load_rhythm_fits(root,'one','area_px')['rows']==[]


def test_fit_endpoint_reads_saved_curve_without_starting_jobs(model_run):
    root, _, _ = model_run
    manager=JobManager(runner=lambda *args,**kwargs: {})
    try:
        client=TestClient(create_app(jobs=manager))
        reply=client.get('/api/exploration/rhythm-fits',params={'run':str(root),'stem':'one','metric':'area_px'})
        assert reply.status_code==200 and reply.json()['rows'][0]['identity']==19
        assert manager.list()==[]
    finally:
        manager.close()
