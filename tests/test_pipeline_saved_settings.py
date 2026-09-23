"""Saved declarations cross the public action boundary without inventing windows."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
from pymicroglia.pipelines import _requests
from tests.test_intervention_options import declaration, fixture


def execute(tmp_path, monkeypatch, settings, request=None):
    from pymicroglia.pipelines.intervention import options
    _, tables, paths, _ = fixture(tmp_path)
    manifest = {'settings': settings}
    monkeypatch.setattr(_requests, 'inputs', lambda run: (manifest, 'saved', paths, {n: __import__('hashlib').sha256(p.read_bytes()).hexdigest() for n,p in paths.items()}, tables, []))
    monkeypatch.setattr(options, 'run_request', lambda resolved, *a, **k: resolved)
    return _requests.execute('intervention_response', tmp_path,
        request or declaration(), claim='Test inheritance of the declared experimental design')


def test_saved_metric_group_is_resolved_in_its_recorded_order(tmp_path, monkeypatch):
    result = execute(tmp_path, monkeypatch, {'metric_groups': {'signals': ['custom_signal']}},
                     declaration(measurements=['@signals']))
    assert [m.column for m in result.measurements] == ['custom_signal']


def test_default_windows_and_recording_override_survive_public_call(tmp_path, monkeypatch):
    windows = declaration()['windows']
    override = deepcopy(windows); override[0]['start'] = -1.
    result = execute(tmp_path, monkeypatch, {'windows': windows, 'recording_windows': {'treated': override}},
                     declaration(windows=[]))
    assert result.recordings['treated']['windows'][0]['start_hours'] == 51.
    assert result.recordings['control']['windows'][0]['start_hours'] == 60.


def test_conflicting_explicit_and_inherited_recording_windows_are_refused(tmp_path, monkeypatch):
    windows = declaration()['windows']
    override = deepcopy(windows); override[0]['start'] = -1.
    with pytest.raises(ValueError, match='Conflicting explicit'):
        execute(tmp_path, monkeypatch, {'recording_windows': {'treated': windows}},
                declaration(recording_windows={'treated': override}))


def test_explicit_shared_windows_override_saved_defaults(tmp_path, monkeypatch):
    windows = deepcopy(declaration()['windows']); windows[0]['start'] = -1.
    result = execute(tmp_path, monkeypatch, {'windows': windows})
    assert result.recordings['treated']['windows'][0]['start_hours'] == 50.


def test_configuration_records_original_per_movie_windows(tmp_path):
    import json
    from pymicroglia.measure.spec import MeasureConfig
    path = tmp_path / 'config.json'
    declared = [{'name': 'early', 'from_frame': 0, 'to_frame': 3}]
    path.write_text(json.dumps({'frame_interval_min': 30, 'movies': [
        {'stem': 'one', 'labels': 'labels.tif', 'raw': 'raw.tif', 'windows': declared}]}))
    assert MeasureConfig.load(path).settings()['recording_windows'] == {'one': declared}
