"""Reject omission of original groups or substitution of partial timings."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('tracking_stage_comparison',
    Path(__file__).resolve().parents[1] / 'render_tracking_stage_comparison.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def scope():
    return {'groups': [{'id': 'tracking', 'name': 'Earlier tracking', 'before_seconds': 11520},
                       {'id': 'history', 'name': 'Remaining identity checks', 'before_seconds': 22980}]}


def unmeasured(key):
    return {'id': key, 'after_status': 'unmeasured', 'complete_group': False}


def test_missing_original_group_rejected(tmp_path):
    with pytest.raises(ValueError, match='missing='):
        module.render(scope(), {'groups': [unmeasured('tracking')]}, tmp_path)


def test_unknown_totals_are_visible_and_not_estimates(tmp_path):
    result = module.render(scope(), {'groups': [unmeasured('tracking'), unmeasured('history')]}, tmp_path)
    assert 'Earlier tracking' in result and 'Remaining identity checks' in result
    assert result.count('Not yet measured') == 2


def test_constituent_cannot_replace_group(tmp_path):
    row = {'id': 'history', 'after_status': 'measured', 'complete_group': False,
           'after_seconds': 10, 'receipt': 'part.json'}
    (tmp_path / 'part.json').write_text('{}')
    with pytest.raises(ValueError, match='coverage'):
        module.render(scope(), {'groups': [unmeasured('tracking'), row]}, tmp_path)


def test_complete_measurement_requires_receipt(tmp_path):
    row = {'id': 'history', 'after_status': 'measured', 'complete_group': True,
           'after_seconds': 10, 'receipt': 'part.json'}
    current = {'groups': [unmeasured('tracking'), row]}
    with pytest.raises(ValueError, match='Missing measurement'):
        module.render(scope(), current, tmp_path)
    (tmp_path / 'part.json').write_text('{}')
    assert '[0 min 10 s](part.json)' in module.render(scope(), current, tmp_path)


def test_duplicate_group_rejected(tmp_path):
    with pytest.raises(ValueError, match='Duplicate'):
        module.render(scope(), {'groups': [unmeasured('tracking'), unmeasured('tracking')]}, tmp_path)


def test_completed_original_timing_replaces_partial_snapshot_with_receipt(tmp_path):
    row=unmeasured('history')
    row.update(before_complete_seconds=10734.513586199959,before_receipt='before.json')
    current={'groups':[unmeasured('tracking'),row]}
    with pytest.raises(ValueError,match='before receipt'):
        module.render(scope(),current,tmp_path)
    (tmp_path/'before.json').write_text(json.dumps({'seconds':row['before_complete_seconds']}))
    assert '[2 h 58 min](before.json)' in module.render(scope(),current,tmp_path)
    row['before_complete_seconds']=1
    with pytest.raises(ValueError,match='differs from receipt'):
        module.render(scope(),current,tmp_path)
