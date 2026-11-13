"""Every plotted scientific value is checked against the original complete CSV."""
import gzip
import hashlib
import importlib
from io import StringIO
import pandas as pd
import pytest
from pymicroglia.figure_tables.saved import Tables
from pymicroglia.visualisation.figures import load
from tests.test_motion_parity import FROZEN_RUN,FIXTURE,expected,_same_cell,workbench_baseline as load_baseline
workbench_baseline=load_baseline()


def _original(spec):
    original=gzip.decompress((FIXTURE/'figures'/f'{spec.slug}.csv.gz').read_bytes())
    hashes=expected()['figures'][spec.slug]['plotted_sha256']
    assert hashlib.sha256(original).hexdigest()==hashes.get('data/der/figure_data.csv',hashes.get('figure_data.csv'))
    return pd.read_csv(StringIO(original.decode()),dtype=str,keep_default_na=False)


@pytest.mark.parametrize('key,view,style_column',[
    ('stable_traits_versus_states','ranking','condition_hue'),('negative_space','composition','colour')])
def test_complete_original_table(key,view,style_column):
    spec=load()[key];module,function=spec.prepare.split(':')
    options={o.name:o.default for o in spec.options}
    if key=='stable_traits_versus_states':
        options['metrics']=['area_px','corrected_mean','skeleton_branches',
                            'solidity','ramification_index','turnover_index']
    data,_=getattr(importlib.import_module(module),function)(Tables(FROZEN_RUN,'parity_A1'),options)
    before=_original(spec)
    after=pd.read_csv(StringIO(data[view]['table'].to_csv(index=False)),dtype=str,keep_default_na=False)
    assert list(after)==list(before) and len(after)==len(before)
    differences=[(column,index) for column in before if column!=style_column
        for index,(a,b) in enumerate(zip(after[column],before[column])) if not _same_cell(column,a,b)]
    assert not differences


@pytest.mark.parametrize('key',['cd68_reporter_rhythm','metric_rhythm_matrix','all_cell_trace_grid'])
def test_fitted_table_matches_motion_and_the_workbench_baseline(key):
    # Motion's settings and recorded options; the values a Workbench fit or
    # detrend writes moved after 0.8.2 and meet the re-recorded baseline.
    table=workbench_baseline.figure_table(key)
    assert not workbench_baseline.differences(f'figure_{key}.csv',table)
    before=_original(load()[key])
    # The current trace grid also records the value after temporal filtering.
    assert not workbench_baseline.motion_differences(table[before.columns],before,workbench_baseline.moved_columns(key))


def test_repeatability_keeps_frozen_insufficient_cycles_refusal():
    from pymicroglia.figure_tables.cycle_repeatability import prepare
    spec=load()['second_verse']
    options={o.name:o.default for o in spec.options}
    options['metrics']='corrected_mean'
    with pytest.raises(ValueError,match='No cells passed the primary rhythm test'):
        prepare(Tables(FROZEN_RUN,'parity_A1'),options)
