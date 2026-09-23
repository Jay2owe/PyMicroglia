"""Every plotted scientific value is checked against the original complete CSV."""
import gzip
import hashlib
import importlib
from io import StringIO
import pandas as pd
import pytest
from pymicroglia.figure_tables.saved import Tables
from pymicroglia.visualisation.figures import load
from tests.test_motion_parity import FROZEN_RUN,FIXTURE,expected,_same_cell


@pytest.mark.parametrize('key,view,style_column',[
    ('cd68_reporter_rhythm','raster',None),('metric_rhythm_matrix','matrix',None),
    ('stable_traits_versus_states','ranking','condition_hue'),
    ('all_cell_trace_grid','traces',None),('negative_space','composition','colour')])
def test_complete_original_table(key,view,style_column):
    spec=load()[key];module,function=spec.prepare.split(':')
    options={o.name:o.default for o in spec.options}
    if key=='all_cell_trace_grid':
        # These are the recorded options of the original frozen figure.
        options.update(metrics=['corrected_mean'],multiple_testing='bh',detrend='robust_linear')
    data,_=getattr(importlib.import_module(module),function)(Tables(FROZEN_RUN,'parity_A1'),options)
    original=gzip.decompress((FIXTURE/'figures'/f'{spec.slug}.csv.gz').read_bytes())
    hashes=expected()['figures'][spec.slug]['plotted_sha256']
    assert hashlib.sha256(original).hexdigest()==hashes.get('data/der/figure_data.csv',hashes.get('figure_data.csv'))
    before=pd.read_csv(StringIO(original.decode()),dtype=str,keep_default_na=False)
    after=pd.read_csv(StringIO(data[view]['table'].to_csv(index=False)),dtype=str,keep_default_na=False)
    assert list(after)==list(before) and len(after)==len(before)
    differences=[(column,index) for column in before if column!=style_column
        for index,(a,b) in enumerate(zip(after[column],before[column])) if not _same_cell(column,a,b)]
    assert not differences


def test_repeatability_keeps_frozen_insufficient_cycles_refusal():
    from pymicroglia.figure_tables.cycle_repeatability import prepare
    spec=load()['second_verse']
    with pytest.raises(ValueError,match='No cells passed the primary rhythm test'):
        prepare(Tables(FROZEN_RUN,'parity_A1'),{o.name:o.default for o in spec.options})
