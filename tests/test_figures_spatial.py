"""Prepared maps retain the frozen values and the complete interpretation."""
import gzip
import hashlib
from io import StringIO
import pandas as pd
import pytest
from pymicroglia.figure_tables.saved import Tables
from pymicroglia.figure_tables.spatial import prepare
from pymicroglia.visualisation.figures import load
from tests.test_motion_parity import FROZEN_RUN,FIXTURE,expected,_same_cell


@pytest.mark.parametrize('key,kind',[
    ('tissue_expansion_sequence','expansion'),('spatial_rhythm_progression','progression'),
    ('tissue_coverage_gaps','gaps'),('reporter_shape_timing','timing'),
    ('spatial_rhythm_maps','period')])
def test_prepared_spatial_values_match_frozen(key,kind):
    spec=load()[key]
    table,*_=prepare(Tables(FROZEN_RUN,'parity_A1'),{o.name:o.default for o in spec.options},kind)
    frozen=expected()['figures'][spec.slug]['plotted_sha256']['figure_data.csv']
    if kind=='period':
        original=gzip.decompress((FIXTURE/'figures/spatial-rhythm-maps.csv.gz').read_bytes())
        assert hashlib.sha256(original).hexdigest()==frozen
        before=pd.read_csv(StringIO(original.decode()),dtype=str,keep_default_na=False)
        after=pd.read_csv(StringIO(table.to_csv(index=False)),dtype=str,keep_default_na=False)
        assert list(after)==list(before) and len(after)==len(before)
        differences=[(column,index) for column in before for index,(a,b) in
                     enumerate(zip(after[column],before[column])) if not _same_cell(column,a,b)]
        assert not differences
    else:
        assert hashlib.sha256(table.to_csv(index=False,lineterminator='\r\n').encode()).hexdigest()==frozen


def test_neighbours_preserves_frozen_refusal():
    spec=load()['neighbour_coordination']
    with pytest.raises(ValueError,match='too few sufficiently observed cells'):
        prepare(Tables(FROZEN_RUN,'parity_A1'),{o.name:o.default for o in spec.options},'neighbours')


def test_period_phase_can_be_drawn_alone_with_qualification(tmp_path):
    from pymicroglia.figure_tables.actions import draw
    from reprofig import extract_records
    draw('spatial_rhythm_maps',FROZEN_RUN,stem='parity_A1',view='phase',output_dir=tmp_path)
    path=tmp_path/'parity_A1_phase.svg'
    record=extract_records(path)[0]
    settings=record.analysis['settings']
    assert 'does not establish comparable timing' in settings['text']['footnote']
    assert settings['text_sources']['footnote']=='default'
    assert set(pd.read_csv(tmp_path/'parity_A1_phase.csv')['view'])=={'phase'}


def test_metadata_exclusion_does_not_hide_changed_analysis_settings():
    import json
    a={'cwd':'first','inputs':{'alpha':.05}}
    b={'cwd':'second','inputs':{'alpha':.01}}
    assert not _same_cell('workbench_run_record_json',json.dumps(a),json.dumps(b))
