"""Prepared maps retain the frozen values and the complete interpretation."""
import gzip
import hashlib
from io import StringIO
import pandas as pd
import pytest
from pymicroglia.figure_tables.saved import Tables
from pymicroglia.figure_tables.spatial import prepare
from pymicroglia.visualisation.figures import load
from tests.test_motion_parity import FROZEN_RUN,FIXTURE,expected,_same_cell,workbench_baseline as load_baseline
workbench_baseline=load_baseline()


def test_unfitted_spatial_values_match_frozen():
    spec=load()['tissue_coverage_gaps']
    table,*_=prepare(Tables(FROZEN_RUN,'parity_A1'),{o.name:o.default for o in spec.options},'gaps')
    frozen=expected()['figures'][spec.slug]['plotted_sha256']['figure_data.csv']
    assert hashlib.sha256(table.to_csv(index=False,lineterminator='\r\n').encode()).hexdigest()==frozen


@pytest.mark.parametrize('key',sorted(workbench_baseline.SPATIAL))
def test_fitted_spatial_values_match_the_workbench_baseline(key):
    # Motion's settings; the values pass through a Workbench fit or detrend,
    # which moved after 0.8.2, so they meet the re-recorded baseline.
    table=workbench_baseline.spatial_table(key)
    assert not workbench_baseline.differences(f'{key}.csv',table)
    if key=='spatial_rhythm_maps':
        original=gzip.decompress((FIXTURE/'figures/spatial-rhythm-maps.csv.gz').read_bytes())
        assert hashlib.sha256(original).hexdigest()==expected()['figures']['spatial-rhythm-maps']['plotted_sha256']['figure_data.csv']
        before=pd.read_csv(StringIO(original.decode()),dtype=str,keep_default_na=False)
        assert not workbench_baseline.motion_differences(table,before)


def test_neighbours_preserves_frozen_refusal():
    spec=load()['neighbour_coordination']
    with pytest.raises(ValueError,match='too few sufficiently observed cells'):
        prepare(Tables(FROZEN_RUN,'parity_A1'),{o.name:o.default for o in spec.options},'neighbours')


def test_workbench_baseline_tolerates_machines_but_not_changed_settings():
    import json
    same=workbench_baseline._same
    record={'inputs':{'alpha':.05},'cwd':'here','result_sha256':'a','environment':{'execution':{'processor':'Intel'}}}
    moved={'inputs':{'alpha':.05,'fold_period_source':'manual'},'cwd':'there','result_sha256':'b','environment':{'execution':{'processor':'AMD'}}}
    assert same('workbench_run_record_json',json.dumps(moved),json.dumps(record))
    assert not same('workbench_run_record_json',json.dumps({**moved,'inputs':{'alpha':.01}}),json.dumps(record))
    assert not same('workbench_run_record_json',json.dumps({'cwd':'here'}),json.dumps(record))
    assert same('estimate_diagnostics',repr({'power':[0.15020407803561397]}),repr({'power':[0.15020407803561453]}))
    assert not same('estimate_diagnostics',repr({'power':[0.1502041]}),repr({'power':[0.1502040]}))


def test_metadata_exclusion_does_not_hide_changed_analysis_settings():
    import json
    a={'cwd':'first','inputs':{'alpha':.05}}
    b={'cwd':'second','inputs':{'alpha':.01}}
    assert not _same_cell('workbench_run_record_json',json.dumps(a),json.dumps(b))
