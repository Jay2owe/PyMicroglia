"""The port retains original numerical values and independently drawable views."""
import hashlib
import importlib
import pandas as pd
import pytest
from pymicroglia.figure_tables.saved import Tables
from pymicroglia.visualisation.figures import load
from tests.test_motion_parity import FROZEN_RUN,expected

CASES=[('regime_ribbon','ribbon'),('regime_transitions','matrix'),('contrast_forest','effects'),
       ('cell_lifecycle_summary',None),('phase_compass','profile'),('sholl_kymograph','kymograph'),('regime_programmes','similarity'),('patch_ledger','coverage'),('regime_reporter_bridge','paired'),('breakout_triggered_average','triggered'),('upheaval_events','raster'),('radial_occupancy_rhythms','matrix'),('recurrence_wall','rate'),('pixel_fate_flow','flow'),('identity_trajectories','map'),('cells_on_screen','counts')]

@pytest.mark.parametrize('key,view',CASES)
def test_original_plotted_data(key,view):
    spec=load()[key];module,name=spec.prepare.split(':')
    data,_=getattr(importlib.import_module(module),name)(Tables(FROZEN_RUN,'parity_A1'),{o.name:o.default for o in spec.options})
    table=data[view]['table'] if view else pd.concat([d['table'].assign(panel=k) for k,d in data.items() if not d['table'].empty],ignore_index=True,sort=False)
    actual=hashlib.sha256(table.to_csv(index=False,lineterminator='\r\n').encode()).hexdigest()
    assert actual==expected()['figures'][spec.slug]['plotted_sha256']['data/der/figure_data.csv']


def test_lifecycle_empty_nonboundary_events_are_drawable(tmp_path):
    # The frozen source failed with NaN axis limits when only censored events
    # were present. Empty evidence now remains an explicitly empty view.
    from pymicroglia.figure_tables.actions import draw
    draw('cell_lifecycle_events',FROZEN_RUN,stem='parity_A1',view='ledger',output_dir=tmp_path)
    assert (tmp_path/'parity_A1_ledger.svg').is_file()


def test_regime_shuffles_validate_before_allocation():
    from pymicroglia.figure_tables.regimes import transitions
    with pytest.raises(ValueError,match='non-negative integer'):
        transitions(Tables(FROZEN_RUN,'parity_A1'),dict(shuffles=-1,normalise='row'))


def test_breath_negative_cell_count_is_rejected():
    from pymicroglia.figure_tables.exchange import breath
    with pytest.raises(ValueError,match='positive'):
        breath(Tables(FROZEN_RUN,'parity_A1'),dict(cells=-2))


@pytest.mark.parametrize('key', ['patch_ledger','regime_reporter_bridge','breakout_triggered_average','upheaval_events','pixel_fate_flow'])
def test_original_supporting_tables(key):
    spec=load()[key];module,name=spec.prepare.split(':')
    data,_=getattr(importlib.import_module(module),name)(Tables(FROZEN_RUN,'parity_A1'),{o.name:o.default for o in spec.options})
    hashes=expected()['figures'][spec.slug]['plotted_sha256']
    matched=0
    for name,table in data.auxiliary.items():
        original=hashes.get('data/der/'+name)
        if original:
            matched+=1
            assert hashlib.sha256(table.to_csv(index=False,lineterminator='\r\n').encode()).hexdigest()==original,name
    assert matched
