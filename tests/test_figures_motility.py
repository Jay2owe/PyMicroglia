"""Exact prepared values match the frozen Motion figure tables."""
import hashlib
import pytest
from pymicroglia.figure_tables.saved import Tables
from pymicroglia.figure_tables import motility
from tests.test_motion_parity import FROZEN_RUN,expected


@pytest.mark.parametrize('key,prepare,options,view,table',[
    ('step-size-distribution',motility.step_size,{'metrics':'step_px_gapless','bins':45},'distribution','figure_data'),
    ('displacement-curves',motility.displacement,{'metrics':'msd_alpha','bins':18,'max_lag':None},'curves','figure_data'),
    ('displacement-curves',motility.displacement,{'metrics':'msd_alpha','bins':18,'max_lag':None},'alpha','cell_distribution'),
])
def test_prepared_view_matches_frozen_plotted_bytes(key,prepare,options,view,table):
    data,_=prepare(Tables(FROZEN_RUN,'parity_A1'),options)
    written=data[view]['table'].to_csv(index=False,lineterminator='\r\n').encode('utf-8')
    assert hashlib.sha256(written).hexdigest()==expected()['figures'][key]['plotted_sha256'][f'data/der/{table}.csv']


def test_displacement_views_save_independently(tmp_path):
    from pymicroglia.figure_tables.actions import draw
    draw('displacement_curves',FROZEN_RUN,stem='parity_A1',view='alpha',output_dir=tmp_path)
    assert (tmp_path/'parity_A1_alpha.svg').is_file()
    assert (tmp_path/'parity_A1_alpha.csv').is_file()
