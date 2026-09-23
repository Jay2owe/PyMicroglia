"""Saved rhythm evidence and footprint accounting preserve their original values."""
import hashlib
import pytest
from pymicroglia.figure_tables import exchange
from pymicroglia.figure_tables.saved import Tables
from pymicroglia.visualisation.figures import load
from tests.test_motion_parity import expected,FROZEN_RUN


@pytest.mark.parametrize('key,prepare,view',[
    ('rhythmicity_above_noise',exchange.noise_floor,'floor'),
    ('conservation_ledger',exchange.conservation,'ledger'),('breath_trace',exchange.breath,'breath')])
def test_prepared_values_match_frozen(key,prepare,view):
    spec=load()[key]
    prepared,_=prepare(Tables(FROZEN_RUN,'parity_A1'),{o.name:o.default for o in spec.options})
    encoded=prepared[view]['table'].to_csv(index=False,lineterminator='\r\n').encode()
    assert hashlib.sha256(encoded).hexdigest()==expected()['figures'][spec.slug]['plotted_sha256']['data/der/figure_data.csv']


def test_exchange_view_has_its_own_table(tmp_path):
    import pandas as pd
    from pymicroglia.figure_tables.actions import draw
    draw('conservation_ledger',FROZEN_RUN,stem='parity_A1',view='against_size',output_dir=tmp_path)
    assert set(pd.read_csv(tmp_path/'parity_A1_against_size.csv').view)=={'against_size'}


def test_explicit_wording_overrides_recorded_default(tmp_path):
    from pymicroglia.figure_tables.actions import draw
    from reprofig import extract_records
    draw('breath_trace',FROZEN_RUN,stem='parity_A1',view='breath',output_dir=tmp_path,
         text={'title':'Controlled software example'})
    settings=extract_records(tmp_path/'parity_A1_breath.svg')[0].analysis['settings']
    assert settings['text']['title']=='Controlled software example'
    assert settings['text_sources']['title']=='call'
