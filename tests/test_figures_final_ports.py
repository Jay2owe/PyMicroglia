"""Eligible paths and deliberate refusals for the last original figures."""
import hashlib
import importlib
import numpy as np
import pandas as pd
import pytest
from pymicroglia.visualisation.figures import load
from pymicroglia.figure_tables.saved import Tables
from tests.test_motion_parity import FROZEN_RUN,expected


def prepare(key,source=None,**overrides):
    spec=load()[key];module,name=spec.prepare.split(':')
    return getattr(importlib.import_module(module),name)(source or Tables(FROZEN_RUN,'parity_A1'),{**{o.name:o.default for o in spec.options},**overrides})[0]


def digest(table):
    return hashlib.sha256(table.to_csv(index=False,lineterminator='\r\n').encode()).hexdigest()


def test_declared_cycle_retains_original_table_and_rules():
    data=prepare('predictability_clock',period_hours=24.)
    hashes=expected()['figures']['predictability-clock']['plotted_sha256']
    assert digest(data['error']['table'])==hashes['data/der/figure_data.csv']
    assert digest(data.auxiliary['next_state_rule.csv'])==hashes['data/der/next_state_rule.csv']
    with pytest.raises(ValueError,match='explicitly'):prepare('predictability_clock')


def test_tissue_maps_retain_every_original_pixel():
    data=prepare('tissue_tectonics');hashes=expected()['figures']['tissue-tectonics']['plotted_sha256']
    for name,table in data.auxiliary.items():
        if name in {'rhythm_map_fits.csv','statistics.csv'}:
            import gzip
            from io import StringIO
            from tests.test_motion_parity import FIXTURE,_same_cell
            original=gzip.decompress((FIXTURE/('tissue_'+name+'.gz')).read_bytes())
            assert hashlib.sha256(original).hexdigest()==hashes['data/der/'+name]
            frozen=pd.read_csv(StringIO(original.decode()),dtype=str,keep_default_na=False)
            actual=table.fillna('').astype(str)
            assert actual.shape==frozen.shape and list(actual)==list(frozen)
            for column in actual:
                assert all(_same_cell(column,a,b) for a,b in zip(actual[column],frozen[column])),column
            continue
        assert digest(table)==hashes['data/der/'+name],name


@pytest.mark.parametrize('key',['own_clock_composite','independence_map'])
def test_unestablished_periods_are_not_compared(key):
    with pytest.raises(ValueError,match='supported'):prepare(key)


def test_missing_original_report_identity_is_clear():
    with pytest.raises(ValueError,match='identity 44'):prepare('cell_report_card')


@pytest.mark.parametrize('layout',['stack','overlay'])
def test_report_card_keeps_raw_values_and_requests_no_implicit_fit(layout):
    data=prepare('cell_report_card',identity=1,images=0,trace_layout=layout)
    original=Tables(FROZEN_RUN,'parity_A1').table('cell_frame').query('identity==1').sort_values('frame_index')
    table=data['traces']['table']
    assert table.fitted_value.isna().all()
    assert data.auxiliary['selected_fit_evidence.csv'].empty
    for metric,rows in table.groupby('metric',sort=False):
        np.testing.assert_array_equal(rows.raw_value,original[metric])
        if layout=='stack':np.testing.assert_array_equal(rows.plotted_value,original[metric])
        elif original[metric].std(ddof=0)>0:
            np.testing.assert_allclose(rows.plotted_value,(original[metric]-original[metric].mean())/original[metric].std(ddof=0),equal_nan=True)
        else:assert rows.plotted_value.eq(0).all() # the original display makes flat traces zero


class Contacts:
    interval=60.
    def table(self,name,optional=False):
        if name=='contacts.csv':return pd.DataFrame(dict(identity_a=[1,1,2,2],identity_b=[2,3,3,4],hours_in_contact=[1.,2.,3.,4.],dilation_px=1))
        if name=='cell_summary.csv':return pd.DataFrame(dict(identity=[1,2,3,4],area_px_median=[10.,20.,15.,60.]))
        return None


def test_contact_network_retains_pairs_and_two_test_families():
    data=prepare('contact_ledger',Contacts(),metrics=['area_px'],shuffles=100)
    assert data['pairs']['table'].pair.tolist()==['2-4','2-3','1-3','1-2']
    evidence=data.auxiliary['statistics.csv']
    assert len(evidence)==2
    assert evidence.pairs.eq(4).all()
    assert evidence.random_state.eq(24051986).all()


def test_constant_histograms_have_visible_nonzero_width():
    from pymicroglia.figure_tables.distributions import histogram
    for values,log in [([0,0,0],False),([1,1,1],False),([1,1,1],True)]:
        table=histogram(values,24,log=log)
        assert (table.bin_right>table.bin_left).all()
        assert table['count'].sum()==len(values)


def test_view_availability_reads_inputs_without_fitting(monkeypatch):
    from pymicroglia.visualisation.figures import available_views
    from pymicroglia import workbench
    monkeypatch.setattr(workbench,'estimate_one',lambda *a,**k:pytest.fail('availability must not fit'))
    assert available_views('cell_report_card',FROZEN_RUN,stem='parity_A1')==('tiles','traces')
    assert available_views('contact_ledger',FROZEN_RUN,stem='parity_A1')==()


class SupportedClocks(Tables):
    """Constructed eligibility evidence; these are not results about the fixture's cells."""
    def __init__(self):super().__init__(FROZEN_RUN,'parity_A1')
    def table(self,name,**kwargs):
        frame=super().table(name,**kwargs).copy()
        if name.removesuffix('.csv')=='rhythms':
            frame['best_period_hours']=12.
            frame['best_phase_hours']=frame.identity.astype(float)%12
            frame['rhythmic']=True
            frame['period_underdetermined']=False
        if name.removesuffix('.csv')=='coupling':
            for column in ['period_a_hours','period_b_hours','pair_phase_reference_period_hours']:
                frame[column]=12.
            frame['phase_difference_fraction']=frame.phase_difference/12.
        return frame


def test_supported_peak_composite_keeps_original_clock_values():
    source=SupportedClocks()
    data=prepare('own_clock_composite',source,metrics=['corrected_mean','turnover_index'],detrend='none')
    original=source.table('cell_frame').groupby('hours').corrected_mean.mean()
    np.testing.assert_array_equal(data['wall_clock']['table']['mean'],original)
    assert data['realigned']['period']==12.
    assert data['carried']['period']==12.
    assert data['realigned']['table'].metric.eq('corrected_mean').all()
    assert data['carried']['table'].metric.eq('turnover_index').all()


def test_supported_independence_display_preserves_saved_pairs():
    source=SupportedClocks()
    data=prepare('independence_map',source,metrics=['corrected_mean'])
    original=source.table('coupling').query("metric=='corrected_mean'")
    np.testing.assert_array_equal(data['distance']['table'].distance,original.distance)
    np.testing.assert_array_equal(data['distance']['table'].phase_difference,original.phase_difference)
    assert data['field']['period']==12.
    assert data['distance']['counts'].sum()==len(original)


class Daily:
    def table(self,name,**kwargs):
        return pd.DataFrame([dict(identity=i,metric=m,daily_profile_measures_enabled=True,
            rhythmic=True,period_underdetermined=False,best_period_hours=24.,
            relative_amplitude=.2*i,interdaily_stability=.3*i,intradaily_variability=.1*i,
            stability_underdetermined=False,days_covered=5.,observations=120,span_hours=119.,
            onset_found=True,onset_hour=(23+i)%24,offset_hour=(5+i)%24,
            active_duration_hours=6.,l5_onset_hour=10.+i,m10_onset_hour=i)
            for m in ['signal','area'] for i in [1,2,3]])


def test_explicit_supported_daily_profiles_keep_denominators_and_intervals():
    source=Daily()
    strength=prepare('rhythm_strength',source)
    assert strength['ranking']['table'].cells.tolist()==[3,3]
    active=prepare('active_span',source)
    assert len(active['spans']['table'])==6
    assert active.auxiliary['onset_agreement.csv'].onset_spread_hours.eq(0).all()
    assert active['agreement']['table']['count'].sum()==3


def test_report_card_records_explicit_estimator_without_implicit_cosine():
    data=prepare('cell_report_card',identity=1,images=0,metrics=['corrected_mean'],
                 fit='corrected_mean',fit_method='lomb',significance_method='lomb',
                 detrend='none')
    evidence=data.auxiliary['selected_fit_evidence.csv']
    assert len(evidence)==1
    assert evidence.iloc[0]['method']=='lomb'
    assert data['traces']['table'].fitted_value.isna().all()


@pytest.mark.parametrize('unknown',[None,'unknown','',float('nan')])
def test_common_period_requires_explicit_sufficiency(unknown):
    from pymicroglia.figure_tables.rhythm_eligibility import common_period
    rows=pd.DataFrame(dict(best_period_hours=[12.],rhythmic=[True],period_underdetermined=[unknown]))
    with pytest.raises(ValueError,match='supported significant periods'):common_period(rows)


@pytest.mark.parametrize('method,descriptive',[('fft_nlls',False),('lomb',True)])
def test_explicit_report_card_curves_retain_native_or_requested_fit(method,descriptive):
    data=prepare('cell_report_card',identity=1,images=0,metrics=['corrected_mean'],
                 fit='corrected_mean',fit_method=method,significance_method='lomb',
                 descriptive_cosinor=descriptive,detrend='none')
    assert data['traces']['table'].fitted_value.notna().any()
    assert data.auxiliary['selected_fit_evidence.csv'].iloc[0]['method']==method
