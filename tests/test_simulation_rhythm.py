"""Detected-outline measurements and optional package test, without tracking."""
import numpy as np
import pytest
from scipy import ndimage as ndi
from pymicroglia import simulation_rhythm as simulation


def test_total_measurement_neighbour_exclusion_and_missing_outlines():
    labels=np.zeros((3,35,35),np.uint16);labels[:2,14:18,14:18]=1;labels[:2,14:18,23:27]=2
    raw=np.full((3,2,35,35),10,np.uint16);raw[:2,:,14:18,14:18]=30;raw[:2,:,14:18,23:27]=60000
    prepared=simulation.prepare_recording(raw,labels)
    assert np.all(prepared['values'][:2,:2]==320)
    assert np.isnan(prepared['values'][:,2]).all()
    assert prepared['entries'][0]['cell']==1 and prepared['entries'][1]['channel']==1
    assert prepared['candidates'][0]==dict(pattern='uniform',blur_pixels=2.5)


def test_optional_test_is_public_discoverable_and_keeps_control_gate():
    import pymicroglia as pm
    assert pm.test_simulation_rhythm is simulation.test_simulation_rhythm
    from pymicroglia import catalogue,registry,rhythm
    assert catalogue.action('test_simulation_rhythm')['method']=='simulation_rhythm.test_simulation_rhythm'
    assert registry.REGISTRY.resolve('test_simulation_rhythm') is simulation.test_simulation_rhythm
    assert 'test_simulation_rhythm' in registry.NEEDS_A_CLAIM
    with pytest.raises(rhythm.ControlMissing):
        pm.test_simulation_rhythm(np.zeros((30,35,35)),outlines=np.ones((30,35,35),np.uint16),interval_hours=.5)


def test_known_geometry_control_runs_complete_public_test_and_saves_results(tmp_path):
    # Exact-outline synthetic control with extra geometry information, not a
    # practical detector evaluation. Real-recording parity uses detected masks.
    t=np.arange(81)*.5;labels=np.zeros((len(t),50,50),np.uint16)
    raw=np.full((len(t),2,50,50),10.)
    for f in range(len(t)):
        labels[f,20:29,20:29+(f//7)%3]=1
        outline=labels[f]==1
        field=simulation._patterns(outline,['centre'])[0]
        for channel,period in enumerate([12.3,17.2]):
            raw[f,channel]+=ndi.gaussian_filter(field,2.5,mode='constant',truncate=4)*(1000+180*np.cos(2*np.pi*t[f]/period))
    labels[10:13]=0
    result=simulation.test_simulation_rhythm(raw,outlines=labels,times_h=t,control={'findings':['synthetic optical control']},
        selection='fixed',pattern='centre',blur_pixels=2.5,processing=['none','mean3'],options={'surrogates':9},output_dir=tmp_path)
    assert len(result['rows'])==4
    assert result['instrumental_control']['findings']==['synthetic optical control']
    for row in result['rows']:
        assert abs(row['period_hours']-[12.3,17.2][row['channel']])<.15
        assert np.isnan(row['prediction'][10:13]).all()
    assert (tmp_path/'simulation_rhythm_results.csv').exists()
    assert (tmp_path/'simulation_rhythm_traces.csv').exists()
    assert result['conditions']['pattern']=='centre'


def test_volume_and_nonmatching_clock_are_rejected():
    with pytest.raises(ValueError,match='single-plane'):
        simulation.prepare_recording(np.ones((12,2,3,35,35)),np.ones((12,35,35),np.uint16))
    with pytest.raises(ValueError,match='shapes must match'):
        simulation.test_simulation_rhythm(np.ones((12,35,35)),outlines=np.ones((12,35,35),np.uint16),times_h=np.arange(13),
             control={},selection='fixed',pattern='uniform',blur_pixels=2.5,options={'surrogates':9})
