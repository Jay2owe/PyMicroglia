"""Compare the package acceleration to the actual bundled native detector."""
from contextlib import ExitStack
import json
import linecache
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from pymicroglia.tracking.detector_acceleration import install


@pytest.fixture
def detector():
    engine=Path(__file__).parents[1]/'src/pymicroglia/tracking/data/motion_engine'
    cached=[]
    def load(name):
        source=(engine/f'code/{name}.py').read_bytes().decode()
        filename=f'<bundled_detector_test>/{name}.py'
        linecache.cache[filename]=(len(source),None,source.splitlines(True),filename)
        cached.append(filename)
        module=ModuleType(f'_test_{name}')
        module.__file__=filename
        exec(compile(source,filename,'exec'),vars(module))
        return module
    module=load('raw_physical_hypotheses')
    try:
        with ExitStack() as stack:
            stack.enter_context(load('peak_search_speed').bucketed_peak_search(module))
            stack.enter_context(load('relative_detection_visibility').relative_detection_visibility(
                module,reuse_core_structure=True))
            stack.enter_context(load('fractional_raw_peaks').fractional_raw_peaks(module))
            yield module
    finally:
        for name in cached:linecache.cache.pop(name,None)


@pytest.mark.parametrize('promotion',['weak','legacy'])
@pytest.mark.parametrize('shape,half',[((1,1),0),((2,7),7),((35,47),7),((79,91),30)])
@pytest.mark.parametrize('kind',['empty','impulse','integer','fractional','tiny_signal'])
def test_matches_bundled_detector(detector,promotion,shape,half,kind):
    setter=getattr(np,'_set_promotion_state',None)
    old=np._get_promotion_state() if setter else None
    if setter:setter(promotion)
    try:
        frame=np.random.default_rng(89).integers(0,600,size=shape,dtype=np.uint16)
        if kind=='empty':frame.fill(0)
        if kind=='impulse':
            frame.fill(0);frame.flat[0]=500
        if kind=='fractional':frame=frame.astype(np.float32)/7.0
        if kind=='tiny_signal':frame=frame.astype(np.float32)*1e-12
        params={'core_window_radius_px':half}
        original=detector.detect_frame
        expected=original(frame,7,params)
        with install(detector):actual=detector.detect_frame(frame,7,params)
        assert detector.detect_frame is original
        assert json.dumps(actual,sort_keys=True)==json.dumps(expected,sort_keys=True)
    finally:
        if setter:setter(old)


def test_empty_signal_retains_early_return(detector):
    frame=np.zeros((9,13),dtype=np.uint16)
    invalid={'minimum_peak_distance_px':float('nan'),'weak_peak_quantile':4}
    expected=detector.detect_frame(frame,0,invalid)
    with install(detector):assert detector.detect_frame(frame,0,invalid)==expected


@pytest.mark.parametrize('scale',[0.5,2.0])
def test_calibrated_settings_are_preserved(detector,scale):
    frame=np.random.default_rng(91).integers(0,600,size=(83,101),dtype=np.uint16)
    params=dict(core_sigma_px=1.2*scale,background_sigma_px=8.0*scale,
                minimum_peak_distance_px=5.0*scale,core_window_radius_px=int(7*scale),
                minimum_core_radius_px=2.0*scale,maximum_core_radius_px=7.0*scale,
                weak_peak_quantile=0.4,strong_peak_quantile=0.8)
    expected=detector.detect_frame(frame,3,params)
    with install(detector):actual=detector.detect_frame(frame,3,params)
    assert json.dumps(actual,sort_keys=True)==json.dumps(expected,sort_keys=True)


def test_window_larger_than_image_needs_no_unbounded_padding(detector):
    frame=np.random.default_rng(71).integers(0,500,size=(11,17),dtype=np.uint16)
    params={'core_window_radius_px':10000}
    expected=detector.detect_frame(frame,0,params)
    with install(detector):actual=detector.detect_frame(frame,0,params)
    assert json.dumps(actual,sort_keys=True)==json.dumps(expected,sort_keys=True)


def test_restores_original_after_error(detector):
    original=detector.detect_frame
    with pytest.raises(RuntimeError):
        with install(detector):raise RuntimeError('interrupted caller')
    assert detector.detect_frame is original


def test_engine_installs_and_restores_acceleration(tmp_path,monkeypatch):
    import sys
    from pymicroglia.tracking import engine
    root=engine._engine(tmp_path)
    config=tmp_path/'recording.json'
    config.write_bytes((Path(engine.__file__).parent/'data/motion_engine/config.json').read_bytes())
    called,seen=[],[]
    def native(*args,**kwargs):
        module=sys.modules['raw_physical_hypotheses']
        seen.append((module,module.detect_frame))
        called.append('native')
    def corrections(*args,**kwargs):called.append('complete corrections')
    monkeypatch.setattr(engine,'_execute_pipeline',native)
    monkeypatch.setattr(engine,'_execute_corrections',corrections)
    engine._run_frozen(root,'sample','run',config,tmp_path/'run.log')
    assert called==['native','complete corrections']
    (module,during),=seen
    # The accelerated detector ran inside, and the run left the native function in place.
    assert Path(during.__code__.co_filename)==Path(install.__wrapped__.__code__.co_filename)
    assert Path(module.detect_frame.__code__.co_filename)!=Path(install.__wrapped__.__code__.co_filename)
    recipe=json.loads((root/'detector_acceleration.json').read_text())
    assert recipe['backend']=='cpu'
    assert len(recipe['source_sha256'])==64


def test_checkpoint_rejects_changed_detector_implementation(tmp_path,monkeypatch):
    import sys
    from types import SimpleNamespace
    from pymicroglia.tracking import checkpoints
    class Pool:pass
    monkeypatch.setitem(sys.modules,'auto_organotypic.array_storage',SimpleNamespace(Pool=Pool))
    config=tmp_path/'config.json'
    config.write_text('{}')
    pipeline=SimpleNamespace(ROOT=tmp_path,_verify_pins=lambda *args:None)
    cfg=SimpleNamespace(path=config)
    saved=checkpoints.Stages(pipeline,cfg,'sample','run',lambda path:None)
    assert 'detector_acceleration.py' in saved.signature['adapter']
    original=checkpoints.sha
    monkeypatch.setattr(checkpoints,'sha',lambda path:
        'changed' if Path(path).name=='detector_acceleration.py' else original(path))
    # A changed implementation recomputes every stage and keeps the earlier state beside it.
    second=checkpoints.Stages(pipeline,cfg,'sample','run',lambda path:None)
    assert second.fresh and second.signature!=saved.signature
    assert len(list(second.state_path.parent.glob('tracking_resume_superseded_*.json')))==1
