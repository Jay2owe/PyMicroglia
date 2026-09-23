"""Legacy arguments retain the complete saved design through the public action."""
import importlib.util
import json
from pathlib import Path
import pytest


@pytest.fixture
def shim():
    root=Path(__file__).resolve().parents[1]
    path=root/'development/motion-shim/__main__.py'
    if not path.is_file(): path=root.parent/'Motion/analysis/__main__.py'
    if not path.is_file(): pytest.skip('Motion compatibility checkout is not installed')
    spec=importlib.util.spec_from_file_location('motion_command_shim',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config(tmp_path):
    value=dict(dataset='Saved design',frame_interval_min=30,verify_hashes=False,
               movies=[dict(stem='cell',labels='labels.tif',raw='raw.tif')],
               calibration_tiffs=['calibration.tif'],
               plots=[dict(name='counts',figure='cells-on-screen')],
               theme={'preset':'house'})
    path=tmp_path/'config.json'
    path.write_text(json.dumps(value),encoding='utf-8')
    return path


def test_measure_config_preserves_calibration_plans_and_source(tmp_path,monkeypatch):
    from pymicroglia.measure import measure
    import importlib
    engine=importlib.import_module('pymicroglia.measure.run')
    path=config(tmp_path)
    monkeypatch.setattr(engine,'run',lambda cfg,*a,**kw:cfg)
    actual=measure(analysis_config=path,output_dir=tmp_path/'out')
    assert actual.dataset=='Saved design'
    assert actual.calibration_tiffs==[tmp_path/'calibration.tif']
    assert actual.plots==[dict(name='counts',figure='cells-on-screen')]
    assert actual.source_path==path
    assert actual.verify_hashes is False
    assert actual.movies[0].labels==tmp_path/'labels.tif'


def test_legacy_run_uses_exact_destination_and_complete_configuration(shim,tmp_path,monkeypatch):
    path=config(tmp_path);calls=[]
    monkeypatch.setattr(shim,'invoke',lambda *a,**kw:calls.append((a,kw)))
    assert shim.main(['run','--config',str(path),'--out',str(tmp_path/'output')])==0
    action,params,claim=calls[0][0]
    assert action=='measure' and params['analysis_config']==str(path)
    assert params['output_dir']==str(tmp_path) and params['run_label']=='output'
    assert params['if_exists']=='error' and claim


def test_follow_translation_keeps_identities_and_event_window(shim,monkeypatch):
    calls=[]
    monkeypatch.setattr(shim,'invoke',lambda *a,**kw:calls.append((a,kw)))
    shim.main(['videos','run','--identity','7','--span','event','--event-hours','4','--no-locator'])
    action,params,_=calls[0][0]
    assert action=='follow' and params['identities']==[7]
    assert params['span']=='event' and params['event_hours']==4 and not params['locator']


def test_invoke_uses_the_public_recorded_call_contract(shim, monkeypatch):
    import pymicroglia.run as runner
    calls = []
    monkeypatch.setattr(runner, "run_recorded",
                        lambda action, **params: calls.append((action, params)) or
                        {"result": {"run": "saved"}, "record": {}, "recorded": True})
    result = shim.invoke("measure", {"analysis_config": "config.json"}, "Question")
    assert calls == [("measure", {"analysis_config": "config.json",
                                  "claim": "Question", "entry": "cli"})]
    assert result["result"] == {"run": "saved"}


def test_expanded_plot_items_keep_separate_output_folders(shim, tmp_path, monkeypatch):
    document = {"settings": {"plots": [{"figure": "clock-face",
                  "for_each": {"view": ["dial", "rose"]}}]}}
    monkeypatch.setattr(shim, "read", lambda path: document)
    calls = []
    monkeypatch.setattr(shim, "invoke", lambda *args, **kwargs: calls.append(args))
    shim.main(["plots", str(tmp_path), "--dry-run"])
    assert len(calls) == 2
    assert calls[0][1]["view"] == "dial" and calls[1][1]["view"] == "rose"
    assert calls[0][1]["output_dir"] != calls[1][1]["output_dir"]



def test_failed_workflow_is_not_reported_as_success(shim, monkeypatch):
    import pymicroglia.run as runner
    from types import SimpleNamespace
    failure=SimpleNamespace(successful=False,results={"analysis":SimpleNamespace(
        outcome=SimpleNamespace(status="failed",reason="Declared input unavailable"))})
    monkeypatch.setattr(runner,"run_recorded",lambda *a,**k:{"result":failure})
    with pytest.raises(RuntimeError,match="Declared input unavailable"):
        shim.invoke("rhythm_discovery",{}, "Synthetic command check")


def test_pipeline_keeps_saved_science_reusable(shim, monkeypatch):
    calls=[]
    monkeypatch.setattr(shim,"read",lambda path:{"pipeline":"rhythm-discovery","name":"screen"})
    monkeypatch.setattr(shim,"invoke",lambda *a,**k:calls.append(a))
    shim.main(["pipeline","run","--request","request.json"])
    assert calls[0][1]["if_exists"]=="skip"


def test_conditions_try_checks_new_stems(shim, tmp_path, capsys):
    path=config(tmp_path)
    data=json.loads(path.read_text())
    data["conditions"]={"control":"_A", "treated":"_B"}
    data["movies"][0]["stem"]="movie_A"
    path.write_text(json.dumps(data),encoding="utf-8")
    assert shim.main(["conditions","--config",str(path),"--try","new_B"])==0
    rows=json.loads(capsys.readouterr().out)
    assert rows[-1]["stem"]=="new_B" and rows[-1]["condition"]=="treated"



@pytest.mark.parametrize("action",["states","cluster"])
def test_learning_command_keeps_exact_destination(shim, tmp_path, monkeypatch, action):
    calls=[]
    monkeypatch.setattr(shim,"invoke",lambda *a,**k:calls.append(a))
    destination=tmp_path/"requested"
    shim.main([action,"run","--out",str(destination)])
    assert Path(calls[0][1]["output_dir"])/calls[0][1]["run_label"]==destination


def test_legacy_ring_option_is_translated_before_plan_validation(shim, tmp_path, monkeypatch):
    document = {"settings":{"plots":[{"figure":"sholl-kymograph",
                 "for_each":{"rings":[2,4]},"as":"rings-{rings}"}]}}
    monkeypatch.setattr(shim,"read",lambda path:document)
    calls=[]
    monkeypatch.setattr(shim,"invoke",lambda *a,**k:calls.append(a))
    shim.main(["plots",str(tmp_path),"--dry-run"])
    assert [call[1]["ring_count"] for call in calls] == [2,4]
    assert all("rings" not in call[1] for call in calls)


def test_retired_example_paths_read_the_same_packaged_configuration(shim):
    from importlib.resources import files
    expected=json.loads(files("pymicroglia").joinpath("data","states.example.json").read_text(encoding="utf-8"))
    assert shim.read(Path("analysis/states.example.json"))==expected
