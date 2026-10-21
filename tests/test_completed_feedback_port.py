"""Full accepted correction packaging, import isolation and result boundary."""
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType
import zipfile

import pytest

from pymicroglia.tracking import completed_feedback as corrections
from pymicroglia.tracking import engine


def test_complete_recipe_and_all_sources_are_packaged():
    archive = Path(corrections.__file__).parent/'data/completed_feedback_engine.zip'
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read('source_hashes.json'))
        assert manifest['accepted_attempt'] == 'R218-A004'
        assert manifest['saved_feedback'] == {'satisfied':1798,'total':2348}
        for name,digest in manifest['files_sha256'].items():
            assert hashlib.sha256(z.read(name)).hexdigest() == digest,name
        settings = json.loads(z.read('settings.json'))
        assert set(settings) == {'density_parameters','operations','revise_isolated_prior_reassignment'}
        assert [op['kind'] for op in settings['operations']['final']] == [
            'gap','restoration','gap','allocation','separation']
        assert settings['operations']['final'][-1]['stage'] == 'r218_a004'
        for group in settings['operations'].values():
            for spec in group:
                assert not Path(spec['code_dir']).is_absolute()
                for name,digest in spec['code_sha256'].items():
                    assert hashlib.sha256(z.read(spec['code_dir']+'/'+name+'.py')).hexdigest()==digest
        assert '95_A' not in z.read('settings.json').decode()


def test_imports_outside_motion_and_restores_existing_modules(tmp_path,monkeypatch):
    root = corrections.extract(tmp_path)
    sentinel = ModuleType('completed_feedback_sequence')
    monkeypatch.setitem(sys.modules,'completed_feedback_sequence',sentinel)
    before = sys.path.copy()
    with corrections.methods(root) as sequence:
        assert callable(sequence.apply)
        assert sequence is not sentinel
    assert sys.modules['completed_feedback_sequence'] is sentinel
    assert sys.path == before


def test_evidence_context_preserves_tracker_being_patched(tmp_path,monkeypatch):
    root = corrections.extract(tmp_path)
    native = ModuleType('tracking')
    native.__file__ = str(root/'code/tracking.py')
    # Use a real module so the other accepted functions can import its API.
    with corrections.methods(root):
        import tracking
        native = tracking
    monkeypatch.setitem(sys.modules,'tracking',native)
    with corrections.methods(root,preserve_native=True):
        assert sys.modules['tracking'] is native
    assert sys.modules['tracking'] is native


@pytest.mark.parametrize('bounded',[False,True])
def test_native_execution_always_runs_complete_corrections(tmp_path,monkeypatch,bounded):
    root = engine._engine(tmp_path)
    order = []
    monkeypatch.setattr(engine,'_execute_pipeline',lambda *args,**kwargs: order.append('native'))
    monkeypatch.setattr(engine,'_execute_corrections',lambda *args,**kwargs: order.append('complete'))
    engine._run_frozen(root,'movie','run',tmp_path/'config.json',tmp_path/'run.log',bounded_memory=bounded)
    assert order == ['native','complete']


def test_packaging_adaptation_preserves_every_scientific_function():
    package = Path(corrections.__file__).resolve().parents[3]
    motion = package.parent/'Motion'
    if not motion.is_dir():
        pytest.skip('Build-source check; source checkout not installed')
    with zipfile.ZipFile(Path(corrections.__file__).parent/'data/completed_feedback_engine.zip') as z:
        manifest=json.loads(z.read('source_hashes.json'))
        for name,adaptation in manifest['packaging_adaptations'].items():
            original=(motion/manifest['source_origins'][name]).read_bytes()
            assert hashlib.sha256(original).hexdigest()==adaptation['original_sha256']
            def functions(data):
                tree=ast.parse(data.decode('utf-8-sig'))
                return [ast.dump(node) for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef))]
            assert functions(original)==functions(z.read(name))


def test_movie_without_process_pairs_does_not_require_other_movies(tmp_path,monkeypatch):
    import pandas as pd
    root=corrections.extract(tmp_path)
    with corrections.methods(root):
        import process_group_composition_v2 as groups
        first=ModuleType('empty_inventory')
        first.scan=lambda movies,table: pd.DataFrame()
        later=ModuleType('refine_inventory')
        def forbidden(*args):
            raise AssertionError('A refining operation cannot run without source pairs')
        later.scan=forbidden
        monkeypatch.setattr(groups,'INVENTORIES',[('first',first),('later',later)])
        tables=list(groups.inventory({'arbitrary_new_movie':{}}))
        assert [stage for stage,_ in tables]==['first','later']
        assert all(table.empty and 'well' in table for _,table in tables)


def test_full_port_checker_rejects_native_only_and_missing_final_stage(tmp_path):
    import importlib.util
    tool=Path(__file__).resolve().parents[1]/'tools/verify_completed_feedback_acceptance.py'
    spec=importlib.util.spec_from_file_location('port_completion',tool)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    archive=Path(corrections.__file__).parent/'data/completed_feedback_engine.zip'
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    recipe=tmp_path/'recipe.json'
    recipe.write_text(json.dumps(dict(movies={'movie':{'expected':{'final':{'r218_a004':{}}}}},inventories={'tracked':{}})))
    report=tmp_path/'summary.json'
    report.write_text(json.dumps(dict(status='pass',native_only=True)))
    with pytest.raises(ValueError,match='every accepted movie'):
        module.verify(recipe,[report],archive)
    report.write_text(json.dumps(dict(movie='movie',status='pass',correction_archive_sha256=digest,boundaries=[])))
    with pytest.raises(ValueError,match='Incomplete or failing'):
        module.verify(recipe,[report],archive)


def test_returned_labels_and_decisions_use_completed_result(tmp_path,monkeypatch):
    import numpy as np
    import tifffile
    shape=(3,8,8)
    for name in ('raw','motion'):
        tifffile.imwrite(tmp_path/(name+'.tif'),np.zeros(shape,np.uint16),photometric='minisblack')
    config=tmp_path/'config.json'
    config.write_text(json.dumps(dict(registered_input_dir='.',motion_input_dir='.',source_frame_offset=0)))
    handoff=tmp_path/'inputs.json'
    handoff.write_text(json.dumps(dict(stems=['movie'],prepared={'movie':dict(config=str(config),pins=dict(
        registered_raw=dict(relative_to_registered_input_dir='raw.tif'),
        motion_composite=dict(relative_to_motion_input_dir='motion.tif')))})))
    root=tmp_path/'engine'
    monkeypatch.setattr(engine,'_engine',lambda folder: root)
    def produce(*args,**kwargs):
        for stage,identity in [('m22_accepted_history',1),('m23_completed_feedback',2)]:
            out=root/stage/'pymicroglia_movie/out'
            out.mkdir(parents=True)
            tifffile.imwrite(out/'movie.tif',np.full(shape,identity,np.uint16),photometric='minisblack')
        old=root/'m22_accepted_history/pymicroglia_movie/mid/accepted_history/21_continuity_evidence/out/gap_evidence.csv'
        new=root/'m23_completed_feedback/pymicroglia_movie/mid/completed_feedback/final/r218_a004/decisions.csv'
        for path in (old,new):
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text('decision\nretained\n')
    monkeypatch.setattr(engine,'_run_frozen',produce)
    result=engine.run(handoff,tmp_path)
    assert np.all(tifffile.imread(result.labels)==2)
    present=result.decisions.present(result.labels.parent)
    assert 'gap_evidence' in present
    assert 'completed_feedback/final/r218_a004/decisions' in present


@pytest.mark.parametrize('restoration', [False, True])
def test_second_gap_compares_with_first_gap_result(tmp_path,monkeypatch,restoration):
    """Exercise real evidence orchestration without recomputing cell geometry."""
    from contextlib import nullcontext
    from types import SimpleNamespace
    import numpy as np
    import pandas as pd
    import weakref
    from pymicroglia.tracking import completed_evidence as evidence
    # This fixture tests evidence ancestry and object lifetimes. Native storage
    # adapters are covered by the real-engine tests; its stub common module
    # deliberately does not provide the native tracking implementation.
    monkeypatch.setattr('pymicroglia.tracking.frozen_io.install',
                        lambda *args, **kwargs: nullcontext())
    shape=(2,4,4)
    params=tmp_path/'m22_accepted_history/run/mid/accepted_history/23_small_cell_blank_recovery/out/producer_metrics.json'
    params.parent.mkdir(parents=True)
    params.write_text('{"parameters": {}}')
    def module(name,**values):
        item=ModuleType(name)
        item.__dict__.update(values)
        monkeypatch.setitem(sys.modules,name,item)
    module('common',Config=SimpleNamespace(load=lambda path:SimpleNamespace(source_frame_offset=0)))
    snapshots={}
    owner=SimpleNamespace(reconcile=lambda old,ledger,*a:(old.copy(),ledger.copy(),[],None))
    detector=SimpleNamespace(detect=lambda *a:([],[]))
    reconciler=SimpleNamespace(select_events=lambda *a:[],reconcile=lambda candidate,ledger,*a:(candidate,ledger,None))
    issue_modules={'m1_reconcile_domain_ownership':owner,'m8_detect_dormant_releases':detector,'m9_reconcile_dormant_releases':reconciler}
    module('accepted_history',_field_motion_reservations=lambda run:set(),I012='fixture',
        align_transition_stack=lambda lag,*a:lag,
        _issue_module=lambda issue,name:issue_modules[name],
        _params=lambda *a:dict(methods=[],ownership={},detection={},reconcile={}))
    module('pipeline',_pinned_paths=lambda cfg,stem:dict(registered_raw='raw',lag_float='lag'),
        _load_anchor_inputs=lambda stem,run:(np.zeros(shape,np.uint16),pd.DataFrame(),0))
    calls=[]
    def collect(*args,**kwargs):
        assert snapshots['12_issue023_generalized.tif']() is None
        assert snapshots['13_issue027_short_blank_recovery.tif']() is None
        if restoration and len(calls)==2:
            for name in ('07_issue011.tif','07_issue011_unclaimed.tif','06_issue010.tif'):
                assert snapshots[name]() is None
        calls.append(kwargs.get('rule_context'))
        value=len(calls)
        return [(np.full(shape,value,np.uint16),pd.DataFrame({'pass':[i]})) for i in (1,2)],[],[]
    module('native_tracking_prefix',collect_tracking_prefix=collect)
    module('recovery_sequence_composition',frozen_method=lambda spec:nullcontext())
    module('measured_soma_gap_memory',measured_soma_gap_memory=object())
    module('measured_restoration_continuity_a002',measured_restoration_continuity=lambda *a,**k:nullcontext())
    def read(path):
        array=np.zeros(shape,np.uint16)
        snapshots[Path(path).name]=weakref.ref(array)
        return array
    monkeypatch.setattr(evidence.tifffile,'imread',read)
    monkeypatch.setattr(evidence,'_table',lambda path:pd.DataFrame({'frame':[0],'t':[0]}))
    monkeypatch.setattr(evidence,'extract',lambda folder:folder)
    monkeypatch.setattr(evidence,'methods',lambda root,**kwargs:nullcontext())
    specs=[{'stage':'first','kind':'gap'}]
    if restoration: specs.append({'stage':'restore','kind':'restoration'})
    specs.append({'stage':'second','kind':'gap'})
    monkeypatch.setattr(evidence,'recipe',lambda root:{'operations':{'final':specs}})
    data=evidence.generate(tmp_path,'arbitrary','run',tmp_path/'config.json',tmp_path)
    first=data['final_evidence']['first']['traces']
    second=data['final_evidence']['second']['traces']
    assert len(calls)==3 and calls[0] is None
    for initial,latest in zip(first,second):
        assert np.all(initial[0]==1) and np.all(initial[1]==2)
        assert np.array_equal(latest[0],initial[1])
        assert np.all(latest[1]==3)


@pytest.mark.parametrize('complete',[False,True])
def test_final_publication_requires_all_stages_and_updates_provenance(tmp_path,monkeypatch,complete):
    from types import SimpleNamespace
    import numpy as np
    import pandas as pd
    import tifffile
    from pymicroglia.tracking import completed_evidence as evidence
    root=corrections.extract(tmp_path)
    manifest=json.loads((root/'source_hashes.json').read_text())
    labels=np.ones((2,4,4),np.uint16)
    labels[:,0,0]=0
    result=np.full_like(labels,2)
    ledger=np.zeros_like(labels)
    native=tmp_path/'m22_accepted_history/run/out'
    native.mkdir(parents=True)
    tifffile.imwrite(native/'movie_provenance.tif',np.zeros_like(labels,np.uint8),photometric='minisblack')
    pipeline=ModuleType('pipeline')
    pipeline.PROVENANCE_INFERRED=1
    pipeline.PROVENANCE_ADDED=4
    pipeline._identity_tables=lambda *args:(pd.DataFrame({'frame':[0]}),pd.DataFrame({'identity':[2]}))
    monkeypatch.setitem(sys.modules,'pipeline',pipeline)
    common=ModuleType('common')
    common.Config=SimpleNamespace(load=lambda path:SimpleNamespace(source_frame_offset=0,values={'frame_interval_min':1}))
    common.save_stack=lambda path,array,interval:tifffile.imwrite(path,array,photometric='minisblack')
    monkeypatch.setitem(sys.modules,'common',common)
    import weakref
    evidence_refs=[]
    def generate(*args):
        ancestor=labels.copy()
        evidence_refs.append(weakref.ref(ancestor))
        return dict(labels=labels,ledger=ledger,raw=np.zeros_like(labels),
                    final_evidence={'large_ancestor':ancestor})
    monkeypatch.setattr(evidence,'generate',generate)
    original_save=common.save_stack
    def save(path,array,interval):
        assert all(reference() is None for reference in evidence_refs)
        original_save(path,array,interval)
    common.save_stack=save
    def apply(movies,folder,*,source_frame_offset,callback,inventory_callback):
        boundaries=manifest['required_boundaries'] if complete else manifest['required_boundaries'][:1]
        for family,stage in boundaries:callback('movie',family,stage,result,ledger,{})
        for stage in manifest['required_inventories']:inventory_callback(stage,pd.DataFrame())
        return {'movie':(result,ledger)}
    monkeypatch.setattr(corrections,'apply',apply)
    if not complete:
        with pytest.raises(RuntimeError,match='Incomplete accepted correction sequence'):
            corrections.run_current(tmp_path,'movie','run',tmp_path/'config.json',tmp_path)
        assert not (tmp_path/'m23_completed_feedback/run/out/movie.tif').exists()
    else:
        out=corrections.run_current(tmp_path,'movie','run',tmp_path/'config.json',tmp_path)
        assert np.array_equal(tifffile.imread(out/'out/movie.tif'),result)
        provenance=tifffile.imread(out/'out/movie_provenance.tif')
        assert np.all(provenance[:,0,0]==5)
        assert np.all(provenance[:,1:,1:]==1)
        assert len(json.loads((out/'run.json').read_text())['stages'])==114
