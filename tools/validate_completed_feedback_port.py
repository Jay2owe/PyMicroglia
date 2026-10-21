"""Frozen-sample validation; study paths belong only in this diagnostic tool.

Recompute correction evidence from immutable native artifacts, then compare
each correction boundary with the accepted study. Never promotes a candidate.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import os
import platform
import pickle
import sys
import time
import traceback

platform.platform = lambda *args, **kwargs: 'Windows-native-validation'
PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE/'src'))
import numpy as np
import pandas as pd
import tifffile
from pymicroglia.tracking import engine
from pymicroglia.tracking.completed_evidence import generate
from pymicroglia.tracking.completed_feedback import apply


def table(path):
    try: return pd.read_csv(path,low_memory=False)
    except pd.errors.EmptyDataError: return pd.DataFrame()


def link(source,target):
    source,target = Path(source).resolve(),Path(target).resolve()
    target.parent.mkdir(parents=True,exist_ok=True)
    os.link(source,target)


def evidence_dependencies(root,movie,run_name,config_path,sources):
    native=Path(root)/'m22_accepted_history'/run_name
    history=native/'mid/accepted_history'
    lineage=Path(root)/'m19_layered_lineage'/run_name
    reservation=Path(root)/'m12_identity_reservation'/run_name
    anchor=Path(root)/'m2_anchor'/run_name/'out'
    paths=[Path(sources['raw']),Path(sources['lag']),config_path,
        PACKAGE/'src/pymicroglia/tracking/completed_evidence.py',
        PACKAGE/'src/pymicroglia/tracking/completed_feedback.py',
        PACKAGE/'src/pymicroglia/tracking/data/motion_engine.zip',
        PACKAGE/'src/pymicroglia/tracking/data/completed_feedback_engine.zip',
        native/'out'/f'{movie}.tif',native/'out'/f'{movie}_unclaimed_original_ids.tif',
        lineage/'mid'/f'{movie}_physical_somas.tif',lineage/'mid'/f'{movie}_attempt005_comparison.tif',
        lineage/'mid/comparison_evidence.csv',lineage/'out/comparison_links.csv',
        reservation/'out'/f'{movie}.tif',anchor/f'{movie}_observations.tif',
        anchor/'observations.csv',anchor/'anchor.json']
    paths += [history/name for name in ('26_latent_body_tracks/out/latent_track_points.csv',
        '12_issue023_generalized.tif','13_issue027_short_blank_recovery.tif',
        '15_issue030_moderate_gap_recovery.tif','23_small_cell_blank_recovery/out/producer_metrics.json',
        '07_issue011.tif','07_issue011_unclaimed.tif','06_issue010.tif','08_issue012.tif')]
    paths += [history/stage/'out/producer_audit.csv' for stage in (
        '22_short_blank_recovery','23_small_cell_blank_recovery','24_moderate_gap_recovery')]
    paths += [p for p in (reservation/'out/motion_reservation_audit.csv',
                         reservation/'out/first_handoff_identity_seeds.csv') if p.is_file()]
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe',type=Path,required=True)
    parser.add_argument('--native-map',type=Path,required=True)
    parser.add_argument('--movie',required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--evidence-cache',type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    p = json.loads(args.recipe.read_text())
    sources = json.loads(args.native_map.read_text())['recordings'][args.movie]
    reference = p['movies'][args.movie]
    root = engine._engine(out)
    run_name = 'frozen_native_fixture'
    native = root/'m22_accepted_history'/run_name
    source_native = Path(reference['inputs']['labels']).parent.parent
    # Read-only hard links avoid another reproduction of the large native data.
    for directory,source in [('out',source_native/'out'),
                             ('mid/accepted_history',Path(reference['inputs']['presence_saved']).parent)]:
        for path in source.rglob('*'):
            if path.is_file(): link(path,native/directory/path.relative_to(source))
    for stage,source in [('m19_layered_lineage',Path(sources['physical']).parents[1]),
                         ('m12_identity_reservation',Path(sources['reservation']).parents[1]),
                         ('m2_anchor',Path(sources['observations']).parents[1])]:
        for path in source.rglob('*'):
            if path.is_file(): link(path,root/stage/run_name/path.relative_to(source))
    config = json.loads(Path(sources['config']).read_text())
    config.update(stems=[args.movie],registered_input_dir=str(Path(sources['raw']).parent),
        motion_input_dir=str(Path(sources['lag']).parent),source_frame_offset=p['source_frame_offset'],
        pinned_files={args.movie:{
            'registered_raw':{'relative_to_registered_input_dir':Path(sources['raw']).name},
            'lag_float':{'relative_to_motion_input_dir':Path(sources['lag']).name}}})
    config_path = out/'recording.json'
    config_path.write_text(json.dumps(config,indent=2))
    rows = []
    boundaries = []
    inventories = []
    started = time.perf_counter()
    sys.path.insert(0,str(root/'code'))
    # The package executes this same boundary inside its native import context.
    import optimized_pipeline
    def execute():
        if args.evidence_cache:
            cache=args.evidence_cache.resolve()
            manifest=json.loads(cache.with_suffix('.json').read_text())
            for path,digest in manifest['dependencies'].items():
                assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
            assert manifest['movie']==args.movie
            assert hashlib.sha256(cache.read_bytes()).hexdigest()==manifest['sha256']
            with cache.open('rb') as handle: data=pickle.load(handle)
        else:
            data = generate(root,args.movie,run_name,config_path,out)
            cache=out/'fresh_evidence.pkl'
            with cache.open('wb') as handle: pickle.dump(data,handle,protocol=5)
            dependencies=evidence_dependencies(root,args.movie,run_name,config_path,sources)
            cache.with_suffix('.json').write_text(json.dumps(dict(movie=args.movie,
                sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
                dependencies={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in dependencies}),indent=2))
        for key,value in data.items():
            if key == 'final_evidence':
                for stage,evidence in value.items():
                    ref = reference['inputs'][key][stage]
                    for role,array in evidence.items():
                        if role == 'traces':
                            for i,(before,after,links) in enumerate(array,1):
                                check_array(before,ref[f'before_pass_{i}'],f'{stage}/before_pass_{i}')
                                check_array(after,ref[f'after_pass_{i}'],f'{stage}/after_pass_{i}')
                                check_table(links,ref[f'after_links_{i}'],f'{stage}/after_links_{i}',serialize=False)
                        elif isinstance(array,np.ndarray): check_array(array,ref[role],stage+'/'+role)
                        elif isinstance(array,pd.DataFrame): check_table(array,ref[role],stage+'/'+role,serialize=False)
                        else:
                            same = array == json.loads(Path(ref[role]).read_text())
                            record(stage+'/'+role,same)
                continue
            ref = reference['inputs'][key]
            if isinstance(value,np.ndarray):
                expected = tifffile.imread(ref)
                if key == 'physical': expected=expected[p['native_prefix_frames']:]
                check_array(value,expected,key)
            elif isinstance(value,pd.DataFrame):
                if key == 'recovery_history': expected=pd.concat([table(v) for v in ref],ignore_index=True)
                else: expected=table(ref)
                if key == 'physical_evidence': expected['t']-=p['native_prefix_frames']
                check_table(value,expected,key,serialize=False)
            else: record(key,value==json.loads(Path(ref).read_text())['parameters'])
        apply({args.movie:data},out,source_frame_offset=p['source_frame_offset'],
            callback=checkpoint,inventory_callback=inventory)
        expected={(family,stage) for family,stages in reference['expected'].items() for stage in stages}
        assert set(boundaries)==expected and len(boundaries)==len(expected)
        assert set(inventories)==set(p['inventories']) and len(inventories)==len(p['inventories'])

    def record(name,equal,**kwargs):
        rows.append(dict(name=name,equal=bool(equal),**kwargs))
        (out/'checks.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(json.dumps(rows[-1]),flush=True)
        if not equal: raise AssertionError(name+' differs')

    def check_array(value,expected,name):
        if isinstance(expected,(str,Path)): expected=tifffile.imread(expected)
        equal = value.shape==expected.shape and value.dtype==expected.dtype and np.array_equal(value,expected)
        record(name,equal,shape=list(value.shape),sha256=hashlib.sha256(value.tobytes()).hexdigest(),
            changed_pixels=int(np.count_nonzero(value!=expected)) if value.shape==expected.shape else None)

    def check_table(value,expected,name,columns=None,drop=(),serialize=True,check_dtype=True):
        if isinstance(expected,(str,Path)): expected=table(expected)
        actual=table(io.StringIO(value.to_csv(index=False))) if serialize else value
        expected=expected.drop(columns=list(drop),errors='ignore')
        if columns is not None: actual,expected=actual[columns],expected[columns]
        try:
            pd.testing.assert_frame_equal(actual,expected,check_exact=True,check_dtype=check_dtype)
            representation={column:dict(actual=str(actual[column].dtype),reference=str(expected[column].dtype))
                for column in actual if str(actual[column].dtype)!=str(expected[column].dtype)}
            record(name,True,representation_differences=representation)
        except AssertionError as exc: record(name,False,difference=str(exc)[:3000])

    def checkpoint(key,family,stage,labels,ledger,tables,arrays=None):
        boundaries.append((family,stage))
        ref=reference['expected'][family][stage]
        prefix=family+'/'+stage+'/'
        check_array(labels,ref['labels'],prefix+'labels')
        check_array(ledger,ref['ledger'],prefix+'ledger')
        for name,value in tables.items():
            if isinstance(value,dict): record(prefix+name,value==json.loads(Path(ref['tables'][name]).read_text()))
            else: check_table(value,ref['tables'][name],prefix+name,columns=ref.get('table_columns',{}).get(name))
        for name,value in (arrays or {}).items(): check_array(value,ref['arrays'][name],prefix+name)

    def inventory(stage,value):
        inventories.append(stage)
        ref=table(p['inventories'][stage])
        ref=ref[ref.well.eq(args.movie)].reset_index(drop=True)
        check_table(value.reset_index(drop=True),ref,'inventory/'+stage,
            drop=['reported_join','mark_signature_count','reported_join_historical'],check_dtype=False)

    error=None
    try: optimized_pipeline.main(['--no-evidence-cache','--stage-timings',str(out/'native_timings.jsonl')],runner=execute)
    except Exception:
        error=traceback.format_exc()
        (out/'failure.txt').write_text(error)
    (out/'summary.json').write_text(json.dumps(dict(status='failed' if error else 'pass',
        movie=args.movie,error=error,checks=len(rows),boundaries=boundaries,inventories=inventories,
        seconds=time.perf_counter()-started,
        correction_archive_sha256=(out/'completed_feedback_engine/archive.sha256').read_text().strip()
            if (out/'completed_feedback_engine/archive.sha256').exists() else None,
        scope='Fresh correction evidence from frozen native artifacts; all accepted correction boundaries. Not a raw-to-native replay.'),indent=2)+'\n')
    if error: raise RuntimeError(error)


if __name__ == '__main__': main()
