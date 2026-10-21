"""Real public image, measurement, media and table actions on bounded inputs.

Fixtures come from the package's established synthetic examples. Every action
keeps native numerical defaults except explicit source geometry/time selections.
Failures and partial pipeline results are retained, never counted as completions.
"""
from __future__ import annotations
import argparse
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import traceback
from benchmark_pipeline_execution import ROOT, digest, source_identity, configure_ledger, dependency_environment, process_tree_rss, MEMORY_SAMPLING


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label',required=True)
    parser.add_argument('--actions',nargs='*')
    parser.add_argument('--baseline-ledger',action='store_true')
    args=parser.parse_args()
    base=ROOT/'development/speed-pass-20261011/complete'/args.label/'core'
    base.mkdir(parents=True,exist_ok=True)
    os.environ['PYMICROGLIA_STORE']=str(base/'store')
    os.environ['PYMICROGLIA_INDEX']=str(base/'index')
    os.environ['PYMICROGLIA_DECISIONS']=str(base/'decisions')
    os.environ['MPLBACKEND']='Agg'
    sys.executable=shutil.which('python') or sys.executable
    sys.path.insert(0,str(ROOT/'tests'))
    import numpy as np
    import pandas as pd
    import psutil
    import tifffile
    from pymicroglia.run import run_action
    from pymicroglia import config
    ledger_identity=configure_ledger(args.baseline_ledger)
    assert config.index_root().resolve()==(base/'index').resolve()
    from tests_support import oscillating_stack
    import fixtures
    source=base/'inputs'
    paired=(base.parents[1]/args.label.replace('-after','-before')/'core/inputs'
            if args.label.endswith('-after') else None)
    if paired is not None and paired.exists() and not source.exists():
        # Preserve byte-identical TIFF inputs, including generated OME UUIDs.
        shutil.copytree(paired,source)
    source.mkdir(exist_ok=True)
    raw=source/'oscillating.ome.tif'
    if not raw.exists():raw=oscillating_stack(source,frames=144)
    triple=source/'three_channel.ome.tif'
    if not triple.exists():fixtures.write_series(triple,frames=16,height=180,width=180)
    tracked=source/'tracked'
    tracked.mkdir(exist_ok=True)
    if not (tracked/'raw.tif').exists():fixtures.tracked_movie(tracked,frames=48)
    frozen=ROOT/'tests/fixtures/motion_parity/run'
    copied=base/'saved-run'
    if not copied.exists():shutil.copytree(frozen,copied)
    # Relocate only this procedural fixture's source references. The frozen
    # repository evidence is read-only; generated stacks retain their own hash.
    manifest_path=copied/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    for movie in manifest.get('movies',[]):
        for name in ('raw','labels'):
            path=tracked/(name+'.tif')
            movie.setdefault('provenance',{}).setdefault('inputs',{})[name]={
                'path':str(path),'sha256':digest(path)}
    manifest_path.write_text(json.dumps(manifest,indent=2))
    centroids=source/'centroids.csv'
    if not centroids.exists():
        table=pd.read_csv(frozen/'parity_A1/tables/cell_frame.csv')
        table=table.rename(columns={'centroid_y':'centroid_y_px','centroid_x':'centroid_x_px'})
        table.to_csv(centroids,index=False)
    cfg=json.loads((ROOT/'tests/fixtures/motion_parity/config.json').read_text())
    trace_csv=source/'wide-traces.csv'
    if not trace_csv.exists():
        frame=pd.read_csv(frozen/'parity_A1/tables/cell_frame.csv')
        wide=frame.pivot(index='hours',columns='identity',values='corrected_mean')
        wide.columns=['cell_'+str(c) for c in wide.columns]
        wide.to_csv(trace_csv)
    four=source/'four_channel.ome.tif'
    if not four.exists():
        data=tifffile.imread(triple)
        # A spatially structured transmitted channel gives the registration
        # algorithm identifiable texture; flat random noise has no true shift.
        ordered=data[:,[1,2,2,2],:,:]
        tifffile.imwrite(four,ordered,ome=True,photometric='minisblack',metadata={
            'axes':'TCYX','TimeIncrement':1800.,'TimeIncrementUnit':'s',
            'Channel':{'Name':['dLuc','brightfield','RFP','GFP']}})
    gaps=source/'gaps';gaps.mkdir(exist_ok=True)
    gap_options=importlib.import_module('tests.test_measure_missing_gaps').inputs(gaps)
    common={'source':raw}
    recipes={
        'register':{'source':raw,'margin_px':16},
        'register_three_channel':{'source':triple,'margin_px':16},
        'export_registered_stack':{'source':raw},
        'remove_cosmic_rays':common,'remove_static_background':common,
        'display_filter':common,'background':common,'segment':common,
        'export_roi':common,'extract_traces':common,'run_controls':common,
        'test_rhythm':common,'unmix':common,
        'dluc_single_cell':{'source':raw,'t0':0},
        'cry1_dluc_photon':{'source':four},
        'measure_expanded_outlines':{'raw':tracked/'raw.tif','labels':tracked/'labels.tif'},
        'measure_fixed_apertures':{'movies':[{'stem':'synthetic','raw':tracked/'raw.tif','centroids':centroids,'masks':tracked/'labels.tif'}]},
        'measure_missing_gaps':gap_options,
        'cell_eligibility':{'source':tracked/'labels.tif','frame_interval_h':.5},
        'pool':{'run_dir':copied},'window':{'run_dir':copied,'windows':cfg['windows']},
        'contrasts':{'run_dir':copied,'contrasts':cfg['contrasts'],'metric_groups':cfg['metric_groups']},
        'channel_figure':{'source':triple},'frames_figure':common,
        'cosmic_ray_preview':common,'registration_figure':common,
        'cell_overlay':common,'roi_overlay':common,
        'stack_to_mp4':{'source':raw,'black_point':0.,'white_point':150.},'composite_video':{'source':triple},
        'phase_green_red_video':{'source':triple},'red_only_video':{'source':triple},
        'tracked_cell_image':{'source':tracked/'raw.tif','labels':tracked/'labels.tif','frame_interval_h':.5},
        'tracked_cell_video':{'source':tracked/'raw.tif','labels':tracked/'labels.tif','frame_interval_h':.5},
        'cell_image_grid':{'raw':tracked/'raw.tif','labels':tracked/'labels.tif','display_options':{'frame_interval_h':.5}},
        'cell_video_grid':{'raw':tracked/'raw.tif','labels':tracked/'labels.tif','display_options':{'frame_interval_h':.5}},
        'cell_review':{'raw':tracked/'raw.tif','labels':tracked/'labels.tif','output':base/'cell_review/output/review.html'},
        'follow':{'run':copied,'stem':'parity_A1','identities':[1,2,3]},
        'cell_card_video':{'run':copied,'stem':'parity_A1'},
        'trace_panel':{'input_csvs':[trace_csv],'auto_columns':'cell_*'},
    }
    prior=ROOT/'development/speed-pass-20261011/complete/before1/core'
    if args.actions and 'cell_overlay' in args.actions:
        with np.load(prior/'segment/output/cell_masks_labels.npz',allow_pickle=False) as labels:
            recipes['cell_overlay']={'source':raw,'labels':labels['data']}
    if args.actions and 'registration_figure' in args.actions:
        shifts=pd.read_csv(prior/'register/output/registration_shifts_and_qc.csv')
        recipes['registration_figure']={'source':raw,'shifts':shifts.to_dict('list')}
    if args.actions and 'automatic_scn_outline' in args.actions:
        import auto_organotypic
        import zipfile
        owner=Path(auto_organotypic.__file__).resolve().parents[2]
        archive=owner/'tests/evidence/parity-evidence.zip'
        with zipfile.ZipFile(archive) as bundle:
            names=[name for name in bundle.namelist() if name.startswith('s1_register/') and '/meanred_' in name]
            selected=sorted(names)[0]
            for name in (selected,selected.replace('/meanred_','/validfield_')):
                (source/Path(name).name).write_bytes(bundle.read(name))
        recipes['automatic_scn_outline']={'source':source/Path(selected).name}
    if args.actions and 'publication_workbook' in args.actions:
        masters=sorted((ROOT/'development/speed-pass-20261011/complete/before1/figures/displacement_curves/output').glob('*.svg'))
        if not masters:raise FileNotFoundError('The validated displacement figure baseline is required')
        recipes['publication_workbook']={'source':masters[0],'output_path':base/'publication_workbook/output/evidence.xlsx'}
    # Public tracking is exercised with actual prepared evidence. These synthetic
    # masks are supplied tracker inputs, not a claim of U-Net cell recovery.
    if args.actions and 'track' in args.actions:
        from pymicroglia.tracking.prepare import build
        yy,xx=np.mgrid[:128,:128];masks=np.zeros((48,128,128),np.uint16)
        for identity,(y,x) in enumerate([(y,x) for y in (24,64,104) for x in (24,64,104)],1):
            masks[:,(yy-y)**2+(xx-x)**2<=7**2]=identity
        pixels=((masks>0)*100+np.random.default_rng(0).random(masks.shape)*2).astype('float32')
        tracking_masks=source/'tracking-masks.tif'
        tifffile.imwrite(tracking_masks,masks,metadata={'axes':'TYX'},photometric='minisblack')
        handoff=source/'motion_inputs.json'
        if handoff.exists() and paired is not None:
            data=handoff.read_text().replace(json.dumps(str(paired))[1:-1],json.dumps(str(source))[1:-1])
            handoff.write_text(data)
        else:
            prepared=build('synthetic',pixels,tracking_masks,
                           source/'motion',frame_interval_min=30,dataset='Controlled tracking performance inputs')
            handoff.write_text(json.dumps({'stems':['synthetic'],'prepared':{'synthetic':prepared}},default=str))
        recipes['track']={'inputs':handoff,'folder':base/'track/output'}
    # The public consumers need actual saved registration, masks, traces and
    # controls. Preserve that declared producer order even for alphabetic scopes.
    requested=args.actions or list(recipes)
    names=[name for name in recipes if name in requested]
    names.extend(name for name in requested if name not in recipes)
    process=psutil.Process();before=source_identity()
    input_hashes={p.relative_to(source).as_posix():digest(p) for p in source.rglob('*') if p.is_file()}
    for action in names:
        target=base/action;target.mkdir(exist_ok=False)
        receipt={'id':'action:'+action,'kind':'execution','label':args.label,'status':'failed',
            'recipe':'established synthetic fixture; public action with live defaults and documented input selections',
            'source_before':before,'ledger_implementation':ledger_identity,
            'input_root':str(source),'input_hashes':input_hashes}
        done=threading.Event();memory=[process.memory_info().rss]
        def monitor():
            while not done.wait(.05):memory.append(process_tree_rss(process))
        thread=threading.Thread(target=monitor,daemon=True);thread.start()
        started=time.perf_counter()
        try:
            if shutil.disk_usage(base).free<600*1024**2:raise OSError('Less than 600 MiB benchmark headroom')
            params=dict(recipes[action])
            if action not in {'pool','window','contrasts','track','cell_review','publication_workbook'}:params['output_dir']=target/'output'
            receipt['params']={k:str(v) for k,v in params.items()}
            started=time.perf_counter()
            result=run_action(action,claim='Controlled synthetic action performance verification; no biological finding.',**params)
            receipt['seconds']=time.perf_counter()-started
            if hasattr(result,'successful') and not result.successful:raise AssertionError('Pipeline returned unsuccessful')
            if isinstance(result,dict) and result.get('status') in {'failed','unavailable','blocked'}:raise AssertionError(result)
            if isinstance(result,dict) and 'clips' in result and not result['clips']:raise AssertionError('No requested clip was produced')
            receipt['status']='measured';receipt['result']=str(result)[:10000]
        except Exception:receipt['error']=traceback.format_exc();receipt['seconds']=time.perf_counter()-started
        finally:
            done.set();thread.join()
            receipt['peak_rss_bytes']=max(memory);receipt['retained_rss_bytes']=process.memory_info().rss
            receipt['memory_scope']='action process and owned descendants; sampled RSS sum'
            receipt['memory_sampling']=MEMORY_SAMPLING
            receipt['outputs']={p.relative_to(target).as_posix():{'sha256':digest(p),'bytes':p.stat().st_size}
                                for p in (target/'output').rglob('*') if p.is_file()}
            receipt['source_unchanged']=before==source_identity()
            receipt['dependencies_unchanged']=ledger_identity['dependency_environment']==dependency_environment()
            (target/'receipt.json').write_text(json.dumps(receipt,indent=2,default=str),encoding='utf-8')
            print(json.dumps({k:receipt[k] for k in ['id','status','seconds','peak_rss_bytes']}),flush=True)
            if receipt.get('error'):print(receipt['error'].splitlines()[-1],flush=True)


if __name__=='__main__':main()
