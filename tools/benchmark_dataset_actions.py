"""Complete optional measurement inputs, pooling/windows, learning and tracking.

All numerical operations use the public registry. The long fixture explicitly
requests daily profiles because two public figures require that scientific input;
the synthetic daily signal is declared here, never inferred from successful fits.
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
from benchmark_pipeline_execution import ROOT,digest,source_identity,configure_ledger,dependency_environment,process_tree_rss,MEMORY_SAMPLING


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--label',required=True)
    parser.add_argument('--baseline-ledger',action='store_true')
    parser.add_argument('--common-period-fixture',action='store_true',
        help='Prepare clean known 24 h signals with native FFT-NLLS estimates and independent Lomb tests for common-period figures')
    parser.add_argument('--actions',nargs='*',default=['measure','window','pool','contrasts','states','cluster'])
    args=parser.parse_args();base=ROOT/'development/speed-pass-20261011/complete'/args.label/'dataset'
    base.mkdir(parents=True,exist_ok=True)
    os.environ['PYMICROGLIA_STORE']=str(base/'store');os.environ['PYMICROGLIA_INDEX']=str(base/'index')
    os.environ['MPLBACKEND']='Agg';sys.executable=shutil.which('python') or sys.executable
    sys.path.insert(0,str(ROOT/'tests'))
    import fixtures
    import numpy as np
    import pandas as pd
    import psutil
    import tifffile
    from pymicroglia.run import run_action
    from pymicroglia.measure.modules import MODULE_NAMES
    shared=configure_ledger(args.baseline_ledger)
    source=base/'inputs'
    paired=(base.parents[1]/args.label.replace('-after','-before')/'dataset/inputs'
            if args.label.endswith('-after') else None)
    if paired is not None and paired.exists() and not source.exists():
        shutil.copytree(paired,source)
    source.mkdir(exist_ok=True)
    if not (source/'labels.tif').exists():
        files=fixtures.tracked_movie(source,frames=240)
        labels=tifffile.imread(files['labels'])
        # Move the second synthetic cell close enough to exercise real contacts.
        moved=np.zeros_like(labels)
        for f in range(len(labels)):
            moved[f]=labels[f];moved[f][labels[f]==2]=0
            y,x=np.where(labels[f]==2);moved[f,y-8,x-6]=2
        rng=np.random.default_rng(0)
        raw=((moved>0)*(100+30*np.sin(2*np.pi*np.arange(240)/48))[:,None,None]
             +rng.random(moved.shape)*5).astype('float32')
        if args.common_period_fixture:
            raw=np.full(moved.shape,10.,dtype='float64')
            for identity in (1,2,3):
                signal=110+30*np.sin(2*np.pi*np.arange(240)/48+(identity-1)*.5)
                raw=np.where(moved==identity,signal[:,None,None],raw)
        tifffile.imwrite(files['labels'],moved,metadata={'axes':'TYX'},photometric='minisblack')
        tifffile.imwrite(files['raw'],raw,metadata={'axes':'TYX'},photometric='minisblack')
    movie={'stem':'parity_A1','raw':source/'raw.tif','labels':source/'labels.tif',
        'unclaimed':source/'unclaimed.tif','evidence':source/'evidence.tif',
        'provenance':source/'provenance.tif','valid_mask':source/'valid_mask.tif',
        'channels':[{'name':'extra','path':source/'extra_channel.tif','channel_index':1,
                     'shifts':source/'extra_shifts.csv','shift_columns':['shift_y','shift_x']}],
        'objects':[{'name':'scenery','path':source/'scenery_objects.tif'}]}
    # Separate learning fixtures preserve actual independent subject identifiers.
    learning=source/'states-input';(learning/'pooled').mkdir(parents=True,exist_ok=True)
    clustering=source/'cluster-input';(clustering/'pooled').mkdir(parents=True,exist_ok=True)
    frame=importlib.import_module('tests.test_states').frames()
    summary=importlib.import_module('tests.test_cell_clustering').measurements()
    frame.to_csv(learning/'pooled/cell_frame.csv',index=False)
    summary.to_csv(clustering/'pooled/cell_summary.csv',index=False)
    (learning/'manifest.json').write_text(json.dumps({'synthetic':True,'movies':[],
        'description':'Independent software learning fixtures from test_states and test_cell_clustering; no biological findings'}))
    (clustering/'manifest.json').write_text((learning/'manifest.json').read_text())
    measured=base/'measurement'/'measured'
    cfg=json.loads((ROOT/'tests/fixtures/motion_parity/config.json').read_text())
    recipes={
        'measure':{'movies':[movie],'output_dir':base/'measurement','run_label':'measured','frame_interval_min':30,
                   'enabled_modules':list(MODULE_NAMES),
                   'module_options':{'rhythms':{'metrics':['corrected_mean','integrated_density_mean3'],
                                      'daily_profile_measures':True}}},
        'window':{'run_dir':measured,'windows':cfg['windows']},
        'pool':{'run_dir':measured},
        'contrasts':{'run_dir':measured,'contrasts':cfg['contrasts'],'metric_groups':cfg['metric_groups']},
        'states':{'run':learning,'output_dir':base/'states'},
        'cluster':{'run':clustering,'output_dir':base/'cluster'},
    }
    if args.common_period_fixture:
        # Actual native fits, not edited period/significance columns. The clean
        # source contains no drift; a free FFT-NLLS estimate can recover its
        # declared 24 h period precisely enough for strict common-period views.
        shared_rhythm={'metrics':['corrected_mean'],'detrend':'none',
                       'period_estimation_method':'fft_nlls'}
        recipes['measure']['module_options']={
            'rhythms':{**shared_rhythm,'daily_profile_measures':True},
            'coupling':{'between_metrics':['corrected_mean'],
                        'detrend':'none','period_estimation_method':'fft_nlls'}}
    before=source_identity();process=psutil.Process()
    input_hashes={p.relative_to(source).as_posix():digest(p) for p in source.rglob('*') if p.is_file()}
    for action in args.actions:
        target=base/action;target.mkdir(exist_ok=False)
        receipt={'id':'action:'+action,'label':args.label,'kind':'execution','status':'failed',
                 'source_before':before,'ledger_implementation':shared,'params':recipes[action],
                 'input_root':str(source),'input_hashes':input_hashes,
                 'recipe':'240 half-hour frames; three procedural cells with one contact pair; declared 24 h signal; optional daily profiles explicitly enabled for figure prerequisites'}
        memory=[process.memory_info().rss];done=threading.Event()
        def monitor():
            while not done.wait(.05):memory.append(process_tree_rss(process))
        thread=threading.Thread(target=monitor,daemon=True);thread.start();start=time.perf_counter()
        try:
            if shutil.disk_usage(base).free<600*1024**2:raise OSError('Less than 600 MiB benchmark headroom')
            result=run_action(action,claim='Controlled synthetic performance verification; no biological finding.',**recipes[action])
            receipt['seconds']=time.perf_counter()-start
            receipt['result']=result;receipt['status']='measured'
        except Exception:receipt['error']=traceback.format_exc();receipt['seconds']=time.perf_counter()-start
        finally:
            done.set();thread.join();receipt['peak_rss_bytes']=max(memory)
            receipt['retained_rss_bytes']=process.memory_info().rss
            receipt['memory_scope']='action process and owned descendants; sampled RSS sum'
            receipt['memory_sampling']=MEMORY_SAMPLING
            receipt['source_unchanged']=before==source_identity()
            receipt['dependencies_unchanged']=shared['dependency_environment']==dependency_environment()
            output_root=(measured if action in {'measure','window','pool','contrasts'} else target)
            receipt['outputs']={Path(os.path.relpath(p,target)).as_posix():{'sha256':digest(p),'bytes':p.stat().st_size}
                                for p in output_root.rglob('*') if p.is_file() and p.name!='receipt.json'}
            (target/'receipt.json').write_text(json.dumps(receipt,indent=2,default=str),encoding='utf-8')
            print(json.dumps({k:receipt[k] for k in ['id','status','seconds','peak_rss_bytes']}),flush=True)
            if receipt.get('error'):print(receipt['error'].splitlines()[-1],flush=True)


if __name__=='__main__':main()
