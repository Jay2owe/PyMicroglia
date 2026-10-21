"""Real browser completion, restart and external edits, with owned server memory.

Instruments paint completion without substituting requests, data, or renderers.
Baseline uses the frozen pre-pass viewer; both start with an empty browser cache.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from urllib.parse import urlencode
from urllib.request import urlopen
from benchmark_pipeline_execution import ROOT,digest,dependency_environment,source_identity,process_tree_rss,MEMORY_SAMPLING
from benchmark_progressive_viewer import port,serve

HOOK = r'''
;(() => {const original=window.PyMicrogliaExploration.mount;
 window.benchmarkImageDigest=async(image)=>{
   if(!image)return null;const canvas=document.createElement('canvas');
   canvas.width=image.naturalWidth;canvas.height=image.naturalHeight;
   const context=canvas.getContext('2d');context.drawImage(image,0,0);
   const digest=await crypto.subtle.digest('SHA-256',context.getImageData(0,0,canvas.width,canvas.height).data);
   return Array.from(new Uint8Array(digest),b=>b.toString(16).padStart(2,'0')).join('');
 };
 window.PyMicrogliaExploration.mount=(...args)=>{
 const e=original(...args);window.benchmarkPaints={};window.benchmarkTissue=null;
 const tissueDraw=e.drawTectonics.bind(e);
 e.drawTectonics=function(view='tissue-map'){
   const key=e.tissueKey(view),selection={identity:e.state.identity,frame:e.state.frame};
   const data=e.data,identities=[...data.identities],result=tissueDraw(view),entry=e.tissueCache.get(key);
   if(view==='tissue-map'){
     const image=entry?.image,error=entry?.error;
     requestAnimationFrame(()=>{window.benchmarkTissue={...selection,key,data,identities,image,error};});
   }
   return result;
 };
 for(const name of ['drawImage','drawTrace','drawBarcode','drawEvents','drawMap','drawNetwork','drawHistory','drawTraceGrid','drawCellCards','drawChange','drawClock','drawDistribution']){
   if(typeof e[name]!=='function')continue;const draw=e[name].bind(e);
   e[name]=function(...values){const selection={identity:e.state?.identity,frame:e.state?.frame};const result=draw(...values);
     requestAnimationFrame(()=>{window.benchmarkPaints[name]={...selection,time:performance.now()};});return result;};
 } return e;};})();
'''


def runtime_assets():
    return {p.relative_to(ROOT).as_posix():digest(p)
            for p in (ROOT/'src/pymicroglia/ui/static').rglob('*')
            if p.is_file() and p.suffix in {'.js','.css','.html'}}


def stop_server(server):
    """End the owned server tree, including a Windows application-alias child."""
    import psutil
    try:
        parent=psutil.Process(server.pid)
        owned=[*reversed(parent.children(recursive=True)),parent]
    except psutil.NoSuchProcess:
        return []
    ids=[p.pid for p in owned]
    for process in owned:
        try:process.terminate()
        except psutil.NoSuchProcess:pass
    _,alive=psutil.wait_procs(owned,timeout=10)
    for process in alive:
        try:process.kill()
        except psutil.NoSuchProcess:pass
    _,alive=psutil.wait_procs(alive,timeout=5)
    if alive:raise RuntimeError('Owned server processes survived shutdown: '+str([p.pid for p in alive]))
    server.wait(timeout=5)
    return ids


def complete_screen_fixture(root):
    """Keep both growth sizes inside the unchanged interactive tissue budget.

    The separate preview benchmark retains its larger 512x768 fixture. This
    complete-screen case must actually render the native tissue map as well.
    """
    import numpy as np
    import tifffile
    root.mkdir()
    recipe={'seed':111026,'frames':48,'shape':[256,384],'cells':96,
            'purpose':'Complete native screen and doubled-timepoint growth within 512 MiB tissue budget'}
    raw=np.random.default_rng(recipe['seed']).integers(0,4000,(48,256,384),dtype=np.uint16)
    labels=np.zeros_like(raw);rows=[]
    for frame in range(48):
        for index in range(96):
            y,x=6+index//12*30,5+index%12*31+frame%4
            labels[frame,y:y+8,x:x+9]=index+1
            rows.append([index+1,frame,frame,100+20*np.cos(2*np.pi*(frame-index-1)/24),72,x+4,y+3.5])
    for name,data in [('raw',raw),('labels',labels)]:
        tifffile.imwrite(root/(name+'.tif'),data,photometric='minisblack',compression='zlib')
    with (root/'cell_frame.csv').open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['identity','frame_index','hours','integrated_density','area_px','centroid_x','centroid_y'])
        writer.writerows(rows)
    (root/'manifest.json').write_text(json.dumps({'movies':[{'stem':'synthetic',
        'spec':{'raw':'raw.tif','labels':'labels.tif'},'display':{'contrast':[0,4000]},
        'tables':{'cell_frame':{'path':'cell_frame.csv'}}}]}))
    (root/'recipe.json').write_text(json.dumps(recipe,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--serve',type=int)
    parser.add_argument('--baseline',type=Path);parser.add_argument('--label',default='boundaries1')
    args=parser.parse_args()
    if args.serve:return serve(args.serve,args.baseline)
    from playwright.sync_api import sync_playwright
    import psutil
    base=ROOT/'development/speed-pass-20261011';out=base/'complete'/args.label
    out.mkdir(parents=True,exist_ok=False)
    source=out/'fixture';complete_screen_fixture(source)
    # A second, bounded recording doubles time points without changing native
    # image dimensions or identities. Input construction is outside all timers.
    if shutil.disk_usage(out).free<700*1024**2:raise OSError('Less than 700 MiB growth-test headroom')
    import numpy as np
    import pandas as pd
    import tifffile
    grown=out/'growth-fixture';grown.mkdir()
    for name in ('raw','labels'):
        pixels=tifffile.imread(source/(name+'.tif'))
        tifffile.imwrite(grown/(name+'.tif'),np.concatenate([pixels,pixels]),photometric='minisblack',compression='zlib')
    del pixels
    frame=pd.read_csv(source/'cell_frame.csv');later=frame.copy()
    later['frame_index']+=48;later['hours']+=48
    pd.concat([frame,later],ignore_index=True).to_csv(grown/'cell_frame.csv',index=False)
    shutil.copy2(source/'manifest.json',grown/'manifest.json')
    result={'operations':[],'errors':[],'restart_checks':[],'browser_versions':[],
            'fixture_sha256':{p.name:digest(p) for p in source.iterdir() if p.is_file()},
            'growth_fixture_sha256':{p.name:digest(p) for p in grown.iterdir() if p.is_file()},
            'dependency_environment':dependency_environment(),'source_before':source_identity(),
            'runtime_assets_before':runtime_assets(),'memory_sampling':MEMORY_SAMPLING}
    original_table=(source/'cell_frame.csv').read_bytes()
    try:
        with sync_playwright() as pw:
            for label in ['before','after','after','before']:
                trial=len(result['operations']);chosen=port();url=f'http://127.0.0.1:{chosen}'
                log=(out/f'{trial}-{label}-server.log').open('w')
                command=[shutil.which('python'),str(Path(__file__).resolve()),'--serve',str(chosen)]
                if label=='before':command+=['--baseline',str(base/'baseline')]
                env={**os.environ,'PYMICROGLIA_STORE':str(out/'store'),'PYMICROGLIA_INDEX':str(out/'index')}
                started=time.perf_counter();server=subprocess.Popen(command,stdout=log,stderr=log,env=env)
                done=threading.Event();memory=[]
                owned_root=psutil.Process()
                def sample():
                    while not done.wait(.05):
                        try:
                            # All descendants belong to this isolated harness:
                            # its server, browser, driver and their workers.
                            memory.append(process_tree_rss(owned_root,include_parent=False))
                        except psutil.Error:pass
                monitor=threading.Thread(target=sample,daemon=True);monitor.start()
                browser=None
                try:
                    deadline=time.monotonic()+90
                    while True:
                        try:
                            with urlopen(url+'/api/config',timeout=1) as response:assert response.status==200
                            break
                        except Exception:
                            if time.monotonic()>deadline:raise TimeoutError('Server startup')
                            time.sleep(.2)
                    browser=pw.chromium.launch(channel='chrome',headless=True,args=['--disable-gpu'])
                    result['browser_versions'].append(browser.version)
                    context=browser.new_context(viewport={'width':1440,'height':1000})
                    page=context.new_page();page.on('pageerror',lambda error:result['errors'].append(str(error)))
                    def instrument(route):
                        response=route.fetch();route.fulfill(response=response,body=response.text()+HOOK)
                    page.route('**/assets/exploration.js*',instrument)
                    page.add_init_script("localStorage.setItem('pm-workspace','micro');localStorage.setItem('pm-page','follow');")
                    page.goto(url+'/?'+urlencode({'run':str(source)}))
                    ready='''() => {const e=window.pyMicrogliaExplorer,p=window.benchmarkPaints||{};
                      if(!e?.image || !e?.exact || !e?.history || e.mediaBusy || e.data?.preview?.pending
                         || e.imageMeta?.preview || e.exact.cell.identity!==e.state.identity
                         || e.frameValues?.frame?.frame_index!==e.state.frame
                         || !e.changeCache.has(e.state.frame))return false;
                      if(e.gridRows?.size!==e.data.identities.length)return false;
                      const comparison=e.groupComparison?.states.get('distribution');
                      if(!comparison?.preview || !comparison.image)return false;
                      const painted=['drawImage','drawChange','drawTrace','drawHistory','drawBarcode','drawEvents','drawMap','drawNetwork','drawCellCards','drawClock','drawTraceGrid','drawDistribution'];
                      const needsTissue=e.comparisons.value('map-layer','tissue-map').startsWith('tectonics:');
                      const tissue=needsTissue?window.benchmarkTissue:null;
                      if(needsTissue && (!tissue || (!tissue.image && !tissue.error)
                         || tissue.data!==e.data || tissue.identity!==e.state.identity || tissue.frame!==e.state.frame
                         || JSON.stringify(tissue.identities)!==JSON.stringify(e.data.identities)
                         || JSON.stringify(JSON.parse(tissue.key).slice(0,3))!==JSON.stringify(JSON.parse(e.tissueKey()).slice(0,3))))return false;
                      return painted.every(name=>p[name]?.identity===e.state.identity && p[name]?.frame===e.state.frame)
                        && (e.peerIdentities||[]).every(i=>e.peerExact.has(i)&&e.peerRows.get(i)?.has(e.state.frame))
                        && (!e.previewContacts || (e.contactCache.has(e.state.frame)&&!e.contactBusy))
                        && !String(e.data.results?.rhythms?.message||'').startsWith('Computing');
                    }'''
                    page.wait_for_function(ready,timeout=120000)
                    initial=time.perf_counter()-started
                    initial_peak=max(memory)
                    snapshot=page.evaluate('''async() => {const e=window.pyMicrogliaExplorer;return {
                        identities:e.data.identities,frames:e.data.frames,metrics:e.data.metrics,
                        exact:e.exact,history:e.history,image:[e.image.naturalWidth,e.image.naturalHeight],
                        gridCells:e.gridRows.size,comparisonCounts:e.groupComparison.states.get('distribution').preview.counts,
                        comparisonPixels:await window.benchmarkImageDigest(e.groupComparison.states.get('distribution').image),
                        paints:Object.keys(window.benchmarkPaints),imageError:e.imageError,historyError:e.historyError,
                        tissueError:window.benchmarkTissue?.error||'',
                        imagePixels:await window.benchmarkImageDigest(e.image),
                        tissuePixels:await window.benchmarkImageDigest(window.benchmarkTissue?.image),
                        tissueSize:window.benchmarkTissue?.image?
                          [window.benchmarkTissue.image.naturalWidth,window.benchmarkTissue.image.naturalHeight]:null};}''')
                    if snapshot['imageError'] or snapshot['historyError'] or snapshot['tissueError']:raise AssertionError(snapshot)
                    rows=[{'id':'workflow:initial-usable-screen','seconds':initial,'snapshot':snapshot,
                           'peak_rss_bytes':initial_peak}]
                    for kind in ['selection','restart','external-edit','growth']:
                        memory_offset=len(memory)
                        t=time.perf_counter()
                        if kind=='selection':
                            page.evaluate("() => {const e=window.pyMicrogliaExplorer;e.select({identity:2,frame:20});}")
                            page.wait_for_function("window.pyMicrogliaExplorer.state.identity===2 && window.pyMicrogliaExplorer.state.frame===20 && !window.pyMicrogliaExplorer.mediaBusy && window.pyMicrogliaExplorer.imageMeta?.frame_index===20 && !window.pyMicrogliaExplorer.imageMeta.preview",timeout=60000)
                            page.wait_for_function(ready,timeout=120000)
                            identifier='workflow:selected-item-full-detail'
                        elif kind=='growth':
                            page.goto(url+'/?'+urlencode({'run':str(grown)}))
                            page.wait_for_function(ready,timeout=120000)
                            page.wait_for_function('window.pyMicrogliaExplorer.data.frames.length===96',timeout=60000)
                            identifier='workflow:growth'
                        else:
                            if kind=='restart':
                                stopped=stop_server(server)
                                server=subprocess.Popen(command,stdout=log,stderr=log,env=env)
                                deadline=time.monotonic()+90
                                while True:
                                    try:
                                        with urlopen(url+'/api/config',timeout=1) as response:assert response.status==200
                                        break
                                    except Exception:
                                        if time.monotonic()>deadline:raise TimeoutError('Server restart')
                                        time.sleep(.2)
                                if server.poll() is not None:raise RuntimeError('Replacement server exited')
                                result['restart_checks'].append({'label':label,'trial':trial,
                                    'terminated_processes':stopped,'new_pid':server.pid,'passed':bool(stopped)})
                            if kind=='external-edit':
                                import io
                                table=list(csv.DictReader(io.StringIO(original_table.decode())))
                                for row in table:
                                    if row['identity']=='2':row['integrated_density']='12345.0'
                                with (source/'cell_frame.csv').open('w',newline='') as stream:
                                    writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
                            page.reload();page.wait_for_function(ready,timeout=120000)
                            identifier='workflow:'+kind
                        elapsed=time.perf_counter()-t
                        boundary_peak=max(memory[memory_offset:] or memory[-1:])
                        detail=page.evaluate('''async() => {const e=window.pyMicrogliaExplorer;return {state:e.state,exact:e.exact,
                            image:[e.image.naturalWidth,e.image.naturalHeight],preview:!!e.imageMeta.preview,
                            gridCells:e.gridRows.size,comparisonCounts:e.groupComparison.states.get('distribution').preview.counts,
                            comparisonPixels:await window.benchmarkImageDigest(e.groupComparison.states.get('distribution').image),
                            imagePixels:await window.benchmarkImageDigest(e.image),
                            tissuePixels:await window.benchmarkImageDigest(window.benchmarkTissue?.image),
                            tissueError:window.benchmarkTissue?.error||''};}''')
                        if detail['tissueError']:raise AssertionError(detail['tissueError'])
                        if kind=='external-edit' and '12345' not in json.dumps(detail['exact']):
                            raise AssertionError('External table edit did not reach the displayed exact cell')
                        rows.append({'id':identifier,'seconds':elapsed,'snapshot':detail,
                                     'peak_rss_bytes':boundary_peak})
                    page.screenshot(path=str(out/f'{trial}-{label}.png'))
                    for row in rows:row.update(label=label,kind='execution',status='measured',
                        memory_scope='isolated server and browser descendants; sampled RSS sum')
                    result['operations'].extend(rows)
                    print(json.dumps([{k:v for k,v in row.items() if k!='snapshot'} for row in rows]),flush=True)
                except Exception:
                    if browser:
                        try:
                            page.screenshot(path=str(out/f'{trial}-{label}-failure.png'))
                            state=page.evaluate("() => {const e=window.pyMicrogliaExplorer;return {state:e?.state,paints:window.benchmarkPaints,mediaBusy:e?.mediaBusy,imageMeta:e?.imageMeta,hasExact:!!e?.exact,hasHistory:!!e?.history,frame:e?.frameValues?.frame,imageError:e?.imageError,historyError:e?.historyError,tissue:e?e.tissueCache.get(e.tissueKey())?.error:null};}")
                            (out/f'{trial}-{label}-failure-state.json').write_text(json.dumps(state,indent=2))
                        except Exception:pass
                    raise
                finally:
                    (source/'cell_frame.csv').write_bytes(original_table)
                    if browser:browser.close()
                    stop_server(server)
                    done.set();monitor.join();log.close()
    except Exception:
        import traceback
        result['failure']=traceback.format_exc();raise
    finally:
        result['dependencies_unchanged']=result['dependency_environment']==dependency_environment()
        result['source_unchanged']=result['source_before']==source_identity()
        result['runtime_assets_unchanged']=result['runtime_assets_before']==runtime_assets()
        (out/'receipt.json').write_text(json.dumps(result,indent=2,default=str),encoding='utf-8')


if __name__=='__main__':main()
