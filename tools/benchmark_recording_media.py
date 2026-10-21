"""Time real recording switches until source pixels AND frame changes are drawn.

Copies measurement metadata into an isolated workspace; image stacks stay read-only.
Run baseline and candidate against the same --fixture for comparable inputs.
"""
import argparse
import base64
import hashlib
from io import BytesIO
import json
from pathlib import Path
import platform
import shutil
import tempfile
import time
import numpy as np
from PIL import Image
import tifffile
from urllib.parse import urlencode

from verify_group_comparison_cdp import Chrome


def fixture(root):
    from pymicroglia.ui.exploration_data import recording, table_path
    from pymicroglia.ui.exploration_images import declared_source
    from pymicroglia.ui.exploration_comparison import workspace, add_runs, save_workspace
    base=Path(__file__).resolve().parents[1]/'development/mask_to_motion_handoff/full_frame_20260921/v2'
    inputs=[base/'analysis_prior_grid/RBM3_MCG_01/results/RBM3_MCG_01/measurements/measure_41b160',
            base/'analysis_new/RBM3_MCG_05/results/RBM3_MCG_05/measurements/measure_fa373c']
    rows=[]
    for index,source in enumerate(inputs):
        original,data,movie=recording(source)
        movie=json.loads(json.dumps(movie))
        run=root/f'run-{index}'
        folder=run/'measure'/movie['stem'];folder.mkdir(parents=True)
        path=table_path(original,movie,'cell_frame')
        shutil.copyfile(path,folder/'cell_frame.csv')
        for role in ['raw','labels']:
            movie['spec'][role]=str(declared_source(original,movie,role))
        movie['tables']={'cell_frame':{'path':'cell_frame.csv','folder':movie['stem']}}
        (run/'manifest.json').write_text(json.dumps({'movies':[movie]}),encoding='utf-8')
        rows.append(dict(run=str(run),stem=movie['stem'],identity=14 if index==0 else 1,
                         frame=42 if index==0 else 56,
                         table_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         source_sha256=movie['spec'].get('sha256',{})))
    save_workspace(rows[0]['run'],add_runs(workspace(rows[0]['run']),[rows[1]['run']]))
    (root/'corpus.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--fixture',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--label',default='baseline')
    parser.add_argument('--url',default='http://127.0.0.1:62862')
    parser.add_argument('--cdp',default='http://127.0.0.1:9334')
    parser.add_argument('--timeout',type=float,default=120)
    parser.add_argument('--check-bounded-media',action='store_true')
    parser.add_argument('--max-switch-seconds',type=float)
    parser.add_argument('--baseline',type=Path)
    args=parser.parse_args()
    root=args.fixture or Path(tempfile.mkdtemp(prefix='pymicroglia-media-performance-'))
    rows=json.loads((root/'corpus.json').read_text()) if (root/'corpus.json').exists() else fixture(root)
    chrome=Chrome(args.cdp)
    context=chrome.call('Target.createBrowserContext',session=False)['browserContextId']
    samples=[]
    result=dict(label=args.label,fixture=str(root),corpus=rows,python=platform.python_version(),
                platform=platform.platform(),samples=samples)
    try:
        target=chrome.call('Target.createTarget',dict(url='about:blank',browserContextId=context),session=False)['targetId']
        chrome.session=chrome.call('Target.attachToTarget',dict(targetId=target,flatten=True),session=False)['sessionId']
        chrome.call('Page.enable');chrome.call('Runtime.enable')
        chrome.call('Emulation.setFocusEmulationEnabled',dict(enabled=True))
        chrome.call('Page.bringToFront')
        chrome.call('Emulation.setDeviceMetricsOverride',dict(width=1440,height=1000,deviceScaleFactor=1,mobile=False))
        chrome.call('Page.addScriptToEvaluateOnNewDocument',dict(source="""
          if(location.protocol==='http:'){
            localStorage.setItem('pm-workspace','micro');localStorage.setItem('pm-page','follow');
            window.mediaRequests=[];const originalFetch=window.fetch.bind(window);
            window.fetch=async(...args)=>{
              const row={path:String(args[0]),start:performance.now()};window.mediaRequests.push(row);
              try{const response=await originalFetch(...args);row.seconds=(performance.now()-row.start)/1000;row.status=response.status;return response;}
              catch(error){row.error=error.name;throw error;}
            };
          }
        """))
        chrome.call('Page.navigate',dict(url=args.url+'/?'+urlencode({'run':rows[0]['run']})))
        chrome.wait("!!window.pyMicrogliaExplorer?.data&&!!window.pyMicrogliaExplorer?.groupComparison?.design",timeout=args.timeout)
        for index in [0,1,0]:
            source=rows[index];started=time.monotonic()
            chrome.js("window.mediaRequests=[]")
            if index!=0 or samples:
                value=json.dumps([source['run'],source['stem']],separators=(',',':'))
                chrome.js(f"(()=>{{const s=document.getElementById('xp-recording-select');s.value={json.dumps(value)};s.dispatchEvent(new Event('change',{{bubbles:true}}));}})()")
                chrome.wait(f"window.pyMicrogliaExplorer.data.run==={json.dumps(source['run'])}",timeout=args.timeout)
            metadata=time.monotonic()-started
            chrome.js(f"window.pyMicrogliaExplorer.select({json.dumps({'identity':source['identity'],'frame':source['frame']})})")
            chrome.wait(f"(()=>{{const e=window.pyMicrogliaExplorer;return !!e.image&&e.imageMeta?.frame_index==={source['frame']}&&e.changeCache.has({source['frame']})}})()",timeout=args.timeout)
            usable=time.monotonic()-started
            chrome.wait("!!window.pyMicrogliaExplorer.history",timeout=args.timeout)
            chrome.wait("!window.pyMicrogliaExplorer.data.preview.pending",timeout=args.timeout)
            assert not chrome.js("window.pyMicrogliaExplorer.imageError||window.pyMicrogliaExplorer.historyError")
            cache=chrome.js("(()=>{const e=window.pyMicrogliaExplorer;return {frame:e.state.frame,sourceFrame:e.imageMeta.source_frame_index,image:e.imageMeta.image,bounds:e.imageMeta.image_bounds,contrast:e.imageMeta.contrast,change:e.changeCache.get(e.state.frame).payload.change,cells:e.data.total_cells}})()")
            next_frame=source['frame']+1;step_start=time.monotonic()
            chrome.js(f"window.pyMicrogliaExplorer.select({{frame:{next_frame}}})")
            chrome.wait(f"(()=>{{const e=window.pyMicrogliaExplorer;return !!e.image&&e.imageMeta?.frame_index==={next_frame}&&e.changeCache.has({next_frame})}})()",timeout=args.timeout)
            step=time.monotonic()-step_start
            requests=chrome.js("window.mediaRequests")
            cache_state=chrome.js("(()=>{const e=window.pyMicrogliaExplorer;return {nearby_frames:e.imageQueue.length,workers:e.imageWorkers,cell_bytes:e.cellImageBytes,cell_frames:e.cellImageCache.size}})()")
            if args.check_bounded_media:
                assert not any('/cell-images?' in row['path'] for row in requests), 'Automatic whole-recording preload'
                assert cache_state['nearby_frames']<=6 and cache_state['workers']<=1,cache_state
                assert cache_state['cell_bytes']<=64*1024**2 or cache_state['cell_frames']<=1,cache_state
            if args.max_switch_seconds is not None and samples:
                assert usable<=args.max_switch_seconds, f'Recording switch took {usable:.3f}s'
            movie=json.loads((Path(source['run'])/'manifest.json').read_text())['movies'][0]
            with tifffile.TiffFile(movie['spec']['raw']) as tif:
                pixels=tif.series[0].pages[cache['sourceFrame']].asarray()
            if cache.get('bounds'):
                x0,y0,x1,y1=cache['bounds'];pixels=pixels[y0:y1,x0:x1]
            lo,hi=cache['contrast']
            expected=np.nan_to_num(np.clip((pixels.astype(float)-lo)*255/(hi-lo),0,255)).astype(np.uint8)
            shown=np.asarray(Image.open(BytesIO(base64.b64decode(cache['image'].split(',')[1]))))
            np.testing.assert_array_equal(shown,expected)
            samples.append(dict(kind='first-detail' if not samples else 'switch',stem=source['stem'],
                                metadata_seconds=metadata,usable_seconds=usable,step_seconds=step,
                                pixels_sha256=hashlib.sha256(cache['image'].encode()).hexdigest(),
                                change_sha256=hashlib.sha256(str(cache['change']).encode()).hexdigest(),
                                cells=cache['cells'],native_pixels_match=True,cache=cache_state,requests=requests))
            if args.baseline:
                baseline=json.loads(args.baseline.read_text())['samples'][len(samples)-1]
                assert samples[-1]['change_sha256']==baseline['change_sha256'], 'Changed pixels differ from baseline'
            print(json.dumps({k:v for k,v in samples[-1].items() if k!='requests'}),flush=True)
        result['errors']=chrome.errors
        assert not chrome.errors,chrome.errors
    except Exception as error:
        result['failure']=str(error)
        raise
    finally:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
        chrome.call('Target.disposeBrowserContext',dict(browserContextId=context),session=False)
        chrome.socket.close()
        print(json.dumps({'output':str(args.output),'fixture':str(root)}),flush=True)


if __name__=='__main__':main()
