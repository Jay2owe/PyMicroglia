"""Real workbench first-preview/full-detail paint, isolated synthetic inputs."""
import argparse
import base64
from io import BytesIO
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from urllib.parse import urlencode
from urllib.request import urlopen


def serve(port, baseline):
    from pymicroglia.ui.server import create_app
    if baseline:
        from benchmark_viewer_speed import baseline as load_baseline
        module=load_baseline(baseline,'exploration_images')
        sys.modules['pymicroglia.ui.exploration_images']=module
    app=create_app()
    if baseline:
        from starlette.routing import Route
        from fastapi.responses import FileResponse
        async def script(request):
            return FileResponse(baseline/'exploration.js',media_type='text/javascript')
        app.router.routes.insert(0,Route('/assets/exploration.js',script))
    import uvicorn
    uvicorn.run(app,host='127.0.0.1',port=port,log_level='warning')


def port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));return sock.getsockname()[1]


def measure(folder):
    from playwright.sync_api import sync_playwright
    servers=[];logs=[];result={'trials':[],'errors':[]}
    try:
        urls={}
        for label in ['before','after']:
            chosen=port();urls[label]=f'http://127.0.0.1:{chosen}'
            log=(folder/(label+'-server.log')).open('w');logs.append(log)
            command=[shutil.which('python'),str(Path(__file__).resolve()),'--serve',str(chosen)]
            if label=='before':command+=['--baseline',str(folder/'baseline')]
            servers.append(subprocess.Popen(command,stdout=log,stderr=log))
        for label,url in urls.items():
            deadline=time.monotonic()+60
            while True:
                try:
                    with urlopen(url+'/api/config',timeout=1) as response:assert response.status==200
                    break
                except Exception:
                    if time.monotonic()>deadline:raise RuntimeError(label+' server startup timed out')
                    time.sleep(.2)
        with sync_playwright() as pw:
            browser=pw.chromium.launch(channel='chrome',headless=True,args=['--disable-gpu'])
            pages={};contexts=[]
            for label,url in urls.items():
                context=browser.new_context(viewport={'width':1440,'height':1000});contexts.append(context)
                page=context.new_page();pages[label]=page
                page.on('pageerror',lambda error:result['errors'].append(str(error)))
                page.add_init_script("localStorage.setItem('pm-workspace','micro');localStorage.setItem('pm-page','follow');")
                start=time.perf_counter()
                page.goto(url+'/?'+urlencode({'run':str(folder/'fixture')}))
                page.wait_for_function('window.pyMicrogliaExplorer?.image && window.pyMicrogliaExplorer?.exact && !window.pyMicrogliaExplorer.mediaBusy',timeout=60000)
                result[label+'_initial_selected_seconds']=time.perf_counter()-start
                # Disable only background look-ahead after initial load; foreground
                # values, source pixels, masks and exact change still use real routes.
                page.evaluate("""() => {const e=window.pyMicrogliaExplorer;
                  e.cancelMedia(); e.queueImages=()=>{}; e.requestPeers=()=>{}; e.requestPeerFrame=()=>{};
                  e.$('image-mode').value='field';
                  const draw=e.drawImage.bind(e);
                  e.drawImage=function(){draw();const trial=window.trial;if(!trial||this.imageMeta?.frame_index!==trial.frame||!this.image)return;
                    const preview=!!this.imageMeta.preview;
                    requestAnimationFrame(()=>{if(window.trial!==trial)return;
                      trial.first??=performance.now()-trial.start;
                      if(!preview)trial.full??=performance.now()-trial.start;
                    });};
                }""")
            for bandwidth in [None,524288]:
                for label,page in pages.items():
                    cdp=page.context.new_cdp_session(page)
                    cdp.send('Network.enable')
                    cdp.send('Network.emulateNetworkConditions',{'offline':False,'latency':0,
                        'downloadThroughput':bandwidth or -1,'uploadThroughput':-1})
                for iteration in range(3):
                    for label in (['before','after'] if iteration%2==0 else ['after','before']):
                        page=pages[label]
                        page.evaluate("""frame=>{const e=window.pyMicrogliaExplorer;e.cancelMedia();
                          e.frameCache.clear();e.imageCache.clear();e.imageCacheBytes=0;
                          e.cellImageCache.clear();e.changeCache.clear();e.cellImageBytes=0;
                          e.state.frame=frame;e.imageMeta=null;e.image=null;
                          window.trial={frame,start:performance.now()};e.loadSelectedFrame();
                        }""",10+iteration)
                        page.wait_for_function('window.trial?.full!=null',timeout=60000)
                        row={'label':label,'bandwidth_bytes_per_second':bandwidth,**page.evaluate('window.trial')}
                        result['trials'].append(row);print(json.dumps(row),flush=True)
            page=pages['after']
            # Hold full detail to inspect the actual temporary image, then change
            # a display control while it is pending. The upgrade keeps that control.
            held=[]
            page.route('**/api/exploration/frame?*',lambda route:held.append(route))
            page.evaluate("""() => {const e=window.pyMicrogliaExplorer;
                e.cancelMedia();e.state.frame=25;e.frameCache.clear();e.loadSelectedFrame();}""")
            page.wait_for_function('window.pyMicrogliaExplorer.imageMeta?.preview===true',timeout=30000)
            result['preview_dimensions']=page.evaluate('[window.pyMicrogliaExplorer.image.naturalWidth,window.pyMicrogliaExplorer.image.naturalHeight]')
            assert max(result['preview_dimensions'])<=256
            page.screenshot(path=str(folder/'viewer-preview.png'))
            page.evaluate("""() => {const e=window.pyMicrogliaExplorer;e.$('image-lut').value='red';e.drawImage();}""")
            for route in held:route.continue_()
            page.unroute('**/api/exploration/frame?*')
            page.wait_for_function('window.pyMicrogliaExplorer.imageMeta?.frame_index===25 && !window.pyMicrogliaExplorer.imageMeta.preview && !window.pyMicrogliaExplorer.mediaBusy',timeout=30000)
            assert page.evaluate("window.pyMicrogliaExplorer.val('image-lut')")=='red'
            result['lut_preserved']=True
            from PIL import Image
            import numpy as np
            import tifffile
            uri=page.evaluate('window.pyMicrogliaExplorer.image.src')
            shown=np.array(Image.open(BytesIO(base64.b64decode(uri.split(',')[1]))))
            with tifffile.TiffFile(folder/'fixture/raw.tif') as tif:raw=tif.pages[25].asarray()
            np.testing.assert_array_equal(shown,(raw.astype(float)*255/4000).astype(np.uint8))
            result['native_pixels_exact']=True
            for theme,width in [('light',1440),('dark',800)]:
                page.emulate_media(color_scheme=theme);page.set_viewport_size({'width':width,'height':900})
                page.screenshot(path=str(folder/(f'viewer-{theme}-{width}.png')))
                result[theme+'_overflow']=page.evaluate('document.documentElement.scrollWidth>innerWidth')
            result['final_image']=page.evaluate("""() => {const e=window.pyMicrogliaExplorer;return {
                width:e.image.naturalWidth,height:e.image.naturalHeight,metadata:e.imageMeta,
                fieldCache:e.frameCache.size};}""")
            result['final_image']['metadata'].pop('image',None)
            browser.close()
    except Exception as error:
        result['failure']=str(error);raise
    finally:
        for process in servers:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        for log in logs:log.close()
        (folder/'browser.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--serve',type=int)
    parser.add_argument('--baseline',type=Path);parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    if args.serve:serve(args.serve,args.baseline)
    else:measure(args.out.resolve())
