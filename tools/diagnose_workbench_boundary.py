"""Inspect the real readiness state after a failed browser completion trial."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import time
from urllib.parse import urlencode
from urllib.request import urlopen
from playwright.sync_api import sync_playwright
from benchmark_workbench_boundaries import ROOT,HOOK,port,stop_server

root=ROOT/'development/speed-pass-20261011'
out=root/'complete/boundaries1'
chosen=port();url=f'http://127.0.0.1:{chosen}'
env={**os.environ,'PYMICROGLIA_STORE':str(out/'diagnostic-store'),'PYMICROGLIA_INDEX':str(out/'diagnostic-index')}
with (out/'diagnostic-server.log').open('w') as log:
    server=subprocess.Popen([shutil.which('python'),str(ROOT/'tools/benchmark_workbench_boundaries.py'),
        '--serve',str(chosen),'--baseline',str(root/'baseline')],env=env,stdout=log,stderr=log)
    try:
        deadline=time.monotonic()+60
        while True:
            try:
                with urlopen(url+'/api/config',timeout=1):break
            except Exception:
                if time.monotonic()>deadline:raise
                time.sleep(.2)
        with sync_playwright() as pw:
            browser=pw.chromium.launch(channel='chrome',headless=True,args=['--disable-gpu'])
            page=browser.new_page(viewport={'width':1440,'height':1000})
            page.add_init_script("localStorage.setItem('pm-workspace','micro');localStorage.setItem('pm-page','follow');")
            def instrument(route):
                response=route.fetch();route.fulfill(response=response,body=response.text()+HOOK)
            page.route('**/assets/exploration.js*',instrument)
            page.goto(url+'/?'+urlencode({'run':str(out/'fixture')}))
            page.wait_for_function('window.pyMicrogliaExplorer?.image && !window.pyMicrogliaExplorer.mediaBusy',timeout=60000)
            page.wait_for_timeout(5000)
            state=page.evaluate("""() => {const e=window.pyMicrogliaExplorer,c=e.groupComparison?.states.get('distribution');
                const t=e.tissueCache.get(e.tissueKey());
                return {state:e.state,preview:e.data.preview,gridCells:e.gridRows.size,identities:e.data.identities.length,
                  comparison:{design:!!e.groupComparison.design,preview:!!c?.preview,image:!!c?.image,key:c?.key,
                    caption:e.groupComparison.element('caption','distribution').textContent},
                  tissue:{layer:e.comparisons.value('map-layer','tissue-map'),entry:!!t,image:!!t?.image,error:t?.error,key:e.tissueKey(),keys:[...e.tissueCache.keys()]},
                  contactBusy:!!e.contactBusy,previewContacts:e.previewContacts,contact:e.contactCache.has(e.state.frame),
                  peerIds:e.peerIdentities,peerExact:[...e.peerExact.keys()],peerRows:[...e.peerRows].map(([k,v])=>[k,[...v.keys()]]),
                  exactIdentity:e.exact.cell.identity,frame:e.frameValues?.frame?.frame_index,
                  change:e.changeCache.has(e.state.frame),rhythmsMessage:e.data.results?.rhythms?.message,
                  paints:window.benchmarkPaints};}""")
            (out/'readiness-diagnostic.json').write_text(json.dumps(state,indent=2))
            print(json.dumps(state,indent=2))
            browser.close()
    finally:stop_server(server)
