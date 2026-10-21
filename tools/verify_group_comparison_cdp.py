"""Check native comparison cards with Chrome's protocol, in an isolated context."""
import base64
import csv
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time
import numpy as np
import tifffile
from urllib.parse import urlencode
from urllib.request import urlopen
from websockets.sync.client import connect
from verify_group_comparison_ui import fixture


class Chrome:
    def __init__(self,endpoint):
        self.socket=connect(json.load(urlopen(endpoint+'/json/version'))['webSocketDebuggerUrl'],open_timeout=10)
        self.number=0
        self.errors=[]
        self.session=None

    def call(self,method,params=None,session=True):
        self.number+=1
        request=dict(id=self.number,method=method,params=params or {})
        if session and self.session:request['sessionId']=self.session
        self.socket.send(json.dumps(request))
        while True:
            response=json.loads(self.socket.recv(timeout=30))
            if response.get('sessionId')==self.session and response.get('method')=='Runtime.exceptionThrown':
                self.errors.append(response['params']['exceptionDetails'])
            if response.get('id')==self.number:
                if 'error' in response:raise RuntimeError(response['error'])
                return response.get('result',{})

    def js(self,expression):
        result=self.call('Runtime.evaluate',dict(expression=expression,returnByValue=True,awaitPromise=True))
        if result.get('exceptionDetails'):raise RuntimeError(result['exceptionDetails'])
        return result['result'].get('value')

    def wait(self,expression,timeout=60):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            if self.js(expression):return
            time.sleep(.1)
        raise TimeoutError(expression)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cdp',default='http://127.0.0.1:9333')
    args=parser.parse_args()
    root=Path(tempfile.mkdtemp(prefix='pymicroglia-comparison-cdp-'))
    fixture(root)
    assert urlopen('http://127.0.0.1:62862/api/config',timeout=10).status==200
    before={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in root.glob('measure/*/cell_frame.csv')}
    chrome=Chrome(args.cdp)
    context=chrome.call('Target.createBrowserContext',session=False)['browserContextId']
    try:
        target=chrome.call('Target.createTarget',dict(url='about:blank',browserContextId=context),session=False)['targetId']
        chrome.session=chrome.call('Target.attachToTarget',dict(targetId=target,flatten=True),session=False)['sessionId']
        chrome.call('Page.enable');chrome.call('Runtime.enable')
        chrome.call('Emulation.setFocusEmulationEnabled',dict(enabled=True))
        chrome.call('Page.bringToFront')
        chrome.call('Emulation.setDeviceMetricsOverride',dict(width=1440,height=1000,deviceScaleFactor=1,mobile=False))
        chrome.call('Page.addScriptToEvaluateOnNewDocument',dict(source="if(location.origin==='http://127.0.0.1:62862'){localStorage.setItem('pm-workspace','micro');localStorage.setItem('pm-page','compare');window.comparisonBulkMediaRequests=0;const mediaFetch=window.fetch.bind(window);window.fetch=(...args)=>{if(String(args[0]).split('?')[0]==='/api/exploration/cell-images')window.comparisonBulkMediaRequests++;return mediaFetch(...args);};}"))
        started=time.monotonic()
        chrome.call('Page.navigate',dict(url='http://127.0.0.1:62862/?'+urlencode({'run':str(root)})))
        print('Opened isolated comparison fixture',flush=True)
        chrome.wait("!!window.pyMicrogliaExplorer?.groupComparison?.states.get('distribution')?.preview")
        first=time.monotonic()-started
        print('Native plot loaded',flush=True)
        counts=chrome.js("window.pyMicrogliaExplorer.groupComparison.states.get('distribution').preview.counts")
        assert counts['cells']==48 and counts['samples']==6,counts
        for name in ['Edit plot','Settings','Export','Exclusions']:
            assert chrome.js(f"[...document.querySelectorAll('#xp-distribution-panel button')].some(button=>button.textContent==={json.dumps(name)})"),name
        chrome.js("document.querySelector('#xp-distribution-comparisons .xp-comparison-toolbar button').click()")
        chrome.wait("!!window.pyMicrogliaExplorer.groupComparison.states.get('distribution-2')?.preview")
        def change(identifier,value):
            chrome.js(f"(()=>{{const input=document.getElementById({json.dumps(identifier)});input.value={json.dumps(value)};input.dispatchEvent(new Event('change',{{bubbles:true}}));}})()")
        change('xp-distribution-view-distribution-2','cells')
        change('xp-distribution-columns','2')
        chrome.wait("window.pyMicrogliaExplorer.groupComparison.states.get('distribution-2').preview?.hits.length===48")
        assert chrome.js("document.getElementById('xp-distribution-view').value")=='samples'
        assert chrome.js("document.getElementById('xp-distribution-columns').getBoundingClientRect().top<document.getElementById('xp-distribution-panel').getBoundingClientRect().top")
        def input_text(identifier, value):
            chrome.js(f"(()=>{{const input=document.getElementById({json.dumps(identifier)});input.value={json.dumps(value)};input.dispatchEvent(new Event('input',{{bubbles:true}}));}})()")
        assert chrome.js("document.getElementById('xp-comparison-design').open")
        input_text('xp-comparison-cell-ids','1, 3-4, 999')
        input_text('xp-comparison-condition','Responder')
        chrome.wait("window.pyMicrogliaExplorer.groupComparison.selectionPreview?.cells===3")
        assert chrome.js("window.pyMicrogliaExplorer.groupComparison.selectionPreview.targets[0].missing_ids")==[999]
        assert not chrome.js("Object.keys(window.pyMicrogliaExplorer.groupComparison.design.recordings[0].cell_conditions).length")
        chrome.js("document.getElementById('xp-comparison-assign').click()")
        chrome.wait("Object.keys(window.pyMicrogliaExplorer.groupComparison.design.recordings[0].cell_conditions).length===3")
        chrome.wait("window.pyMicrogliaExplorer.groupComparison.states.get('distribution-2').preview?.groups.length===3")
        assert chrome.js("window.pyMicrogliaExplorer.groupComparison.design.recordings.slice(1).every(row=>!Object.keys(row.cell_conditions).length)")
        chrome.wait("document.getElementById('xp-comparison-design-status').textContent==='Saved'")
        chrome.call('Page.reload')
        chrome.wait("window.pyMicrogliaExplorer?.groupComparison?.states.get('distribution-2')?.preview?.hits.length===48")
        assert chrome.js("Object.keys(window.pyMicrogliaExplorer.groupComparison.design.recordings[0].cell_conditions).length")==3
        change('xp-comparison-target','pattern')
        input_text('xp-comparison-pattern','[')
        input_text('xp-comparison-condition','Pattern group')
        chrome.wait("document.getElementById('xp-comparison-selection-status').textContent.includes('Invalid regular expression')")
        assert chrome.js("document.getElementById('xp-comparison-assign').disabled")
        input_text('xp-comparison-pattern','^COMPARISON_[01]$')
        chrome.wait("window.pyMicrogliaExplorer.groupComparison.selectionPreview?.recordings===2")
        assert chrome.js("window.pyMicrogliaExplorer.groupComparison.selectionPreview.cells")==16
        chrome.js("document.getElementById('xp-comparison-assign').click()")
        chrome.wait("window.pyMicrogliaExplorer.groupComparison.design.recordings[0].condition===window.pyMicrogliaExplorer.groupComparison.design.recordings[1].condition&&Object.keys(window.pyMicrogliaExplorer.groupComparison.design.recordings[0].cell_conditions).length===0")
        chrome.wait("document.getElementById('xp-comparison-design-status').textContent==='Saved'")
        change('xp-comparison-target','recordings')
        chrome.js("document.getElementById('xp-comparison-clear-selection').click();document.querySelectorAll('#xp-comparison-recordings input[data-field=selected]')[1].click()")
        input_text('xp-comparison-condition','Control')
        chrome.wait("window.pyMicrogliaExplorer.groupComparison.selectionPreview?.cells===8")
        chrome.js("document.getElementById('xp-comparison-assign').click()")
        chrome.wait("window.pyMicrogliaExplorer.groupComparison.design.recordings[1].condition===window.pyMicrogliaExplorer.groupComparison.design.recordings[2].condition")
        input_text('xp-comparison-condition','Recovery')
        chrome.js("document.getElementById('xp-comparison-clear-selection').click();document.querySelector('#xp-comparison-recordings input[data-field=selected]').click()")
        chrome.wait("window.pyMicrogliaExplorer.groupComparison.selectionPreview?.cells===8")
        chrome.js("document.getElementById('xp-comparison-assign').click()")
        chrome.wait("window.pyMicrogliaExplorer.groupComparison.states.get('distribution').preview?.groups.length===3")
        chrome.wait("document.getElementById('xp-comparison-design-status').textContent==='Saved'")
        chrome.call('Page.reload')
        chrome.wait("window.pyMicrogliaExplorer?.groupComparison?.states.get('distribution-2')?.preview?.hits.length===48")
        assert chrome.js("document.getElementById('xp-distribution-columns').value")=='2'
        print('Assignments and independent cards restored',flush=True)
        chrome.js("(()=>{const e=window.pyMicrogliaExplorer;e.groupComparison.design.recordings[0].condition=e.groupComparison.design.recordings[1].condition;e.groupComparison.renderAssignments();e.groupComparison.changed();})()")
        chrome.wait("window.pyMicrogliaExplorer.groupComparison.states.get('distribution').preview?.groups.length===2")
        chrome.js("document.getElementById('xp-distribution-run').click()")
        chrome.wait("!!window.pyMicrogliaExplorer.groupComparison.states.get('distribution').result?.statistics.length")
        stats=chrome.js("window.pyMicrogliaExplorer.groupComparison.states.get('distribution').result.statistics[0]")
        assert stats['n_a']==stats['n_b']==3
        completed=chrome.js("window.pyMicrogliaExplorer.groupComparison.states.get('distribution').resultId")
        change('xp-distribution-test','welch_t')
        chrome.wait("document.getElementById('xp-distribution-result-status').textContent.includes('previous completed')")
        assert chrome.js("window.pyMicrogliaExplorer.groupComparison.states.get('distribution').resultId")==completed
        change('xp-distribution-test','mannwhitney')
        for view in ['effects','box','violin','histogram','time_course','samples']:
            change('xp-distribution-view',view)
            chrome.wait(f"window.pyMicrogliaExplorer.groupComparison.states.get('distribution').displayedRequest?.options.view==={json.dumps(view)}")
        print('All descriptive views and explicit statistics passed',flush=True)
        chrome.js("(()=>{const e=window.pyMicrogliaExplorer;e.rhythmAnalyses.plotFilters.distribution={integrated_density:{enabled:true,filter:{applied:{expression:{field:'observed_frames',cmp:'lt',value:5},mode:'exclude',unknown:'keep'}}}};e.groupComparison.draw('distribution');e.saveSession();})()")
        chrome.wait("window.pyMicrogliaExplorer.groupComparison.states.get('distribution').preview?.counts.cells===0")
        assert chrome.js("window.pyMicrogliaExplorer.groupComparison.states.get('distribution-2').preview.counts.cells")==48
        chrome.js("(()=>{const e=window.pyMicrogliaExplorer;delete e.rhythmAnalyses.plotFilters.distribution;e.styles.distribution={...e.styles.distribution,height:240};e.saveSession();e.groupComparison.draw('distribution');})()")
        chrome.wait("document.getElementById('xp-comparison-design-status').textContent==='Saved'")
        saved=json.loads((root/'AI_Exports/comparisons/workspace.json').read_text())
        assert saved['plots']['distribution']['appearance']['height']==240
        chrome.call('Page.reload')
        chrome.wait("window.pyMicrogliaExplorer?.groupComparison?.states.get('distribution')?.preview?.height===240")
        chrome.wait("!!window.pyMicrogliaExplorer.groupComparison.states.get('distribution-2')?.preview?.hits.length")
        target=chrome.js("(()=>{const e=window.pyMicrogliaExplorer;return e.groupComparison.states.get('distribution-2').preview.hits.find(hit=>hit.stem!==e.data.stem);})()")
        chrome.js("""(()=>{const e=window.pyMicrogliaExplorer,c=document.getElementById('xp-distribution-2'),s=e.groupComparison.states.get('distribution-2'),h=s.preview.hits.find(hit=>hit.stem!==e.data.stem),r=c.getBoundingClientRect();c.dispatchEvent(new MouseEvent('click',{bubbles:true,clientX:r.left+h.x*r.width/s.preview.width,clientY:r.top+h.y*r.height/s.preview.height}));})()""")
        chrome.wait(f"window.pyMicrogliaExplorer.data.stem==={json.dumps(target['stem'])}")
        chrome.wait("window.pyMicrogliaExplorer.groupComparison.states.get('distribution-2')?.preview?.counts.cells===48")
        assert chrome.js("window.pyMicrogliaExplorer.groupComparison.design.recordings.length")==6
        chrome.js("window.pyMicrogliaExplorer.rhythmAnalyses.refreshPlotLabels()")
        count_label=chrome.js("document.querySelector('#xp-distribution-panel .xp-plot-cohort').textContent")
        assert count_label=='Main: 48/48',count_label
        assert chrome.js("getComputedStyle(document.getElementById('xp-distribution-caption')).display")!='none'
        assert chrome.js("getComputedStyle(document.getElementById('xp-distribution-caption').parentElement).display")!='none'
        print('Shared settings, plot exclusions and source click passed',flush=True)
        for theme,width in [('dark',1440),('light',800)]:
            chrome.call('Emulation.setDeviceMetricsOverride',dict(width=width,height=1000,deviceScaleFactor=1,mobile=False))
            chrome.js(f"document.documentElement.dataset.pmTheme={json.dumps(theme)};document.getElementById('xp-distribution-comparisons').scrollIntoView()")
            chrome.wait('document.documentElement.scrollWidth<=innerWidth')
            chrome.wait("[...window.pyMicrogliaExplorer.groupComparison.states].every(([id,state])=>!state.panel.isConnected||state.preview&&!state.panel.querySelector('[id*=distribution-caption]')?.textContent.includes('Drawing'))")
            screenshot=chrome.call('Page.captureScreenshot',dict(format='png',captureBeyondViewport=False))
            (root/f'comparison-{theme}-{width}.png').write_bytes(base64.b64decode(screenshot['data']))
        extra_runs=[]
        metadata=json.loads((root/'manifest.json').read_text())
        for index in range(2):
            extra=root/f'additional-run-{index}'
            folder=extra/'measure'/metadata['movies'][0]['stem']
            folder.mkdir(parents=True)
            (folder/'cell_frame.csv').write_bytes((root/'measure'/metadata['movies'][0]['stem']/'cell_frame.csv').read_bytes())
            if index:
                table=folder/'cell_frame.csv'
                with table.open(newline='') as stream:
                    reader=csv.DictReader(stream);fields=reader.fieldnames;rows=list(reader)
                for row in rows:
                    row['integrated_density']=str(float(row['integrated_density'])+10000)
                with table.open('w',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
            movie=json.loads(json.dumps(metadata['movies'][0]))
            labels=np.zeros((4,32,32),dtype=np.uint16)
            for identity in range(1,9):
                y,x=4+(identity-1)//4*12,4+(identity-1)%4*6
                labels[:,y:y+3,x:x+3]=identity
            raw=np.arange(4*32*32,dtype=np.uint16).reshape(4,32,32)+index*3000
            spec=movie['spec'];spec['sha256']={}
            for role,pixels in [('raw',raw),('labels',labels)]:
                path=extra/(role+'.tif')
                tifffile.imwrite(path,pixels,photometric='minisblack')
                spec[role]=str(root/'previous-location'/extra.name/path.name)
                spec['sha256'][role]=hashlib.sha256(path.read_bytes()).hexdigest()
            movie['display']={'contrast':[0,8000]}
            (extra/'manifest.json').write_text(json.dumps({'run':{'folder':str(root/'previous-location'/extra.name)},'movies':[movie]}))
            extra_runs.append(str(extra))
        assert chrome.js("!document.getElementById('xp-comparison-add-run').closest('#xp-comparison-assignment-advanced')")
        input_text('xp-comparison-add-run','\n'.join(extra_runs+[str(root)]))
        chrome.js("document.getElementById('xp-comparison-add-recordings').click()")
        chrome.wait("window.pyMicrogliaExplorer.groupComparison.states.get('distribution').preview?.counts.recordings===8")
        assert chrome.js("window.pyMicrogliaExplorer.groupComparison.states.get('distribution').preview.counts.cells")==64
        assert chrome.js("document.getElementById('xp-comparison-source-count').textContent")== '8 recordings loaded together'
        chrome.wait("document.getElementById('xp-comparison-design-status').textContent==='Saved'")
        chrome.call('Page.reload')
        chrome.wait("window.pyMicrogliaExplorer?.groupComparison?.states.get('distribution-2')?.preview?.counts.recordings===8")
        assert chrome.js("window.pyMicrogliaExplorer.groupComparison.states.get('distribution-2').preview.counts.cells")==64
        assert chrome.js("window.pyMicrogliaExplorer.groupComparison.design.recordings.filter(row=>row.stem==='comparison_0').length")==3
        print('Batch loading multiple saved runs in one window restored after reload',flush=True)
        options=chrome.js("[...document.getElementById('xp-recording-select').options].map(option=>({value:option.value,label:option.textContent}))")
        assert len(options)==len({option['value'] for option in options})==8
        duplicate_labels=[option['label'] for option in options if json.loads(option['value'])[1]=='comparison_0']
        assert len(set(duplicate_labels))==3,duplicate_labels
        chrome.js("(()=>{window.holdOverview=true;window.heldOverviews=[];const fetchNow=window.fetch.bind(window);window.fetch=async(...args)=>{if(window.holdOverview&&String(args[0]).startsWith('/api/exploration/overview?'))await new Promise(resolve=>window.heldOverviews.push(resolve));return fetchNow(...args);};})()")
        for extra in extra_runs:
            change('xp-recording-select',json.dumps([extra,'comparison_0'],separators=(',',':')))
            chrome.wait(f"window.pyMicrogliaExplorer.data.run==={json.dumps(extra)}&&window.pyMicrogliaExplorer.data.stem==='comparison_0'")
            chrome.wait("window.pyMicrogliaExplorer.groupComparison.states.get('distribution-2')?.preview?.counts.cells===64")
            assert chrome.js("window.pyMicrogliaExplorer.groupComparison.design.recordings.length")==8
            assert chrome.js("window.pyMicrogliaExplorer.groupComparison.run")==str(root)
            assert chrome.js("window.pyMicrogliaExplorer.data.identities.length")==8
            assert chrome.js("document.getElementById('xp-recording-select').options.length")==8
            assert json.loads(chrome.js("document.getElementById('xp-recording-select').value"))==[extra,'comparison_0']
            chrome.wait("!!window.pyMicrogliaExplorer.image&&!!window.pyMicrogliaExplorer.history&&!!window.pyMicrogliaExplorer.cellImageCache.get(0)?.image")
            assert chrome.js("window.pyMicrogliaExplorer.data.preview.pending"), 'Media must appear before delayed measurements'
            assert not chrome.js("window.pyMicrogliaExplorer.imageError||window.pyMicrogliaExplorer.historyError")
            shown_image=chrome.js("window.pyMicrogliaExplorer.cellImageCache.get(0).payload.image")
            if extra==extra_runs[0]:first_source_image=shown_image
            else:assert shown_image!=first_source_image, 'Recording switch must replace cell pixels'
            chrome.js("window.pyMicrogliaExplorer.select({frame:1})")
            chrome.wait("!!window.pyMicrogliaExplorer.cellImageCache.get(1)?.image&&window.pyMicrogliaExplorer.imageMeta?.frame_index===1")
            assert chrome.js("window.pyMicrogliaExplorer.cellImageCache.get(1).payload.image")!=shown_image
            chrome.wait("window.pyMicrogliaExplorer.allPositions().length===8")
            cropped_positions=chrome.js("window.pyMicrogliaExplorer.allPositions()")
        change('xp-image-mode','field')
        chrome.wait("!!window.pyMicrogliaExplorer.image&&!window.pyMicrogliaExplorer.frameData.partial_layers&&window.pyMicrogliaExplorer.changeCache.get(1)?.payload.identity===window.pyMicrogliaExplorer.state.identity")
        assert chrome.js("window.pyMicrogliaExplorer.allPositions()")==cropped_positions
        change('xp-cell-select','2')
        chrome.wait("window.pyMicrogliaExplorer.history?.identity===2&&!!window.pyMicrogliaExplorer.image&&window.pyMicrogliaExplorer.changeCache.get(1)?.payload.identity===2")
        change('xp-image-mode','cell')
        chrome.wait("window.pyMicrogliaExplorer.imageMeta?.partial_layers===true&&window.pyMicrogliaExplorer.imageMeta?.layers.accepted[0]?.identity===2")
        chrome.js("window.holdOverview=false;window.heldOverviews.splice(0).forEach(resolve=>resolve())")
        chrome.wait("!window.pyMicrogliaExplorer.data.preview.pending&&window.pyMicrogliaExplorer.data.cells.length===8")
        assert chrome.js("window.pyMicrogliaExplorer.data.cells.every(c=>!!c.values.integrated_density)")
        assert chrome.js("window.pyMicrogliaExplorer.data.cells[0].values.integrated_density[0]>=10000"), 'Delayed overview must belong to current source'
        assert chrome.js("window.comparisonBulkMediaRequests")==0, 'Interactive media must not preload whole recording strips'
        print('Current-frame images, changed pixels, full-field mode and cell switching passed without bulk strip loads',flush=True)
        chrome.wait("document.getElementById('xp-comparison-design-status').textContent==='Saved'")
        chrome.call('Page.addScriptToEvaluateOnNewDocument',dict(source="""
          const sourceFetch=window.fetch.bind(window);
          window.fetch=async(...args)=>{
            if(String(args[0]).endsWith('/api/exploration/luts')){
              await new Promise(resolve=>setTimeout(resolve,2500));
              window.comparisonStartedBeforeColours=!!window.pyMicrogliaExplorer?.pending;
            }
            if(String(args[0]).endsWith('/api/exploration/comparison-preview'))await new Promise(resolve=>setTimeout(resolve,1000));
            return sourceFetch(...args);
          };
        """))
        chrome.call('Page.reload')
        chrome.wait(f"window.pyMicrogliaExplorer?.data?.run==={json.dumps(extra_runs[-1])}")
        chrome.wait("window.pyMicrogliaExplorer.groupComparison.states.get('distribution-2')?.preview?.counts.cells===64")
        assert chrome.js("window.comparisonStartedBeforeColours"), 'Reload must exercise queued recording initialization'
        assert chrome.js("window.pyMicrogliaExplorer.groupComparison.run")==str(root)
        assert chrome.js("document.getElementById('xp-recording-select').options.length")==8
        assert json.loads(chrome.js("document.getElementById('xp-recording-select').value"))==[extra_runs[-1],'comparison_0']
        print('Top recording dropdown preserves combined workspace, duplicate names and selected source after reload',flush=True)
        assert not chrome.errors,chrome.errors
    except Exception:
        print(json.dumps({'output':str(root),'errors':chrome.errors,'browser':chrome.js("(()=>{const e=window.pyMicrogliaExplorer;return {run:e?.data?.run,pending:e?.pending?.run,ready:e?.lutsReady,session:e?.session,stored:localStorage.getItem('pm-exploration-session-v1'),status:e?.$('selection-status')?.textContent}})()")}),flush=True)
        raise
    finally:
        chrome.call('Target.disposeBrowserContext',dict(browserContextId=context),session=False)
        chrome.socket.close()
    assert before=={path:hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in before}
    print(json.dumps({'output':str(root),'first_usable_seconds':first,'full_cells':48,'recording_points':6,'shared_controls':True,'persistent_assignments':True,'independent_plots':True,'native_tests':True,'plot_exclusions':True,'cross_recording_click':True,'narrow_layout':True,'errors':[]}))


if __name__=='__main__':main()
