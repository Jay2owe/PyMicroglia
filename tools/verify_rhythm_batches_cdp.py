"""Exercise batch rhythm controls in an isolated, two-recording workspace."""
import csv
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time
from urllib.parse import urlencode

from verify_group_comparison_cdp import Chrome


def main():
    root = Path(tempfile.mkdtemp(prefix='pymicroglia-rhythm-batch-'))
    sources = []
    for number in range(2):
        run = root / str(number)
        folder = run / 'measure' / 'same_name'
        folder.mkdir(parents=True)
        with (folder / 'cell_frame.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['identity', 'frame_index', 'hours', 'integrated_density_mean3'])
            for identity in (1, 2):
                for frame in range(145):
                    writer.writerow([identity, frame, frame / 2,
                                     100 + 20 * math.cos(2 * math.pi * (frame / 2 - identity - number) / 24)])
        (run / 'manifest.json').write_text(json.dumps({'movies': [{'stem': 'same_name'}]}))
        sources.append(run)
    original = [hashlib.sha256((run / 'measure/same_name/cell_frame.csv').read_bytes()).hexdigest()
                for run in sources]
    chrome = Chrome('http://127.0.0.1:9333')
    context = chrome.call('Target.createBrowserContext', session=False)['browserContextId']
    try:
        target = chrome.call('Target.createTarget', {'url': 'about:blank', 'browserContextId': context}, session=False)['targetId']
        chrome.session = chrome.call('Target.attachToTarget', {'targetId': target, 'flatten': True}, session=False)['sessionId']
        chrome.call('Page.enable'); chrome.call('Runtime.enable')
        chrome.call('Emulation.setFocusEmulationEnabled', {'enabled': True})
        chrome.call('Page.bringToFront')
        chrome.call('Emulation.setDeviceMetricsOverride', {'width': 1440, 'height': 1000, 'deviceScaleFactor': 1, 'mobile': False})
        chrome.call('Page.addScriptToEvaluateOnNewDocument', {'source': "localStorage.setItem('pm-workspace','micro');localStorage.setItem('pm-page','follow');"})
        start = time.monotonic()
        chrome.call('Page.navigate', {'url': 'http://127.0.0.1:62862/?' + urlencode({'run': str(sources[0])})})
        chrome.wait("!!window.pyMicrogliaExplorer?.rhythmAnalyses?.rows[0]?.runAll && !!window.pyMicrogliaExplorer.groupComparison.design")
        controls_seconds = time.monotonic() - start
        print('Batch controls available', flush=True)
        chrome.js("(()=>{const e=window.pyMicrogliaExplorer;e.$('comparison-add-run').value=" + json.dumps(str(sources[1])) + ";e.$('comparison-add-recordings').click();})()")
        chrome.wait("window.pyMicrogliaExplorer.groupComparison.design.recordings.length===2 && !window.pyMicrogliaExplorer.groupComparison.addingRecordings")
        chrome.js("(()=>{const r=window.pyMicrogliaExplorer.rhythmAnalyses.rows[0];r.$('rhythm-recipe').value='lomb';r.$('rhythm-recipe').onchange();r.$('rhythm-detrend').value='none';r.$('rhythm-normalization').value='zscore';r.$('rhythm-prefilter').value='none';r.runAll.click();})()")
        chrome.wait("window.pyMicrogliaExplorer.rhythmAnalyses.batchReferences.length===1")
        batch = chrome.js("[...window.pyMicrogliaExplorer.rhythmAnalyses.batches.values()][0]")
        assert len(batch['recordings']) == 2
        assert batch['params']['options']['trace_normalization'] == 'zscore'
        assert batch['params']['options']['detrend'] == 'none'
        assert batch['params']['options']['trace_filter'] == {'method': 'none'}
        # Selection may change while the batch runs; all sources stay loaded.
        chrome.call('Runtime.evaluate', {'expression': "(()=>{const e=window.pyMicrogliaExplorer;e.switchRecording(e.groupComparison.design.recordings[1]);})()", 'awaitPromise': False})
        chrome.wait("[...window.pyMicrogliaExplorer.rhythmAnalyses.batches.values()][0]?.state==='completed'", timeout=120)
        chrome.wait("window.pyMicrogliaExplorer.rhythmAnalyses.rows.some(row=>row.result?.resolved_options.trace_normalization==='zscore' && row.shown.checked)")
        chrome.wait("window.pyMicrogliaExplorer.groupComparison.design.recordings.every(row=>!!row.analysis_ids?.integrated_density_mean3)")
        completed = chrome.js("[...window.pyMicrogliaExplorer.rhythmAnalyses.batches.values()][0]")
        assert completed['progress']['completed'] == 2
        assert chrome.js("window.pyMicrogliaExplorer.$('recording-select').options.length") == 2
        assert chrome.js("[...document.querySelectorAll('.xp-rhythm-batches li')].every(row=>row.textContent.includes('Ready'))")
        chrome.call('Page.reload', {'ignoreCache': True})
        chrome.wait("window.pyMicrogliaExplorer?.groupComparison?.design?.recordings.length===2 && window.pyMicrogliaExplorer.rhythmAnalyses.batches.size===1")
        chrome.wait("window.pyMicrogliaExplorer.rhythmAnalyses.rows.some(row=>row.result?.analysis_id===" + json.dumps(completed['recordings'][1]['analysis_id']) + " && row.shown.checked)")
        chrome.js("document.querySelector('button[data-page=compare]').click()")
        chrome.wait("window.pyMicrogliaExplorer.groupComparison.states.get('distribution')?.preview?.counts.cells===4")
        counts = chrome.js("window.pyMicrogliaExplorer.groupComparison.states.get('distribution').preview.counts")
        chrome.js("document.querySelector('button[data-page=follow]').click()")
        chrome.wait("window.pyMicrogliaExplorer.$('cell-image').width>1")
        # Every generated row uses the same batch button helper.
        chrome.js("window.pyMicrogliaExplorer.rhythmAnalyses.addButton.click()")
        assert chrome.js("window.pyMicrogliaExplorer.rhythmAnalyses.rows.every(row=>row.runAll.textContent==='Run all recordings')")
        # Check controls remain contained in a narrow viewport in each theme.
        chrome.call('Emulation.setDeviceMetricsOverride', {'width': 800, 'height': 1000, 'deviceScaleFactor': 1, 'mobile': False})
        for theme in ('light', 'dark'):
            chrome.js(f"document.documentElement.dataset.pmTheme={json.dumps(theme)}")
            assert chrome.js("[...document.querySelectorAll('.xp-rhythm-run-controls button')].every(button=>{const r=button.getBoundingClientRect();return r.right<=innerWidth && r.left>=0;})")
        for run, digest in zip(sources, original):
            assert hashlib.sha256((run / 'measure/same_name/cell_frame.csv').read_bytes()).hexdigest() == digest
        assert not chrome.errors, chrome.errors
        report = {'fixture': str(root), 'controls_seconds': controls_seconds,
                  'batch_id': completed['id'], 'recordings': completed['recordings'],
                  'counts': counts, 'restored': True, 'themes_and_narrow_controls': True,
                  'source_tables_unchanged': True}
        (root / 'verification.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
    finally:
        chrome.call('Target.disposeBrowserContext', {'browserContextId': context}, session=False)
        chrome.socket.close()


if __name__ == '__main__':
    main()
