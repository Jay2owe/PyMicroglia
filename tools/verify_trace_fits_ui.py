"""Verify optional saved model overlays on the real recording's linked traces."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from time import perf_counter
from urllib.parse import urlencode

from playwright.sync_api import sync_playwright


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--url',default='http://127.0.0.1:62855')
    args=parser.parse_args()
    run=args.run.resolve();source=run/'measure/95_A3/cell_frame.csv'
    before=hashlib.sha256(source.read_bytes()).hexdigest()
    output=Path(tempfile.mkdtemp(prefix='pymicroglia-trace-fits-'))
    errors=[];jobs=[];requests=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,executable_path=r'C:\Program Files\Google\Chrome\Application\chrome.exe')
        try:
            page=browser.new_page(viewport={'width':1440,'height':1000},device_scale_factor=2)
            page.on('pageerror',lambda error:errors.append(str(error)))
            page.on('request',lambda request:jobs.append(request.url) if request.method=='POST' and 'rhythm-jobs' in request.url else None)
            page.on('request',lambda request:requests.append(request.url) if 'rhythm-fits?' in request.url else None)
            page.add_init_script("localStorage.setItem('pm-workspace','micro');localStorage.setItem('pm-page','follow');")
            started=perf_counter()
            page.goto(args.url+'/?'+urlencode({'run':str(run)}))
            page.wait_for_function("window.pyMicrogliaExplorer?.rhythmSchema && window.pyMicrogliaExplorer.exact")
            first_usable=perf_counter()-started
            assert not page.locator('#xp-trace-fit').is_checked() and not requests
            page.locator('#xp-cell-select').select_option('6')
            page.wait_for_function("window.pyMicrogliaExplorer.exact?.cell.identity===6")
            page.evaluate("() => {window.fitStrokes=[];const stroke=CanvasRenderingContext2D.prototype.stroke;CanvasRenderingContext2D.prototype.stroke=function(...args){if(this.getLineDash().length)window.fitStrokes.push({canvas:this.canvas.id,identity:this.canvas.dataset.identity,dash:this.getLineDash()});return stroke.apply(this,args);};}")
            page.locator('#xp-trace-panel .settings:not(.xp-plot-settings)').evaluate('(details)=>details.open=true')
            started=perf_counter();page.locator('#xp-trace-fit').check()
            try:
                page.wait_for_function("document.getElementById('xp-trace').dataset.rhythmFit==='available'",timeout=30000)
            except Exception:
                print(page.evaluate("() => ({fit:document.getElementById('xp-trace').dataset.rhythmFit,caption:document.getElementById('xp-trace-caption').textContent,metric:window.pyMicrogliaExplorer.val('trace-metric'),aggregation:window.pyMicrogliaExplorer.val('trace-aggregation'),enabled:document.getElementById('xp-trace-fit').checked,cache:[...window.pyMicrogliaExplorer.fitCache?.values()||[]].map(v=>({state:v.state,message:v.message,count:v.rows.size}))})"))
                print('browser errors',errors,'fit requests',requests)
                page.screenshot(path=str(output/'failed.png'))
                raise
            first_fit=perf_counter()-started
            assert page.locator('#xp-trace-grid-fit').is_checked()
            assert page.evaluate("window.fitStrokes.some(row=>row.canvas==='xp-trace'&&row.dash.join(',')==='5,3')")
            page.wait_for_function("document.querySelector('#xp-peer-one-trace').dataset.rhythmFit==='available'")
            page.screenshot(path=str(output/'selected-fit.png'))
            page.locator('#xp-trace-grid-panel').scroll_into_view_if_needed()
            page.wait_for_function("document.querySelector('#xp-trace-grid canvas[data-rhythm-fit=available]')")
            assert page.evaluate("window.fitStrokes.some(row=>row.identity&&row.dash.join(',')==='5,3')")
            page.screenshot(path=str(output/'grid-fits.png'))
            request_count=len(requests);started=perf_counter()
            page.locator('#xp-trace-grid-fit').uncheck()
            assert not page.locator('#xp-trace-fit').is_checked()
            assert page.locator('#xp-trace').get_attribute('data-rhythm-fit')=='off'
            page.locator('#xp-trace-grid-fit').check()
            page.wait_for_function("document.getElementById('xp-trace').dataset.rhythmFit==='available'")
            cached_toggle=perf_counter()-started
            assert len(requests)==request_count
            # Different light aggregation uses its own saved model; an untested metric has none.
            page.locator('#xp-trace-grid-aggregation').select_option('mean')
            page.wait_for_function("document.getElementById('xp-trace').dataset.rhythmFit==='available' && [...window.pyMicrogliaExplorer.fitCache.keys()].some(key=>key.includes('signal_mean'))",timeout=30000)
            assert len(requests)==request_count+1
            page.locator('#xp-trace-grid-metric').select_option('signal_median')
            page.wait_for_function("document.getElementById('xp-trace').dataset.rhythmFit==='unavailable'")
            assert 'No saved' in page.locator('#xp-trace-caption').inner_text()
            page.locator('#xp-trace-grid-metric').select_option('light')
            page.locator('#xp-trace-grid-aggregation').select_option('total')
            page.reload()
            page.wait_for_function("window.pyMicrogliaExplorer?.exact && document.getElementById('xp-trace').dataset.rhythmFit==='available'")
            assert page.locator('#xp-trace-fit').is_checked() and page.locator('#xp-trace-grid-fit').is_checked()
            for theme,width in [('dark',1440),('light',800)]:
                page.set_viewport_size({'width':width,'height':1000})
                page.locator('#micro-theme').select_option(theme)
                assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
                page.screenshot(path=str(output/f'fit-{theme}-{width}.png'))
            assert not errors and not jobs,(errors,jobs)
        finally:
            browser.close()
    assert hashlib.sha256(source.read_bytes()).hexdigest()==before
    receipt={'first_usable_seconds':round(first_usable,3),'first_fit_seconds':round(first_fit,3),
             'cached_toggle_seconds':round(cached_toggle,3),'measurements_unchanged':True,
             'dashed_lines_drawn':True,'selected_peer_and_grid':True,'reload_preserved':True,
             'metric_specific_fits':True,'browser_errors':errors,'analysis_jobs':jobs,'screenshots':str(output)}
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt))


if __name__=='__main__':main()
