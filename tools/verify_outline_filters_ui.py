"""Real-browser acceptance for independent grey outline rules and native exports."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

from playwright.sync_api import sync_playwright


def capture(page, path):
    """Capture the actual app surface; this is a UI screenshot, not a figure export."""
    import base64
    session = page.context.new_cdp_session(page)
    session.send('Page.bringToFront')
    result = session.send('Page.captureScreenshot', {'format': 'png', 'fromSurface': True, 'captureBeyondViewport': False})
    path.write_bytes(base64.b64decode(result['data']))
    session.detach()


def verify(url, run, decisions, output):
    run, decisions, output = run.resolve(), decisions.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    inputs = list(run.glob('measure/*/cell_frame.csv'))
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    errors, jobs = [], []
    with sync_playwright() as playwright:
        browser = playwright.chromium.connect_over_cdp('http://127.0.0.1:9333')
        context = browser.contexts[0]
        page = next((p for p in context.pages if p.url.startswith(url)), None) or context.new_page()
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.on('request', lambda r: jobs.append(r.url) if r.method == 'POST' and ('jobs' in r.url) else None)
        page.goto(url)
        page.evaluate("localStorage.setItem('pm-workspace','micro');localStorage.setItem('pm-page','follow');localStorage.removeItem('pm-exploration-session-v1');")
        from urllib.parse import urlencode
        page.goto(url + '/?' + urlencode({'run': str(run)}))
        page.wait_for_function('window.pyMicrogliaExplorer?.sourceData', timeout=60000)
        page.evaluate('window.pyMicrogliaExplorer.select({frame:10})')
        page.wait_for_function('window.pyMicrogliaExplorer?.image', timeout=60000)
        original = page.evaluate('window.pyMicrogliaExplorer.sourceData.identities')
        page.locator('#xp-grey-filter-toggle').click()
        page.wait_for_function('window.pyMicrogliaExplorer.outlineFilters.schema')
        page.locator('#xp-grey-filter-panel input[aria-label="Import per-cell decisions"]').set_input_files(str(decisions))
        page.wait_for_function("window.pyMicrogliaExplorer.outlineFilters.tree.field==='review_passed'")
        start = perf_counter()
        page.locator('#xp-grey-filter-apply').click()
        page.wait_for_function('window.pyMicrogliaExplorer.outlineFilters.applied')
        apply_seconds = perf_counter() - start
        grey = page.evaluate('[...window.pyMicrogliaExplorer.greyOutlineIds]')
        assert grey and len(grey) < len(original)
        assert page.evaluate('window.pyMicrogliaExplorer.data.identities') == original
        assert page.evaluate('window.pyMicrogliaExplorer.sourceData.identities') == original
        assert page.locator('#xp-cell-select option').count() == len(original)
        selected = grey[0]
        page.locator('#xp-cell-select').select_option(str(selected))
        page.wait_for_function('(id)=>window.pyMicrogliaExplorer.state.identity===id && window.pyMicrogliaExplorer.exact?.cell.identity===id', arg=selected)
        page.evaluate('(()=>{const e=window.pyMicrogliaExplorer;if(!e.exact.cell.observed[e.state.frame])e.select({frame:e.exact.cell.observed.findIndex(Boolean)});})()')
        page.wait_for_function('(()=>{const e=window.pyMicrogliaExplorer;return e.history?.identity===e.state.identity && e.imageMeta?.frame_index===e.state.frame && e.history.observations.some(row=>row.frame_index===e.state.frame&&row.observation);})()', timeout=60000)
        assert page.evaluate('(id)=>window.pyMicrogliaExplorer.outlineColour(id,"white")', selected) == '#aaaaaa'
        assert page.evaluate('window.pyMicrogliaExplorer.outlineColour(-1,"white")') == 'white'
        # Capture actual contour ink, including a selected grey cell.
        colours = page.evaluate("""(()=>{const e=window.pyMicrogliaExplorer,original=e.strokeContours,ink=[];e.strokeContours=function(ctx,item,project,colour,width){ink.push(colour);return original.call(this,ctx,item,project,colour,width);};e.drawImage();e.strokeContours=original;return ink;})()""")
        assert '#aaaaaa' in colours, colours
        capture(page, output / 'grey-outlines-desktop.png')
        # The export form must receive structured decisions, not a JSON string filename.
        page.get_by_role('button', name='Export outlined video', exact=True).click()
        page.wait_for_selector('#action-fields [name="cell_filter"]')
        exported = json.loads(page.locator('#action-fields [name="cell_filter"]').input_value())
        assert exported['identities'] == page.evaluate('window.pyMicrogliaExplorer.outlineFilters.applied.identities')
        page.evaluate('document.querySelector("#action-dialog").close()')
        # An edited draft never changes the applied grey decisions.
        page.locator('#xp-grey-filter-clear').click()
        assert page.evaluate('[...window.pyMicrogliaExplorer.greyOutlineIds]') == grey
        page.reload()
        page.wait_for_function('window.pyMicrogliaExplorer?.outlineFilters?.applied', timeout=60000)
        assert page.evaluate('[...window.pyMicrogliaExplorer.greyOutlineIds]') == grey
        assert page.evaluate('window.pyMicrogliaExplorer.data.identities') == original
        assert page.evaluate('window.pyMicrogliaExplorer.outlineFilters.tree.children.length') == 0
        # Independent exclusions: changing one measurement's cohort leaves grey rules intact.
        page.evaluate("""async()=>{const e=window.pyMicrogliaExplorer,row=e.rhythmAnalyses.rows[0];await row.editor.load();row.editor.tree={field:'identity',cmp:'eq',value:e.sourceData.identities[0]};row.editor.mode='exclude';await row.editor.evaluate(true);}""")
        assert page.evaluate('[...window.pyMicrogliaExplorer.greyOutlineIds]') == grey
        assert page.evaluate('window.pyMicrogliaExplorer.rhythmAnalyses.rows[0].editor.applied.excluded') == 1
        page.evaluate('window.pyMicrogliaExplorer.rhythmAnalyses.rows[0].editor.off()')
        page.locator('#xp-grey-filter-off').click()
        assert page.evaluate('window.pyMicrogliaExplorer.greyOutlineIds') is None
        assert page.evaluate('window.pyMicrogliaExplorer.data.identities') == original
        for theme in ('light', 'dark'):
            page.evaluate('theme=>document.documentElement.dataset.pmTheme=theme', theme)
            capture(page, output / f'controls-{theme}.png')
        page.set_viewport_size({'width': 800, 'height': 900})
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        capture(page, output / 'controls-narrow.png')
        page.set_viewport_size({'width': 1440, 'height': 900})
        assert not errors and not jobs, {'errors': errors, 'jobs': jobs}
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in before.items())
    result = {'status': 'passed', 'original_cells': len(original), 'grey_cells': len(grey),
              'apply_seconds': apply_seconds, 'native_contour_ink': colours,
              'draft_and_applied_restored': True, 'independent_exclusions': True,
              'source_tables_unchanged': True, 'analysis_jobs': jobs, 'browser_errors': errors}
    (output / 'browser_receipt.json').write_text(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:56691')
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--decisions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.url, args.run, args.decisions, args.output)))
