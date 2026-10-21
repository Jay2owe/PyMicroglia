"""Exercise independently saved rhythm setups, exclusion inheritance and guides."""
import argparse
import json
from pathlib import Path
import tempfile
from urllib.parse import urlencode

from playwright.sync_api import sync_playwright
from verify_exploration_rhythms_ui import fixture


def check_shared_rows(page):
    """Catch first-row-only styling and controls that point at another row."""
    rows = page.evaluate('''() => [...document.querySelectorAll('.xp-rhythm-row')].map(row => {
      const bounds=row.getBoundingClientRect();
      return [...row.querySelectorAll('.xp-rhythm-configuration select, .xp-rhythm-run-controls button, .xp-filter-count, .xp-rhythm-progress')].map(el => {
        const box=el.getBoundingClientRect(),style=getComputedStyle(el);
        const layout=el.hidden||el.classList.contains('xp-filter-count')?{}:
          {x:Math.round((box.x-bounds.x)*10)/10, y:Math.round((box.y-bounds.y)*10)/10, width:Math.round(box.width*10)/10};
        return {key:el.dataset.originalId, font:style.font, padding:style.padding, color:style.color,
                ...(el.tagName==='PROGRESS'?{progressWidth:style.width}:{}), ...layout};
      });
    })''')
    assert len(rows) >= 2
    assert [control['key'] for control in rows[0]] == [
        'xp-analysis-metric', 'xp-rhythm-recipe', 'xp-rhythm-detrend', 'xp-rhythm-normalization',
        'xp-run-rhythms', 'xp-rhythm-settings-toggle', 'xp-filter-count', 'xp-rhythm-progress']
    assert all(row == rows[0] for row in rows[1:]), rows
    assert page.evaluate('''() => [...document.querySelectorAll('.xp-rhythm-row [aria-controls]')].every(el =>
      el.getAttribute('aria-controls').split(/\\s+/).every(id => el.closest('.xp-rhythm-row').contains(document.getElementById(id))))''')
    for panel in ('image', 'outline', 'trace'):
        assert page.locator(f'#xp-{panel}-panel').get_by_role('button', name='Exclusions', exact=True).count() == 0
    for panel in ('trace-grid', 'barcode'):
        assert page.locator(f'#xp-{panel}-panel').get_by_role('button', name='Exclusions', exact=True).count() == 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:62858')
    parser.add_argument('--run', type=Path)
    args = parser.parse_args()
    root = Path(tempfile.mkdtemp(prefix='pymicroglia-rhythm-rows-'))
    fixture(root)
    print(root, flush=True)
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, executable_path=r'C:\Program Files\Google\Chrome\Application\chrome.exe')
        page = browser.new_page(viewport={'width': 1440, 'height': 1000})
        page.on('pageerror', lambda error: errors.append(error.stack))
        api_errors = []
        page.on('response', lambda response: api_errors.append(response.url)
                if response.status >= 400 and any(part in response.url.split('?')[0] for part in ('rhythm-', '/filter-','/barcode-order')) else None)
        page.add_init_script("localStorage.setItem('pm-page','follow');localStorage.setItem('pm-workspace','micro');")
        try:
            page.goto(args.url + '/?' + urlencode({'run': str(root)}))
            page.wait_for_function('window.pyMicrogliaExplorer?.rhythmAnalyses?.schema', timeout=60000)
            page.wait_for_function('window.pyMicrogliaExplorer.rhythmAnalyses.rows[0]?.editor && document.querySelector("#xp-analysis-metric-1")')
            source = root / 'measure/synthetic_one/cell_frame.csv'
            original = source.read_bytes()
            page.locator('#xp-rhythm-settings-toggle').click()
            page.locator('#xp-rhythm-surrogates').fill('9')
            page.locator('#xp-run-rhythms').click()
            page.wait_for_function('window.pyMicrogliaExplorer.rhythmAnalyses.rows[0]?.result', timeout=120000)
            first = page.evaluate('window.pyMicrogliaExplorer.rhythmAnalyses.rows[0].result.analysis_id')
            assert 'Completed' not in page.locator('#xp-rhythm-applied').inner_text()
            assert 'Shown:' in page.locator('#xp-rhythm-applied').inner_text()
            page.get_by_role('button', name='+ Add analysis', exact=True).click()
            check_shared_rows(page)
            page.locator('#xp-rhythm-settings-toggle-analysis-2').click()
            assert page.locator('#xp-rhythm-settings-analysis-2').evaluate('(el)=>el.open')
            assert page.locator('#xp-rhythm-settings-toggle-analysis-2').get_attribute('aria-expanded') == 'true'
            page.locator('#xp-rhythm-settings-toggle-analysis-2').click()
            page.get_by_role('button', name='+ Add analysis', exact=True).click()
            check_shared_rows(page)
            page.locator('[data-row="3"]').get_by_role('button', name='Remove', exact=True).click()
            page.locator('#xp-analysis-metric-2').select_option('signal_mean')
            page.locator('#xp-rhythm-recipe-analysis-2').select_option('lomb')
            page.locator('#xp-rhythm-normalization-analysis-2').select_option('zscore')
            page.locator('#xp-run-rhythms-analysis-2').click()
            page.wait_for_function('window.pyMicrogliaExplorer.rhythmAnalyses.rows[1]?.result', timeout=120000)
            second = page.evaluate('window.pyMicrogliaExplorer.rhythmAnalyses.rows[1].result.analysis_id')
            assert first != second
            assert page.evaluate('window.pyMicrogliaExplorer.data.results.rhythms.metrics') == ['integrated_density_mean3', 'signal_mean']
            page.locator('#xp-rhythm-recipe-analysis-2').select_option('fft_component')
            assert page.evaluate('window.pyMicrogliaExplorer.rhythmAnalyses.rows[1].result.analysis_id') == second
            assert 'Draft:' in page.locator('#xp-rhythm-pending-analysis-2').inner_text()
            # Main exclusions are a display cohort; the scientific denominator stays recorded.
            page.locator('#xp-filter-toggle').click()
            page.wait_for_function('window.pyMicrogliaExplorer.rhythmAnalyses.rows[0].editor.schema')
            page.evaluate('''async () => {
              const editor=window.pyMicrogliaExplorer.rhythmAnalyses.rows[0].editor;
              editor.tree={field:"identity",cmp:"eq",value:26};editor.render();await editor.evaluate(true);
            }''')
            print(page.locator('#xp-filter-status').inner_text(), flush=True)
            assert page.evaluate('window.pyMicrogliaExplorer.rhythmAnalyses.metricFilters.integrated_density_mean3.identities') == [19]
            assert page.evaluate('window.pyMicrogliaExplorer.rhythmAnalyses.allowed("trace-grid","signal_mean",26)')
            page.wait_for_function('document.querySelectorAll("#xp-trace-grid .xp-grid-cell")[1].hidden')
            assert page.evaluate('window.pyMicrogliaExplorer.hits.barcode.cells.map(cell=>cell.identity).sort((a,b)=>a-b)') == [19]
            page.locator('#xp-barcode-comparisons').get_by_role('button', name='Add plot', exact=True).click()
            page.evaluate('''() => {const e=window.pyMicrogliaExplorer;e.$("barcode-metric-barcode-2").value="light";e.$("barcode-aggregation-barcode-2").value="mean";e.$("barcode-smoothing-barcode-2").value="1";e.comparisons.drawView(e.comparisons.views.get("barcode-2"));e.saveSession();}''')
            assert page.evaluate('window.pyMicrogliaExplorer.hits["barcode-2"].cells.map(cell=>cell.identity).sort((a,b)=>a-b)') == [19,26]
            page.locator('#xp-rhythm-exclusion-scope').select_option('all')
            assert not page.evaluate('window.pyMicrogliaExplorer.rhythmAnalyses.allowed("trace-grid","signal_mean",26)')
            assert page.evaluate('window.pyMicrogliaExplorer.hits["barcode-2"].cells.map(cell=>cell.identity).sort((a,b)=>a-b)') == [19]
            page.locator('#xp-rhythm-exclusion-scope').select_option('metric')
            # Opening a plot's copied editor must not create an override.
            page.locator('#xp-barcode-panel').get_by_role('button', name='Exclusions', exact=True).click()
            page.wait_for_function('window.pyMicrogliaExplorer.rhythmAnalyses.plotEditors.get("barcode").editor.schema')
            assert not page.evaluate('window.pyMicrogliaExplorer.rhythmAnalyses.plotFilters.barcode?.integrated_density_mean3?.enabled')
            page.evaluate('''async () => {
              const editor=window.pyMicrogliaExplorer.rhythmAnalyses.plotEditors.get("barcode").editor;
              editor.tree={op:"OR",children:[]};editor.render();await editor.evaluate(true);
            }''')
            assert page.evaluate('window.pyMicrogliaExplorer.rhythmAnalyses.allowed("barcode","integrated_density_mean3",26)')
            assert not page.evaluate('window.pyMicrogliaExplorer.rhythmAnalyses.allowed("trace-grid","integrated_density_mean3",26)')
            if not page.locator('#xp-trace-grid-settings').evaluate('(el)=>el.open'):
                page.locator('#xp-trace-grid-panel').get_by_role('button', name='Settings', exact=True).click()
            page.locator('#xp-trace-time-guides').fill('24')
            page.locator('#xp-trace-time-guides').dispatch_event('change')
            page.wait_for_function('JSON.parse(document.querySelector("#xp-trace").dataset.timeGuides||"[]").length===4')
            page.reload()
            page.wait_for_function('window.pyMicrogliaExplorer?.rhythmAnalyses?.rows[1]?.result', timeout=60000)
            assert page.evaluate('window.pyMicrogliaExplorer.rhythmAnalyses.rows.map(r=>r.result.analysis_id)') == [first,second]
            assert page.locator('#xp-rhythm-recipe-analysis-2').input_value() == 'fft_component'
            assert page.locator('#xp-trace-time-guides').input_value() == '24'
            assert page.evaluate('window.pyMicrogliaExplorer.rhythmAnalyses.plotFilters.barcode.integrated_density_mean3.enabled')
            assert page.evaluate('window.pyMicrogliaExplorer.rhythmAnalyses.allowed("trace-grid","signal_mean",26)')
            assert page.evaluate('window.pyMicrogliaExplorer.hits["barcode-2"].cells.map(cell=>cell.identity).sort((a,b)=>a-b)') == [19,26]
            page.locator('#xp-barcode-order').select_option('phase:integrated_density_mean3')
            page.locator('#xp-barcode-panel').get_by_role('button', name='Exclusions', exact=True).click()
            page.locator('#xp-barcode-panel').get_by_role('button', name='Use main exclusions', exact=True).click()
            assert page.evaluate('window.pyMicrogliaExplorer.hits.barcode.cells.map(cell=>cell.identity).sort((a,b)=>a-b)') == [19]
            for width in (1440,800):
                page.set_viewport_size({'width':width,'height':1000})
                assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
                for theme in ('light', 'dark'):
                    page.evaluate('(theme)=>document.documentElement.dataset.pmTheme=theme', theme)
                    check_shared_rows(page)
            page.set_viewport_size({'width':1440,'height':1000})
            page.locator('#xp-rhythm-settings-toggle').click()
            page.locator('#xp-filter-toggle').click()
            page.screenshot(path=str(root / 'rows.png'))
            page.locator('#xp-trace-grid-panel').scroll_into_view_if_needed()
            page.locator('#xp-trace-grid-panel').screenshot(path=str(root / 'guides.png'))
            assert source.read_bytes() == original
            assert not errors
            assert not api_errors, api_errors
            if args.run:
                import hashlib
                from time import perf_counter
                real = args.run.resolve()
                paths = list(real.glob('measure/*/cell_frame.csv'))
                before = {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
                started = perf_counter()
                page.goto(args.url+'/?'+urlencode({'run':str(real)}))
                page.wait_for_function('window.pyMicrogliaExplorer?.rhythmAnalyses?.rows[0]?.editor && window.pyMicrogliaExplorer.rhythmAnalyses.rows[0].result', timeout=60000)
                print('Real recording usable:',round(perf_counter()-started,3),'seconds',flush=True)
                if not page.locator('#xp-trace-grid-settings').evaluate('(el)=>el.open'):
                    page.locator('#xp-trace-grid-panel').get_by_role('button', name='Settings', exact=True).click()
                page.locator('#xp-trace-time-guides').fill('24')
                page.locator('#xp-trace-time-guides').dispatch_event('change')
                page.locator('#xp-trace-grid-panel').scroll_into_view_if_needed()
                page.wait_for_function('window.pyMicrogliaExplorer.gridRows.size===window.pyMicrogliaExplorer.data.identities.length', timeout=60000)
                page.locator('#xp-trace-grid-panel').screenshot(path=str(root/'real-guides.png'))
                page.locator('.xp-rhythm-rows').screenshot(path=str(root/'real-analyses.png'))
                assert {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths} == before
                assert not errors
                assert not api_errors, api_errors
            (root / 'verification.json').write_text(json.dumps({'analyses':[first,second],'errors':errors,'api_errors':api_errors,'source_unchanged':True},indent=2),encoding='utf-8')
        finally:
            browser.close()


if __name__ == '__main__':
    main()
