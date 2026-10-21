"""Exercise native comparison cards in the dedicated automation browser."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from time import perf_counter
from urllib.parse import urlencode

import pandas as pd
from playwright.sync_api import sync_playwright


def fixture(root):
    movies=[]
    for i in range(6):
        stem=f'comparison_{i}'
        folder=root/'measure'/stem
        folder.mkdir(parents=True)
        pd.DataFrame([dict(identity=cell, frame_index=frame, hours=float(frame),
            signal_mean=10+cell+i*5+frame, integrated_density=100+cell+i*20+frame,
            area_px=cell*3+frame) for cell in range(1,9) for frame in range(4)]).to_csv(folder/'cell_frame.csv',index=False)
        movies.append(dict(stem=stem,spec={'condition':'Control' if i<3 else 'Treatment'},
            tables={},summary={'frames':4,'scale':{'minutes_per_frame':60}}))
    (root/'manifest.json').write_text(json.dumps({'movies':movies}),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--url',default='http://127.0.0.1:62862')
    args=parser.parse_args()
    root=Path(tempfile.mkdtemp(prefix='pymicroglia-group-comparison-'))
    fixture(root)
    before={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in root.glob('measure/*/cell_frame.csv')}
    errors=[]
    with sync_playwright() as playwright:
        browser=playwright.chromium.connect_over_cdp('http://127.0.0.1:9333')
        context=browser.new_context(viewport={'width':1440,'height':1000},accept_downloads=True)
        page=context.new_page()
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.add_init_script("localStorage.setItem('pm-workspace','micro');localStorage.setItem('pm-page','compare');")
        try:
            started=perf_counter()
            print('Opening isolated comparison fixture',flush=True)
            page.goto(args.url+'/?'+urlencode({'run':str(root)}))
            print('Waiting for native comparison',flush=True)
            page.wait_for_function("window.pyMicrogliaExplorer?.groupComparison?.states.get('distribution')?.preview",timeout=60000)
            first=perf_counter()-started
            assert page.evaluate("window.pyMicrogliaExplorer.groupComparison.states.get('distribution').preview.counts.samples")==6
            assert page.evaluate("window.pyMicrogliaExplorer.groupComparison.states.get('distribution').preview.counts.cells")==48
            panel=page.locator('#xp-distribution-panel')
            for name in ('Edit plot','Settings','Export','Exclusions'):
                assert panel.get_by_role('button',name=name,exact=True).count()==1,name
            page.locator('#xp-distribution-comparisons').get_by_role('button',name='Add plot',exact=True).click()
            page.wait_for_function("window.pyMicrogliaExplorer.groupComparison.states.get('distribution-2')?.preview")
            page.locator('#xp-distribution-view-distribution-2').select_option('cells')
            page.wait_for_function("window.pyMicrogliaExplorer.groupComparison.states.get('distribution-2').preview?.hits.length===48")
            assert page.locator('#xp-distribution-view').input_value()=='samples'
            page.locator('#xp-distribution-columns').select_option('2')
            assert page.locator('#xp-distribution-columns').bounding_box()['y']<panel.bounding_box()['y']
            assert page.locator('#xp-comparison-design').evaluate('(element)=>element.open')
            assert page.locator('#xp-comparison-recordings tbody tr').count()==6
            page.locator('#xp-comparison-target').select_option('recordings')
            page.locator('#xp-comparison-condition').fill('Recovery')
            page.locator('#xp-comparison-recordings input[data-field="selected"]').first.check()
            page.wait_for_function('window.pyMicrogliaExplorer.groupComparison.selectionPreview?.cells===8')
            page.locator('#xp-comparison-assign').click()
            page.wait_for_function("window.pyMicrogliaExplorer.groupComparison.states.get('distribution')?.preview?.groups.length===3")
            page.wait_for_function("document.querySelector('#xp-comparison-design-status').textContent==='Saved'")
            page.reload(wait_until='domcontentloaded')
            page.wait_for_function("window.pyMicrogliaExplorer?.groupComparison?.states.get('distribution-2')?.preview?.hits.length===48",timeout=60000)
            assert page.locator('#xp-distribution-columns').input_value()=='2'
            assert 'Recovery' in page.locator('#xp-comparison-recordings tbody tr').first.inner_text()
            # Native statistics are explicit and saved; no rhythmicity job is run.
            page.evaluate("""() => {const e=window.pyMicrogliaExplorer;e.groupComparison.design.recordings[0].condition=e.groupComparison.design.recordings[1].condition;e.groupComparison.renderAssignments();e.groupComparison.changed();}""")
            page.wait_for_function("window.pyMicrogliaExplorer.groupComparison.states.get('distribution')?.preview?.groups.length===2")
            panel.get_by_role('button',name='Settings',exact=True).click()
            page.locator('#xp-distribution-statistics > summary').click()
            page.locator('#xp-distribution-run').click()
            page.wait_for_function("window.pyMicrogliaExplorer.groupComparison.states.get('distribution')?.result?.statistics.length",timeout=60000)
            stats=page.evaluate("window.pyMicrogliaExplorer.groupComparison.states.get('distribution').result.statistics[0]")
            assert stats['n_a']==stats['n_b']==3
            completed=page.evaluate("window.pyMicrogliaExplorer.groupComparison.states.get('distribution').resultId")
            page.locator('#xp-distribution-test').select_option('welch_t')
            page.wait_for_function("document.querySelector('#xp-distribution-result-status').textContent.includes('previous completed')")
            assert page.evaluate("window.pyMicrogliaExplorer.groupComparison.states.get('distribution').resultId")==completed
            page.locator('#xp-distribution-test').select_option('mannwhitney')
            page.locator('#xp-distribution-view').select_option('effects')
            page.wait_for_function("window.pyMicrogliaExplorer.groupComparison.states.get('distribution')?.displayedRequest?.options.view==='effects'")
            for view in ('box','violin','histogram','time_course','samples'):
                page.locator('#xp-distribution-view').select_option(view)
                page.wait_for_function("view=>window.pyMicrogliaExplorer.groupComparison.states.get('distribution')?.displayedRequest?.options.view===view",arg=view)
            # A plot override must affect all recordings and leave the other card intact.
            page.evaluate("""() => {const e=window.pyMicrogliaExplorer;e.rhythmAnalyses.plotFilters.distribution={integrated_density:{enabled:true,filter:{applied:{expression:{field:'observed_frames',cmp:'lt',value:5},mode:'exclude',unknown:'keep'}}}};e.groupComparison.draw('distribution');}""")
            page.wait_for_function("window.pyMicrogliaExplorer.groupComparison.states.get('distribution')?.preview?.counts.cells===0")
            assert page.evaluate("window.pyMicrogliaExplorer.groupComparison.states.get('distribution-2').preview.counts.cells")==48
            page.evaluate("delete window.pyMicrogliaExplorer.rhythmAnalyses.plotFilters.distribution;window.pyMicrogliaExplorer.groupComparison.draw('distribution')")
            page.wait_for_function("window.pyMicrogliaExplorer.groupComparison.states.get('distribution')?.preview?.counts.cells===48")
            # Shared appearance and exclusion editors save through the same session callback.
            page.evaluate("""() => {const e=window.pyMicrogliaExplorer;e.styles.distribution={...e.styles.distribution,height:240};e.saveSession();e.groupComparison.draw('distribution');}""")
            page.wait_for_function("document.querySelector('#xp-comparison-design-status').textContent==='Saved'")
            saved=json.loads((root/'AI_Exports/comparisons/workspace.json').read_text())
            assert saved['plots']['distribution']['appearance']['height']==240
            page.reload(wait_until='domcontentloaded')
            page.wait_for_function("window.pyMicrogliaExplorer?.groupComparison?.states.get('distribution')?.preview?.height===240",timeout=60000)
            # Clicking a native cell point opens its source without replacing the design.
            target=page.evaluate("""() => {const e=window.pyMicrogliaExplorer,s=e.groupComparison.states.get('distribution-2');return s.preview.hits.find(hit=>hit.stem!==e.data.stem);}""")
            canvas=page.locator('#xp-distribution-2')
            canvas.scroll_into_view_if_needed()
            box=canvas.bounding_box()
            dimensions=page.evaluate("window.pyMicrogliaExplorer.groupComparison.states.get('distribution-2').preview")
            canvas.click(position={'x':target['x']*box['width']/dimensions['width'],'y':target['y']*box['height']/dimensions['height']})
            page.wait_for_function("stem=>window.pyMicrogliaExplorer.data.stem===stem",arg=target['stem'])
            page.wait_for_function("window.pyMicrogliaExplorer.groupComparison.states.get('distribution-2')?.preview?.counts.cells===48")
            assert page.evaluate("window.pyMicrogliaExplorer.groupComparison.design.recordings.length")==6
            for theme,width in [('dark',1440),('light',800)]:
                page.set_viewport_size({'width':width,'height':1000})
                page.evaluate("theme=>document.documentElement.dataset.pmTheme=theme",theme)
                page.locator('#xp-distribution-comparisons').scroll_into_view_if_needed()
                page.wait_for_function("document.documentElement.scrollWidth<=innerWidth")
                page.locator('#xp-distribution-comparisons').screenshot(path=str(root/f'comparison-{theme}-{width}.png'))
            assert not errors,errors
        except Exception as error:
            print(repr(error),flush=True)
            try:page.screenshot(path=str(root/'failed.png'),timeout=5000)
            except Exception:pass
            print(json.dumps({'output':str(root),'errors':errors}),flush=True)
            raise
        finally:
            context.close()
    assert before=={str(path):hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in before}
    print(json.dumps({'output':str(root),'first_usable_seconds':first,'native_samples':6,'full_cells':48,'independent_cards':True,'saved_assignments':True,'saved_shared_appearance':True,'cross_recording_click':True,'draft_test_status':True,'plot_exclusion_override':True,'themes_and_narrow_layout':True,'errors':errors}))


if __name__=='__main__':
    main()
