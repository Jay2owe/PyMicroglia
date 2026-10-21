"""Check event filtering, original-frame playback and a real native event-video job."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from time import perf_counter
from urllib.parse import urlencode

from playwright.sync_api import sync_playwright


def sidebar_appearance(page):
    """Measure shared navigation; only the active-section highlight may change."""
    return page.evaluate("""() => {
      const selectors=['.masthead','.brand','.top-tabs','#micro-sidebar',
        '#micro-sidebar .sidebar-head','#micro-sidebar .sidebar-head h2',
        '#micro-sidebar .sidebar-head p','#micro-nav','#micro-theme',
        '#micro-sidebar>.sidebar-foot'];
      const snapshot=element=>{const b=element.getBoundingClientRect(),s=getComputedStyle(element);
        return {bounds:[b.x,b.y,b.width,b.height].map(v=>Math.round(v*100)/100),
          padding:s.padding,font:s.font,borderRadius:s.borderRadius,
          colour:s.color,background:s.backgroundColor,display:s.display};};
      const result=Object.fromEntries(selectors.map(selector=>[selector,snapshot(document.querySelector(selector))]));
      for(const button of document.querySelectorAll('#micro-nav button')){
        const item=snapshot(button);delete item.colour;delete item.background;
        result['button:'+button.textContent]=item;
      }
      return result;
    }""")


def wait_grid(page):
    page.wait_for_function("""window.pyMicrogliaEvents?.data &&
      document.querySelectorAll('#ev-list [data-preview=ready]').length===window.pyMicrogliaEvents.filtered().length &&
      document.getElementById('ev-grid-progress').dataset.completed==window.pyMicrogliaEvents.filtered().length""",timeout=60000)


def wait_frame(page):
    page.wait_for_function("""document.getElementById('ev-image').dataset.eventId===window.pyMicrogliaEvents.selected.id &&
      document.getElementById('ev-image').dataset.frame==window.pyMicrogliaEvents.frame &&
      !document.getElementById('ev-image').hasAttribute('aria-busy')""",timeout=30000)


def check(page, url, run, output):
    errors, posts = [], []
    page.on('pageerror',lambda error:errors.append(str(error)))
    page.on('request',lambda request:posts.append(request.url) if request.method=='POST' else None)
    # Migrate a previous one-event review to the whole grid, retaining display settings.
    prior={'version':1,'run':str(run),'stem':'95_A3','filter':'merges','status':'not-censored',
           'search':'33','settings':{'crop_px':96,'event_hours':6,'fps':6,'cell_lut':'red'},'media':'preview'}
    page.add_init_script("if(!sessionStorage.getItem('pm-events-test-seeded')){localStorage.setItem('pm-workspace','events');localStorage.setItem('pm-events-v1',"+json.dumps(json.dumps(prior))+");sessionStorage.setItem('pm-events-test-seeded','1');}")
    started=perf_counter()
    page.goto(url+'/?'+urlencode({'run':str(run)}))
    page.wait_for_function("document.querySelector('#ev-list [data-preview=ready] img')?.naturalWidth>0",timeout=30000)
    first_usable=perf_counter()-started
    wait_grid(page)
    all_loaded=perf_counter()-started
    total=page.evaluate('window.pyMicrogliaEvents.data.total')
    lifecycle_saved=page.evaluate('window.pyMicrogliaEvents.data.lifecycle_saved')
    assert page.locator('#ev-list .ev-tile').count()==total==(300 if lifecycle_saved else 104)
    assert page.locator('#ev-gallery').is_visible() and page.locator('#ev-detail').is_hidden()
    assert page.locator('#ev-filter').input_value()=='all' and page.locator('#ev-status').input_value()=='all'
    assert page.locator('#ev-search').input_value()==''
    assert page.locator('.top-tabs button').all_text_contents()==['Auto-Organotypic','Motion tracks','PyMicroglia extras']
    assert page.locator('#micro-nav [data-page=events]').get_attribute('class')=='active'
    assert page.evaluate("localStorage.getItem('pm-workspace')==='micro' && localStorage.getItem('pm-page')==='events'")
    assert page.locator('#micro-sidebar #events-sidebar').count()==0
    assert page.locator('#ev-options #events-sidebar').is_visible()
    assert page.locator('#motion-workspace').is_hidden() and page.locator('#auto-workspace').is_hidden()
    assert ('original accepted Motion masks' if lifecycle_saved else 'No saved birth/death') in page.locator('#ev-source-note').inner_text()
    assert not posts, posts
    assert page.evaluate("""window.pyMicrogliaEvents.data.rows.filter(row=>row.source==='tracking').every(row=>{
      const tile=document.querySelector('[data-event="'+row.id+'"]');
      return Number(tile.dataset.sourceFrame)===row.source_frame && !tile.querySelector('.ev-tile-mask').hidden;
    })"""), 'Saved missing masks must remain explicitly absent in every gap preview'
    keys=page.locator('#ev-tabs button').evaluate_all('(buttons)=>buttons.map(b=>b.dataset.type)')
    assert {'all','born','divided','merges','died','lost','vanished','gaps','states'}<=set(keys)
    page.locator('#ev-tab-born').click()
    if lifecycle_saved:
        wait_grid(page)
        assert page.locator('#ev-list .ev-tile').count()==8
        assert page.evaluate("window.pyMicrogliaEvents.filtered().every(row=>row.status==='candidate')")
        assert page.evaluate("window.pyMicrogliaEvents.filtered().some(row=>row.identity===141 && row.partners.includes(39) && row.source_frame===12)")
        assert page.locator('.ev-tile-mask:visible').count()==0
        page.screenshot(path=str(output/'events-births.png'))
        page.locator('#ev-tab-vanished').click()
    assert page.locator('#ev-list .ev-tile').count()==0
    assert ('No saved vanished' if lifecycle_saved else 'No saved births') in page.locator('#ev-empty').inner_text()
    page.locator('#ev-tab-vanished' if lifecycle_saved else '#ev-tab-born').press('Home')
    wait_grid(page)
    assert page.locator('#ev-tab-all').get_attribute('aria-selected')=='true'
    # Earlier requests cannot paint a stale category after rapid tab changes.
    page.evaluate("document.getElementById('ev-tab-gaps').click();document.getElementById('ev-tab-merges').click();document.getElementById('ev-tab-all').click()")
    wait_grid(page)
    assert page.locator('#ev-list .ev-tile').count()==total
    red=page.locator('#ev-list img').first.get_attribute('src')
    page.locator('#ev-cell_lut').select_option('grays')
    wait_grid(page)
    assert page.locator('#ev-list img').first.get_attribute('src')!=red
    page.locator('#ev-cell_lut').select_option('red')
    wait_grid(page)
    for theme in ['light','dark']:
        page.locator('#micro-theme').select_option(theme)
        for width in [1440,800]:
            page.set_viewport_size({'width':width,'height':900})
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
            image=page.locator('.ev-tile-media').first.bounding_box()
            assert abs(image['width']-image['height'])<2
            assert page.locator('#ev-list .ev-tile').count()==total
            assert page.locator('#micro-theme').bounding_box()['y']<900
            if width>=800:
                grid=page.locator('#ev-gallery').bounding_box()
                options=page.locator('#ev-options').bounding_box()
                assert options['x']>=grid['x']+grid['width'],(options,grid)
            events_sidebar=sidebar_appearance(page)
            page.screenshot(path=str(output/f'events-grid-{theme}-{width}.png'))
            page.locator('#micro-nav [data-page=follow]').click()
            page.wait_for_function("window.pyMicrogliaExplorer?.data && document.documentElement.dataset.pmPage==='follow'",timeout=30000)
            assert sidebar_appearance(page)==events_sidebar,{'theme':theme,'width':width}
            assert page.locator('#events-workspace').is_hidden()
            page.locator('#micro-nav [data-page=events]').click()
            assert sidebar_appearance(page)==events_sidebar
            wait_grid(page)
    page.set_viewport_size({'width':1440,'height':900})
    page.locator('#ev-tab-merges').click()
    page.locator('#ev-search').fill('33')
    wait_grid(page)
    started=perf_counter()
    page.locator('#ev-list .ev-tile').first.click()
    wait_frame(page)
    selection=perf_counter()-started
    assert page.locator('#ev-gallery').is_hidden() and page.locator('#ev-detail').is_visible()
    selected=page.evaluate("({identity:window.pyMicrogliaEvents.selected.identity,source_frame:window.pyMicrogliaEvents.selected.source_frame,frame:window.pyMicrogliaEvents.frame,event:window.pyMicrogliaEvents.selected.id})")
    assert selected['identity']==33 and selected['source_frame']==19
    assert 'no mask' in page.locator('#ev-frame-note').inner_text()
    assert 'not a confirmed cell fate' in page.locator('#ev-event-source').inner_text()
    media=page.locator('.ev-stage').bounding_box()
    assert abs(media['width']-media['height'])<2
    page.locator('#ev-next').click();wait_frame(page)
    assert page.evaluate('window.pyMicrogliaEvents.frame')==selected['frame']+1
    page.locator('#ev-back').click()
    assert page.locator('#ev-gallery').is_visible()
    page.locator('#ev-open').click();wait_frame(page)
    page.locator('#ev-event_hours').fill('1');page.locator('#ev-event_hours').dispatch_event('change')
    assert not any('/api/events/clips' in request or '/api/jobs' in request for request in posts), posts
    page.locator('#ev-generate').click()
    page.wait_for_function("window.pyMicrogliaEvents.selected && window.pyMicrogliaEvents.data.clips.some(c=>c.event_id===window.pyMicrogliaEvents.selected.id&&c.options?.event_hours===1) && !document.getElementById('ev-video').hidden",timeout=90000)
    page.wait_for_function("document.getElementById('ev-video').readyState>=2",timeout=30000)
    assert page.locator('#ev-download').is_visible() and page.locator('#ev-script').is_visible()
    page.locator('#ev-video').evaluate('(video)=>video.play()')
    page.wait_for_function("document.getElementById('ev-video').currentTime>0.1")
    page.locator('#ev-video').evaluate('(video)=>video.pause()')
    page.screenshot(path=str(output/'events-video.png'))
    page.locator('#ev-preview-mode').click();wait_frame(page)
    page.locator('#ev-inspect').click()
    page.wait_for_function("document.documentElement.dataset.pmPage==='follow' && window.pyMicrogliaExplorer?.state.identity===33 && window.pyMicrogliaExplorer.state.frame===window.pyMicrogliaEvents.frame",timeout=30000)
    assert page.locator('#events-workspace').is_hidden() and page.locator('#exploration-host').is_visible()
    page.locator('#micro-nav [data-page=events]').click()
    assert page.locator('#ev-filter').input_value()=='merges' and page.locator('#ev-search').input_value()=='33'
    for workspace in ['auto','motion']:
        page.locator(f'[data-workspace={workspace}]').click()
        assert page.locator('#events-workspace').is_hidden()
        assert page.locator(f'#{workspace}-workspace').is_visible()
        page.locator('[data-workspace=micro]').click()
        assert page.locator('#micro-nav [data-page=events]').get_attribute('class')=='active'
    page.locator('#ev-back').click()
    selected_id=page.evaluate('window.pyMicrogliaEvents.selected.id')
    page.reload();wait_grid(page)
    assert page.evaluate('window.pyMicrogliaEvents.selected.id')==selected_id
    assert page.locator('#ev-gallery').is_visible() and page.locator('#ev-detail').is_hidden()
    assert page.locator('#ev-filter').input_value()=='merges' and page.locator('#ev-search').input_value()=='33'
    assert page.locator('#ev-event_hours').input_value()=='1'
    assert not errors, errors
    assert len([p for p in posts if '/api/events/clips' in p])==1
    return {'events':total,'first_usable_seconds':round(first_usable,3),'all_loaded_seconds':round(all_loaded,3),
            'selection_seconds':round(selection,3),'selected':selected,'browser_errors':errors,'posts':posts,
            'all_events_loaded_automatically':True,'sidebar_appearance_unchanged':True,'event_options_on_right':True,
            'lifecycle_saved':lifecycle_saved,'birth_candidates':8 if lifecycle_saved else 0,
            'empty_types_visible':True,'source_alignment_and_missing_masks_preserved':True,'session_migration_and_reload':True}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--url',default='http://127.0.0.1:62855')
    args=parser.parse_args()
    run=args.run.resolve();table=run/'measure/95_A3/cell_frame.csv'
    before=hashlib.sha256(table.read_bytes()).hexdigest()
    output=Path(tempfile.mkdtemp(prefix='pymicroglia-events-ui-'))
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path=r'C:\Program Files\Google\Chrome\Application\chrome.exe',headless=True)
        try:
            page=browser.new_page(viewport={'width':1440,'height':900})
            receipt=check(page,args.url,run,output)
        finally:browser.close()
    assert hashlib.sha256(table.read_bytes()).hexdigest()==before
    receipt.update(measurements_unchanged=True,source_sha256=before,screenshots=str(output))
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt))


if __name__=='__main__':main()
