"""Run through browser-use-auto with exec(Path(...).read_text()).

Checks the real browser's audit page without running a recording-wide analysis.
The backend tests independently check scientific parity and script replay.
"""
import json
import os
from pathlib import Path
import time

audit_check_output = Path(os.environ.get("TEMP", ".")) / "pymicroglia-rhythm-audit-ui"
audit_check_output.mkdir(exist_ok=True)


def audit_check_wait(expression, timeout=45):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if js(expression):
            return
        time.sleep(.2)
    raise AssertionError("Timed out: " + expression)


js("location.reload()")
audit_check_wait("!!document.querySelector('[data-page=\"rhythm-audit\"]')", 60)
js("document.querySelector('[data-page=\"rhythm-audit\"]').click()")
audit_check_wait("!!window.pyMicrogliaRhythmAudit?.schema", 60)
audit_check_original_session = js("localStorage.getItem('pm-exploration-session-v1')")
audit_check_original_audit = js("localStorage.getItem('pm-rhythm-audit-v1')")
js("""window.raCheck={posts:[],errors:[]};
window.addEventListener('error',event=>raCheck.errors.push(event.message));
window.addEventListener('unhandledrejection',event=>raCheck.errors.push(String(event.reason)));
window.raCheckFetch=window.fetch;window.fetch=(url,options={})=>{
if(options.method==='POST'&&String(url).includes('-jobs'))raCheck.posts.push(String(url));
return window.raCheckFetch(url,options);};
document.querySelector('[data-page="rhythm-audit"]').click();""")
audit_check_wait("!!window.pyMicrogliaRhythmAudit?.schema")
assert js("document.documentElement.dataset.pmPage") == "rhythm-audit"
js("""(()=>{const audit=window.pyMicrogliaRhythmAudit;
audit.setForm({...audit.request(),detrend_methods:['linear','robust_linear'],period_methods:['lomb','fft_nlls'],trace_normalization:'none',input_filters:[{method:'none'}]});
if(audit.request().input_filters.length!==1)throw Error('An extra filter was enabled by default');
document.querySelector('#ra-add-filter').click();
document.querySelector('#ra-add-filter').click();
document.querySelector('#ra-input-filters').children[1].querySelector('input').value=5;
document.querySelector('#ra-add-filter').click();
document.querySelector('#ra-input-filters').children[2].querySelector('select').value='mean';
audit.changed();audit.run();})()""")
audit_check_wait("window.pyMicrogliaRhythmAudit.result?.options.detrend_methods.length===2 && window.pyMicrogliaRhythmAudit.result?.options.period_methods.length===2 && !window.pyMicrogliaRhythmAudit.running", 90)
assert js("document.querySelectorAll('.ra-tile').length") == 4
assert js("document.querySelectorAll('.ra-tile canvas[data-drawn=true]').length") == 4
assert js("window.pyMicrogliaRhythmAudit.result.results.length") == 16
assert js("[...new Set(window.pyMicrogliaRhythmAudit.result.results.map(row=>row.preprocessor))]") == ['raw','median_3','median_5','mean_3']
assert js("document.querySelectorAll('.ra-tile .ra-result').length") == 16
assert js("raCheck.posts.length") == 1
assert js("""[...document.querySelectorAll('#rhythm-audit-host .ra-panel')].every(panel=>
['Edit plot','Settings','Export'].every(label=>[...panel.querySelectorAll('.xp-panel-actions button')].filter(button=>button.textContent===label).length===1))""")

audit_check_identity = js("window.pyMicrogliaRhythmAudit.result.audit_id")
js("document.querySelector('#ra-cell').selectedIndex=0;document.querySelector('#ra-cell').dispatchEvent(new Event('change',{bubbles:true}));")
assert js("window.pyMicrogliaRhythmAudit.result.audit_id") == audit_check_identity
assert "Settings changed" in js("document.querySelector('#ra-pending').textContent") or js("document.querySelector('#ra-cell').value==String(window.pyMicrogliaRhythmAudit.result.options.identity)")
assert js("raCheck.posts.length") == 1
js("document.querySelector('#ra-use-shown').click()")
assert "matches" in js("document.querySelector('#ra-pending').textContent")

audit_check_selection_ms = js("""(()=>{const start=performance.now();const audit=window.pyMicrogliaRhythmAudit;
const row=audit.result.results.find(row=>row.preprocessor==='mean_3');audit.inspect(row.detrend,row.estimator,row.preprocessor);
return performance.now()-start;})()""")
assert not js("document.querySelector('#ra-promote').disabled")
audit_check_rows = js("window.pyMicrogliaExplorer.rhythmAnalyses.rows.length")
audit_check_expected = js("window.pyMicrogliaRhythmAudit.transfer(window.pyMicrogliaRhythmAudit.selected).options")
js("window.pyMicrogliaRhythmAudit.promote()")
audit_check_wait("document.documentElement.dataset.pmPage==='follow'")
assert js("window.pyMicrogliaExplorer.rhythmAnalyses.rows.length") == audit_check_rows + 1
audit_check_actual = js("window.pyMicrogliaExplorer.rhythmAnalyses.options(window.pyMicrogliaExplorer.rhythmAnalyses.rows.at(-1))")
for audit_check_key, audit_check_value in audit_check_expected.items():
    assert audit_check_actual[audit_check_key] == audit_check_value, audit_check_key
assert js("raCheck.posts.length") == 1  # transfer config starts no population run

js("""window.pyMicrogliaExplorer.options.onAudit({identity:window.pyMicrogliaExplorer.state.identity,
metric:window.pyMicrogliaExplorer.rhythmAnalyses.rows.at(-1).metric.value});""")
audit_check_wait("document.documentElement.dataset.pmPage==='rhythm-audit'")
js("""(()=>{document.querySelector('#ra-settings-audit-grid').open=true;
for(const [id,value] of [['ra-guide-every','12'],['ra-guide-offset','3']]){const el=document.getElementById(id);el.value=value;el.dispatchEvent(new Event('change'));}
const el=document.querySelector('#xp-style-audit-grid-lineWidth');el.value=2.5;el.dispatchEvent(new Event('change'));
window.pyMicrogliaRhythmAudit.save();})()""")
audit_check_saved_rows = js("JSON.parse(localStorage.getItem('pm-exploration-session-v1')).rhythmRows.rows.length")
js("location.reload()")
audit_check_wait("!!window.pyMicrogliaRhythmAudit?.result && !!window.pyMicrogliaRhythmAudit?.facade", 90)
assert js("document.documentElement.dataset.pmPage") == "rhythm-audit"
assert js("window.pyMicrogliaRhythmAudit.result.audit_id") == audit_check_identity
assert js("window.pyMicrogliaRhythmAudit.request().input_filters") == [{'method':'none'},{'method':'median','window_points':3},{'method':'median','window_points':5},{'method':'mean','window_points':3}]
assert js("window.pyMicrogliaRhythmAudit.styles['audit-grid'].lineWidth") == 2.5
assert js("window.pyMicrogliaRhythmAudit.view.guides") == "12"
assert js("window.pyMicrogliaRhythmAudit.view.offset") == "3"
assert js("JSON.parse(localStorage.getItem('pm-exploration-session-v1')).rhythmRows.rows.length") == audit_check_saved_rows
assert js("document.querySelectorAll('#rhythm-audit-host .ra-tile canvas[data-drawn]').length") == 4
assert js("window.pyMicrogliaExplorer.rhythmAnalyses.options(window.pyMicrogliaExplorer.rhythmAnalyses.rows.at(-1)).trace_filter") == audit_check_expected["trace_filter"]
assert js("window.pyMicrogliaExplorer.rhythmAnalyses.rows.at(-1).$('rhythm-prefilter').value") == "mean"
assert js("Number(window.pyMicrogliaExplorer.rhythmAnalyses.rows.at(-1).$('rhythm-median-points').value)") == audit_check_expected["trace_filter"]["window_points"]

audit_check_layouts = []
for audit_check_theme in ("light", "dark"):
    for audit_check_width in (1440, 800):
        cdp("Emulation.setDeviceMetricsOverride", width=audit_check_width, height=1000, deviceScaleFactor=1, mobile=False)
        js(f"document.documentElement.dataset.pmTheme={audit_check_theme!r};window.pyMicrogliaRhythmAudit.draw()")
        audit_check_wait("document.querySelector('#xp-audit-raw').width>100")
        audit_check_geometry = js("""(()=>{const host=document.querySelector('#rhythm-audit-host'),r=host.getBoundingClientRect();return {
pageOverflow:document.documentElement.scrollWidth>innerWidth+1,hostWidth:r.width,
guides:[...host.querySelectorAll('.xp-guide-controls label')].map(el=>el.getBoundingClientRect().y),
actions:[...host.querySelectorAll('.xp-panel-actions')].map(el=>({left:el.getBoundingClientRect().left,right:el.getBoundingClientRect().right})),
canvasHeight:document.querySelector('#xp-audit-raw').getBoundingClientRect().height};})()""")
        assert not audit_check_geometry["pageOverflow"], (audit_check_theme, audit_check_width, audit_check_geometry)
        assert len(set(audit_check_geometry["guides"])) == 1
        assert all(item["right"] <= audit_check_width + 1 for item in audit_check_geometry["actions"])
        assert audit_check_geometry["canvasHeight"] == 200  # no device-ratio growth on redraw
        audit_check_layouts.append({"theme": audit_check_theme, "width": audit_check_width, **audit_check_geometry})

cdp("Emulation.clearDeviceMetricsOverride")
js("document.documentElement.dataset.pmTheme=localStorage.getItem('pm-extras-theme')==='dark'?'dark':'light';document.querySelector('#ra-settings-audit-grid').open=false;window.pyMicrogliaRhythmAudit.draw();window.scrollTo(0,0)")
audit_check_result = {"audit_id": audit_check_identity, "transferred_options": audit_check_expected,
                      "selection_ms": audit_check_selection_ms, "layouts": audit_check_layouts,
                      "saved_rows": audit_check_saved_rows, "status": "passed"}
(audit_check_output / "verification.json").write_text(json.dumps(audit_check_result, indent=2))
js("localStorage.setItem('pm-exploration-session-v1',"+json.dumps(audit_check_original_session)+")")
js("localStorage.setItem('pm-rhythm-audit-v1',"+json.dumps(audit_check_original_audit)+");location.reload()")
print(json.dumps(audit_check_result))
