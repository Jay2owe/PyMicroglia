import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def test_recording_navigation_does_not_wait_for_ordered_workspace_saves():
    script = r"""
    const assert=require('assert'),fs=require('fs'),vm=require('vm');
    const stored=new Map(),context={window:{},setTimeout,clearTimeout,URLSearchParams,
      localStorage:{getItem:key=>stored.get(key)||null,setItem:(key,value)=>stored.set(key,value)}};
    let source=fs.readFileSync('./src/pymicroglia/ui/static/exploration.js','utf8');
    source=source.replace('window.PyMicrogliaExploration=','window.TestExplorer=Explorer;window.PyMicrogliaExploration=');
    vm.runInNewContext(source,context);
    vm.runInNewContext(fs.readFileSync('./src/pymicroglia/ui/static/group-comparison.js','utf8'),context);
    const E=context.window.TestExplorer.prototype,G=context.window.PyMicrogliaGroupComparison.prototype;
    const tick=()=>new Promise(resolve=>setImmediate(resolve));
    (async()=>{
      for(const changed of [false,true]){
        const labels={},writes=[],releases=[];
        const e={data:{run:'first',stem:'one'},token:0,options:{},
          pause(){},cancelMedia(){},saveSession(){},syncRecordingPicker(){},
          $(id){return labels[id]||(labels[id]={textContent:'Saved'});}};
        const design={revision:1},comparison={e,design,run:'owner',epoch:1,
          lastSnapshot:changed?'old':JSON.stringify(design),saveTimer:null,saveQueue:Promise.resolve(),
          save(){return this.design;},checkpoint:G.checkpoint,persist:G.persist};
        e.groupComparison=comparison;
        e.options.api=async(url,options)=>{
          if(options?.method==='PUT'){
            writes.push(JSON.parse(options.body));await new Promise(resolve=>releases.push(resolve));return {};
          }
          assert(url.includes('overview=false'));return {run:'second',stem:'two'};
        };
        e.setData=async data=>{e.data=data;comparison.epoch++;};
        let completed=false;
        E.switchRecording.call(e,{run:'second',stem:'two'}).then(value=>{assert(value);completed=true;});
        await tick();assert(completed,'Recording navigation must finish while disk save is held');
        assert.equal(e.data.run,'second');assert.equal(writes.length,changed?1:0);
        if(changed){
          releases.shift()();await comparison.saveQueue;
          assert.equal(labels['comparison-design-status'].textContent,'Saved');
          assert.equal(JSON.parse(stored.get('pm-comparison-draft:owner')).pending,false);
          comparison.lastSnapshot=JSON.stringify({revision:3});
          const first=comparison.persist(1,'owner',{revision:2},JSON.stringify({revision:2}));
          const second=comparison.persist(2,'owner',{revision:3},comparison.lastSnapshot);
          await tick();assert.equal(writes.length,2,'Second write must wait for first');
          releases.shift()();await first;await tick();
          assert.equal(labels['comparison-design-status'].textContent,'Saving\u2026');
          assert.equal(writes.length,3);assert.equal(writes[2].design.revision,3);
          releases.shift()();await second;
          assert.equal(labels['comparison-design-status'].textContent,'Saved');
        }
      }
      let painted=0;
      const payload={design:{plots:{},layout:{}}},state={key:JSON.stringify(payload),preview:{}};
      const pending={e:{data:{identities:[1]},mediaBusy:true,comparisons:{views:new Map([['plot',{}]])}},
        design:{},states:new Map([['plot',state]]),bindPanel(){},
        element(){return {disabled:false,closest(){return {};}};},value(){return 'box';},
        request(){return payload;},paint(){painted++;},resultStatus(){}};
      await G.draw.call(pending,'plot');assert.equal(painted,0);
      pending.e.mediaBusy=false;await G.draw.call(pending,'plot');assert.equal(painted,1);
      console.log('Navigation, unchanged saves, durability, save ordering and foreground priority passed');
    })().catch(error=>{console.error(error);process.exitCode=1;});
    """
    subprocess.run(["node", "-e", script], cwd=ROOT, check=True, timeout=20)


def test_rhythm_batch_submission_and_results_survive_recording_navigation():
    script = r"""
    const assert=require('assert'),fs=require('fs'),vm=require('vm');
    const context={window:{},URLSearchParams,localStorage:{setItem(){}},console};
    vm.runInNewContext(fs.readFileSync('./src/pymicroglia/ui/static/rhythm-rows.js','utf8'),context);
    const R=context.window.PyMicrogliaRhythmRows.prototype,tick=()=>new Promise(resolve=>setImmediate(resolve));
    (async()=>{
      let release,request;
      const options={metrics:['area_px'],detrend:'none',trace_normalization:'zscore'},row={$(){return {};}};
      const sources=[{run:'one',stem:'same'},{run:'two',stem:'same'}];
      const batch={id:'batch',params:{run:'owner',options:{...options}},recordings:sources};
      const manager={rows:[row],e:{data:{run:'one',stem:'same'},groupComparison:{run:'owner',design:{recordings:sources}},
        options:{api:async(url,payload)=>{request=JSON.parse(payload.body);await new Promise(resolve=>release=resolve);return batch;}},saveSession(){}},
        options(){return options;},batchSources:R.batchSources,describe(){},rememberBatch(value){this.batch=value;},renderBatches(){},pollBatches(){}};
      const submission=R.runAll.call(manager,row);await tick();
      options.detrend='linear';manager.e.data={run:'two',stem:'same'};
      release();await submission;
      assert.equal(request.run,'owner');assert.equal(request.recordings.length,2);
      assert.equal(request.options.detrend,'none','Queued settings are a snapshot');
      assert.equal(row.batchId,'batch','Batch remains associated after navigation');
      let fetched,applied=0;
      const selected={metric:{value:'area_px'},shown:{checked:false},result:null};
      const result={analysis_id:'saved',metrics:['area_px']};
      const current={run:'two',stem:'same'},complete={id:'batch',params:{options:{...options}},
        recordings:[{run:'two',stem:'same',state:'completed',job_id:'job',analysis_id:'saved'}]};
      const receiving={epoch:2,schema:{},batchApplied:new Set(),rows:[selected],history:[],
        e:{data:current,options:{api:async()=>{await new Promise(resolve=>fetched=resolve);return result;}},url(){return 'saved-result';},saveSession(){}},
        options(){return options;},shown:R.shown,historyOptions(){},publish(){applied++;}};
      const stale=R.applyBatch.call(receiving,complete);await tick();receiving.epoch++;receiving.e.data={run:'one',stem:'same'};fetched();await stale;
      assert.equal(applied,0);assert.equal(selected.result,null,'Old result cannot replace a newly selected recording');
      receiving.e.data=current;
      const fresh=R.applyBatch.call(receiving,complete);await tick();fetched();await fresh;
      assert.equal(applied,1);assert.equal(selected.result.analysis_id,'saved');assert(selected.shown.checked);
      const manual={metric:{value:'area_px'},shown:{checked:true},result:{analysis_id:'manual',metrics:['area_px']}};
      selected.shown.checked=false;receiving.rows.push(manual);receiving.epoch++;
      const revisit=R.applyBatch.call(receiving,complete);await tick();fetched();await revisit;
      assert(manual.shown.checked);assert(!selected.shown.checked,'Returning to a source preserves a manually selected result');
      console.log('Batch settings snapshots, combined sources, navigation and saved-result guards passed');
    })().catch(error=>{console.error(error);process.exitCode=1;});
    """
    subprocess.run(["node", "-e", script], cwd=ROOT, check=True, timeout=20)


def test_display_means_do_not_cross_missing_or_decimated_frames():
    script = """
    const c=require('./src/pymicroglia/ui/static/exploration-core.js');
    const input=[1,2,3,null,5,6,7];
    const a=c.smooth(input,[0,1,2,3,4,5,6],3);
    const b=c.smooth([1,2,3],[0,3,6],3);
    const d={run:'a',stem:'b',identities:[19,26],frames:[{},{}]};
    console.log(JSON.stringify({a,b,input,s:c.selection(d,{identity:99,frame:8})}));
    """
    result = json.loads(subprocess.check_output(["node", "-e", script], cwd=ROOT, text=True))
    assert result["a"] == [None, 2, None, None, None, 6, None]
    assert result["b"] == [None, None, None]
    assert result["input"] == [1, 2, 3, None, 5, 6, 7]
    assert result["s"]["identity"] == 19 and result["s"]["frame"] == 1


def test_unique_template():
    from html.parser import HTMLParser
    from pymicroglia.ui.exploration_catalog import VIEWS
    html = (ROOT / "src/pymicroglia/ui/static/exploration.html").read_text(encoding="utf-8")
    class Panels(HTMLParser):
        def __init__(self):
            super().__init__()
            self.panels = []
        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if "data-plot-type" in attrs:
                self.panels.append((attrs["data-plot-type"], attrs.get("data-plot-scope", "inspection"), attrs.get("id")))
    parser = Panels()
    parser.feed(html)
    canonical = [family for family, scope, _ in parser.panels if scope == "inspection"]
    assert len(canonical) == len(set(canonical)) == len(VIEWS)
    assert set(canonical) == {key for key, _, _ in VIEWS}
    # The explicit all-cell comparison complements the selected-cell inspector.
    # It is the only additional home; accidental duplicate panels still fail.
    assert [(family, scope, key) for family, scope, key in parser.panels if scope != "inspection"] == [
        ("time-series", "cell-grid", "xp-trace-grid-panel")]
    assert "Synthetic" not in html and "simulated" not in html.lower()


def test_fit_alignment_keeps_missing_and_mismatched_frames_blank():
    script="""
    const c=require('./src/pymicroglia/ui/static/exploration-core.js');
    console.log(JSON.stringify(c.alignFit({indices:[1,2,4],values:[10,20,40]},
      [0,1,2,3,4],[1,2,null,4,5])));
    """
    actual=json.loads(subprocess.check_output(['node','-e',script],cwd=ROOT,text=True))
    assert actual==[None,10,None,None,40]


def test_overlay_uses_measurement_scale_for_fits_without_changing_inputs():
    script = """
    const c=require('./src/pymicroglia/ui/static/exploration-core.js');
    const raw=[100,null,200,300],fit=[150,null,200,350];
    console.log(JSON.stringify({raw,fit,
      normalized:c.scaleSeries(raw,fit,true),native:c.scaleSeries(raw,fit,false),
      constant:c.scaleSeries([7,7,null],[7,8,null],true),
      missing:c.scaleSeries([null,null],[null,null],true)}));
    """
    result = json.loads(subprocess.check_output(["node", "-e", script], cwd=ROOT, text=True))
    assert result["raw"] == [100, None, 200, 300]
    assert result["fit"] == [150, None, 200, 350]
    assert result["normalized"]["values"] == [0, None, .5, 1]
    assert result["normalized"]["fitted"] == [.25, None, .5, 1.25]
    assert result["native"]["values"] == result["raw"]
    assert result["native"]["fitted"] == result["fit"]
    assert result["constant"]["values"] == [.5, .5, None]
    assert result["constant"]["fitted"] == [.5, 1.5, None]
    assert result["missing"]["values"] == [None, None]


def test_pending_rhythm_settings_compare_actual_method_test_and_detrending():
    script = """
    const c=require('./src/pymicroglia/ui/static/exploration-core.js');
    const schema={defaults:{fit_method:'fft_nlls',fft_component_test:true,detrend:'robust_linear',
      multiple_testing:'none',period_min_hours:2,period_max_hours:48,rhythmic_alpha:.05,min_observations:24,min_cycles:2},
      tests:[{key:'lomb'},{key:'f'}]};
    const original={metrics:['total','mean']};
    const a=c.rhythmPlan(original,schema),b=c.rhythmPlan({metrics:['mean','total']},schema);
    const changed=c.rhythmPlan({...original,detrend:'polynomial'},schema);
    const separate=c.rhythmPlan({...original,fft_component_test:false,fit_method:'f'},schema);
    console.log(JSON.stringify({a,b,changed,separate,same:c.sameRhythmPlan(a,b),
      dirty:!c.sameRhythmPlan(a,changed),original}));
    """
    result = json.loads(subprocess.check_output(["node", "-e", script], cwd=ROOT, text=True))
    assert result["a"]["test"] == "fft_component"
    assert result["same"] and result["dirty"]
    assert result["separate"]["test"] == "f"
    assert result["original"]["metrics"] == ["total", "mean"]


def test_component_controls_use_declared_baselines_and_keep_legacy_choice():
    script = r"""
    const assert=require('assert'),fs=require('fs'),vm=require('vm');
    const context={window:{},setTimeout,clearTimeout,URLSearchParams};
    let source=fs.readFileSync('./src/pymicroglia/ui/static/exploration.js','utf8');
    source=source.replace('window.PyMicrogliaExploration=', 'window.TestExplorer=Explorer;window.PyMicrogliaExploration=');
    vm.runInNewContext(source,context);
    const schema={recipes:[{test:'fft_component',required:{detrend:'asymmetric_least_squares'},
      compatible_detrends:['asymmetric_least_squares','robust_linear']}],tests:[{key:'lomb',label:'Lomb'}]};
    for(const detrend of ['asymmetric_least_squares','robust_linear','none']){
      const state={'rhythm-method':'fft_nlls','rhythm-detrend':detrend,'rhythm-correction':'none','rhythm-test':'fft_component'};
      const option={value:'fft_component',disabled:false};
      const control={value:'fft_component',options:[option]};
      const e={rhythmSchema:schema,val:key=>state[key],$(key){return control;},updateRhythmSettings(){},saveSession(){}};
      context.window.TestExplorer.prototype.rhythmControlsChanged.call(e);
      assert.equal(option.disabled,detrend==='none');
      assert.equal(control.value,detrend==='none'?'lomb':'fft_component');
      if(detrend!=='none')assert.equal(e.rhythmCompatibilityNotice,'');
    }
    """
    subprocess.run(['node', '-e', script], cwd=ROOT, check=True, capture_output=True, text=True)
