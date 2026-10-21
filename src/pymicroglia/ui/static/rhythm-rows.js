/* Reusable analysis rows and inherited/overridden display cohorts. */
(() => {
  "use strict";
  const copy=value=>JSON.parse(JSON.stringify(value)),C=window.PyMicrogliaExplorationCore;
  const fields=[["rhythm-method","fit_method"],["rhythm-detrend","detrend"],["rhythm-min","period_min_hours",true],["rhythm-max","period_max_hours",true],["rhythm-observations","min_observations",true],["rhythm-cycles","min_cycles",true],["rhythm-alpha","rhythmic_alpha",true],["rhythm-correction","multiple_testing"],["rhythm-normalization","trace_normalization"],["rhythm-view","trace_view"],["rhythm-surrogates","component_surrogates",true],["rhythm-block","component_block_hours",true],["rhythm-seed","component_seed",true]];
  const emptyFilter=()=>({schema_version:1,expression:{op:"OR",children:[]},mode:"exclude",unknown:"keep",applied:null});
  const signature=value=>JSON.stringify(value,(_,item)=>item&&typeof item==='object'&&!Array.isArray(item)?Object.fromEntries(Object.keys(item).sort().map(key=>[key,item[key]])):item);
  class RhythmRows {
    constructor(e){
      this.e=e;this.rows=[];this.metricFilters={};this.plotFilters={};this.plotEditors=new Map();this.displayCache=new Map();this.revision=0;this.epoch=0;
      const toolbar=e.host.querySelector(".xp-rhythm-toolbar"),summary=e.host.querySelector(".xp-rhythm-summary"),settings=e.$("rhythm-settings"),filter=e.$("filter-panel");
      this.filterTemplate=filter.cloneNode(true);this.templates=[toolbar,summary,settings,filter].map(el=>el.cloneNode(true));
      this.container=document.createElement("section");this.container.className="xp-rhythm-rows";this.container.setAttribute("aria-label","Rhythm analyses");toolbar.before(this.container);
      const heading=document.createElement("div");heading.className="xp-rhythm-rows-heading";const title=document.createElement("strong");title.textContent="Rhythm analyses";
      this.addButton=document.createElement("button");this.addButton.type="button";this.addButton.textContent="+ Add analysis";this.addButton.title="Add an independent setup, starting from the current measurement's settings and exclusions";
      this.addButton.onclick=()=>{this.add();e.saveSession();};heading.append(title,this.addButton);this.container.append(heading);
      this.batches=new Map();this.batchApplied=new Set();this.batchReferences=[];
      try{const saved=JSON.parse(localStorage.getItem('pm-rhythm-batches-v1')||'null');if(saved?.version===1)this.batchReferences=saved.batches||[];}catch{}
      this.batchContainer=document.createElement('section');this.batchContainer.className='xp-rhythm-batches';this.batchContainer.setAttribute('aria-label','Analysis across recordings');this.container.append(this.batchContainer);
      const traceSettings=e.$("trace-panel").querySelector(".settings .controls"),gridToolbar=e.$("trace-grid-panel").querySelector(".xp-grid-toolbar");
      const traceValues=document.createElement("select");traceValues.id="xp-trace-values";traceValues.add(new Option("Shown analysis","analysis"));traceValues.add(new Option("Raw measurements","raw"));
      const valueLabel=document.createElement("label");valueLabel.textContent="Trace values ";valueLabel.title="Show saved detrending and normalisation, or original measured values";valueLabel.append(traceValues);traceSettings.append(valueLabel);
      const guideControls=document.createElement("div");guideControls.className="xp-guide-controls";guideControls.setAttribute("role","group");guideControls.setAttribute("aria-label","Time interval guides");gridToolbar.append(guideControls);
      for(const [id,title,value] of [["trace-time-guides","Guides every (h)",""],["trace-time-guides-offset","Guide offset (h)","0"]]){const label=document.createElement("label"),input=document.createElement("input");input.id="xp-"+id;input.type="number";input.step="any";input.value=value;input.placeholder="Off";label.textContent=title+" ";label.title="Vertical dashed lines along recording time; leave interval empty to hide";label.append(input);guideControls.append(label);input.onchange=()=>{const interval=+e.val("trace-time-guides"),span=e.data?e.data.frames.at(-1).hours-e.data.frames[0].hours:0;input.setCustomValidity(interval<0||interval>0&&span/interval>999?"Choose a positive interval with at most 1000 guide lines":"");if(!input.reportValidity())return;e.draw();e.saveSession();};}
      traceValues.onchange=()=>{e.gridPreparedKey=null;e.draw();e.saveSession();};
      this.initial=[toolbar,summary,settings,filter];this.install();
    }
    install(){
      const e=this.e;
      const empty=e.empty.bind(e);e.empty=message=>{++this.epoch;this.schema=null;this.addButton.disabled=true;this.rows.forEach(row=>{clearTimeout(row.timer);row.$("run-rhythms").disabled=true;row.shown.disabled=true;row.$("rhythm-applied").textContent="";row.$("rhythm-pending").hidden=true;});for(const cache of this.displayCache.values())clearTimeout(cache.timer);empty(message);};
      e.loadRhythmOptions=saved=>this.reset(saved);e.updateRhythmSettings=()=>this.rows.forEach(row=>this.describe(row));e.rhythmStatus=()=>e.updateRhythmSettings();
      e.runRhythms=()=>this.rows[0]&&this.run(this.rows[0]);e.selectedRhythmMetrics=()=>this.rows.map(row=>row.metric.value);
      const save=e.saveSession.bind(e);e.saveSession=()=>{save();if(e.session&&e.data&&!e.restoring){e.session.rhythmRows=this.save();try{localStorage.setItem("pm-exploration-session-v1",JSON.stringify(e.session));}catch{}}};
      const methods={drawBarcode:"barcode",drawMap:"tissue-map",drawClock:"clock",drawEvents:"events",drawNetwork:"network",drawDistribution:"distribution",drawTrace:"trace",drawTraceGrid:"trace-grid",drawTraceGridCell:"trace-grid",drawTraceComparison:"trace-grid",drawPeerTrace:"trace",drawImage:"image",drawHistory:"outline",drawChange:"image"};
      for(const [name,view] of Object.entries(methods)){const original=e[name].bind(e);e[name]=(...args)=>this.withCohort(["drawBarcode","drawMap"].includes(name)?args[0]||view:view,()=>original(...args));}
      const series=e.cellTraceSeries.bind(e);e.cellTraceSeries=(cell,indices)=>series(cell,indices).filter(item=>this.allowed(e.plotView||"trace",item.metric,item.identity)).map(item=>this.displaySeries(item));
    }
    save(){return {version:1,rows:this.rows.map(row=>({id:row.id,options:this.options(row),analysis_id:row.result?.analysis_id||row.analysisId||null,shown:row.shown.checked,job_id:row.jobId||null,batch_id:row.batchId||null,advanced:row.$("rhythm-settings").open})),metricFilters:this.metricFilters,plotFilters:this.plotFilters};}
    batchSources(){const e=this.e,design=e.groupComparison?.design;if(design?.recordings.some(row=>row.run===e.data.run&&row.stem===e.data.stem))return design.recordings.map(({run,stem})=>({run,stem}));return e.data.recordings.map(({stem})=>({run:e.data.run,stem}));}
    rememberBatch(batch,previous=null){
      const applied=this.batchReferences.find(row=>row.id===(previous||batch.id))?.applied||[];
      if(previous){this.batches.delete(previous);this.batchReferences=this.batchReferences.filter(row=>row.id!==previous);}
      this.batches.set(batch.id,batch);this.batchReferences=this.batchReferences.filter(row=>row.id!==batch.id);
      this.batchReferences.push({id:batch.id,run:batch.params.run,applied});
      try{localStorage.setItem('pm-rhythm-batches-v1',JSON.stringify({version:1,batches:this.batchReferences}));}catch{}
    }
    async runAll(row){
      if(row.batchSubmitting)return;const options=copy(this.options(row)),sources=this.batchSources();row.batchSubmitting=true;row.error=null;this.describe(row);
      row.$('rhythm-status').textContent='Queuing all loaded recordings…';
      try{const batch=await this.e.options.api('/api/exploration/rhythm-batches',{method:'POST',body:JSON.stringify({run:this.e.groupComparison?.run||this.e.data.run,recordings:sources,options})});this.rememberBatch(batch);row.batchId=batch.id;this.renderBatches();this.pollBatches();}
      catch(error){row.error=error.message;}
      finally{row.batchSubmitting=false;if(this.rows.includes(row))this.describe(row);this.e.saveSession();}
    }
    async resumeBatch(batch){
      try{const next=await this.e.options.api(`/api/exploration/rhythm-batches/${batch.id}/resume`,{method:'POST',body:JSON.stringify({run:batch.params.run})});this.rememberBatch(next,batch.id);this.renderBatches();this.pollBatches();}
      catch(error){batch.displayError=error.message;this.renderBatches();}
    }
    async pollBatches(){
      clearTimeout(this.batchTimer);if(this.batchPolling)return;this.batchPolling=true;
      try{for(const reference of this.batchReferences.slice()){
        const old=this.batches.get(reference.id);if(old&&['completed','failed','interrupted'].includes(old.state)){await this.applyBatch(old);continue;}
        try{const batch=await this.e.options.api(`/api/exploration/rhythm-batches/${reference.id}?`+new URLSearchParams({run:reference.run}));this.batches.set(batch.id,batch);await this.applyBatch(batch);}
        catch(error){if(old)old.displayError=error.message;}
      }}finally{this.batchPolling=false;this.renderBatches();if([...this.batches.values()].some(batch=>['queued','running'].includes(batch.state)))this.batchTimer=setTimeout(()=>this.pollBatches(),1000);}
    }
    async applyBatch(batch){
      const e=this.e,data=e.data,epoch=this.epoch;if(!data||!this.schema)return;
      const group=e.groupComparison;let changed=false,remembered=false;
      const reference=this.batchReferences?.find(row=>row.id===batch.id);
      for(const entry of batch.recordings.filter(entry=>entry.state==='completed')){
        const source=group?.design?.recordings.find(row=>row.run===entry.run&&row.stem===entry.stem);
        const key=JSON.stringify([entry.run,entry.stem,entry.analysis_id,'comparison']);
        if(source&&!this.batchApplied.has(key)&&!reference?.applied?.includes(key)){source.analysis_ids??={};for(const metric of batch.params.options.metrics||[])if(source.analysis_ids[metric]!==entry.analysis_id){source.analysis_ids[metric]=entry.analysis_id;changed=true;}this.batchApplied.add(key);if(reference){reference.applied??=[];reference.applied.push(key);remembered=true;}}
      }
      if(changed){group.rhythmHistories.clear();for(const state of group.states.values())state.key=null;group.changed();group.checkpoint();}
      if(remembered)try{localStorage.setItem('pm-rhythm-batches-v1',JSON.stringify({version:1,batches:this.batchReferences}));}catch{}
      const entry=batch.recordings.find(entry=>entry.run===data.run&&entry.stem===data.stem&&entry.state==='completed');
      if(!entry)return;const key=JSON.stringify([epoch,batch.id,entry.analysis_id]);if(this.batchApplied.has(key))return;
      try{
        const result=await e.options.api(e.url('rhythms',{analysis_id:entry.analysis_id}));if(epoch!==this.epoch||data!==e.data)return;
        const shown=this.shown(batch.params.options.metrics?.[0]),activate=!shown||shown.batchId===batch.id||shown.result?.analysis_id===result.analysis_id;
        let row=this.rows.find(row=>row.batchId===batch.id||signature(this.options(row))===signature(batch.params.options));
        if(!row)row=this.add({options:batch.params.options,shown:true,batch_id:batch.id});if(!row)return;
        // Editing a row during a batch keeps the new draft separate from the completed setup.
        if(signature(this.options(row))!==signature(batch.params.options)){row=this.add({options:batch.params.options,shown:true,batch_id:batch.id});if(!row)return;}
        row.result=result;row.analysisId=result.analysis_id;row.batchId=batch.id;row.error=null;row.shown.checked=activate;
        if(activate)for(const other of this.rows)if(other!==row&&other.metric.value===row.metric.value)other.shown.checked=false;
        if(!this.history.some(saved=>saved.analysis_id===result.analysis_id))this.history.unshift(result);
        this.rows.forEach(row=>this.historyOptions(row));this.batchApplied.add(key);this.publish();e.saveSession();
      }catch(error){batch.displayError=error.message;}
    }
    renderBatches(){
      const source=this.e.data;if(!source)return;const open=new Set([...this.batchContainer.querySelectorAll('details[open]')].map(el=>el.dataset.batch));this.batchContainer.replaceChildren();
      for(const batch of this.batches.values()){
        if(!batch.recordings.some(row=>row.run===source.run&&row.stem===source.stem))continue;
        const details=document.createElement('details'),summary=document.createElement('summary'),list=document.createElement('ul');details.dataset.batch=batch.id;details.open=open.has(batch.id);
        const metric=batch.params.options.metrics?.map(key=>this.schema?.metrics.find(row=>row.key===key)?.label||key).join(', ')||'Rhythms',p=batch.progress;
        summary.textContent=`All recordings · ${metric} · ${p.completed}/${p.total} ready${p.failed?` · ${p.failed} failed`:''}${p.interrupted?` · ${p.interrupted} interrupted`:''}`;details.append(summary,list);
        const settings=document.createElement('p'),options=batch.params.options;settings.textContent=[['methods',options.fit_method],['detrends',options.detrend],['normalizations',options.trace_normalization]].filter(([,key])=>key).map(([catalogue,key])=>this.schema?.[catalogue]?.find(row=>row.key===key)?.label||key.replaceAll('_',' ')).join(' · ');settings.title=`Queued ${new Date(batch.submitted_at).toLocaleString()}`;details.insertBefore(settings,list);
        const duplicate=stem=>batch.recordings.filter(row=>row.stem===stem).length>1;
        for(const entry of batch.recordings){const item=document.createElement('li'),name=document.createElement('span'),status=document.createElement('span');name.textContent=entry.stem+(duplicate(entry.stem)?' · '+entry.run.split(/[\\/]/).at(-1):'');name.title=entry.run;
          const progress=entry.progress;status.textContent=entry.state==='completed'?(entry.reused?'Ready (saved result reused)':'Ready'):entry.state==='running'?`Running${progress?.total?` · ${progress.completed||0}/${progress.total} traces`:''}`:entry.state==='queued'?'Queued':entry.error||entry.state;item.append(name,status);list.append(item);}
        if(['failed','interrupted'].includes(batch.state)){const resume=document.createElement('button');resume.type='button';resume.textContent='Retry unfinished recordings';resume.onclick=()=>this.resumeBatch(batch);details.append(resume);}
        if(batch.displayError){const error=document.createElement('p');error.textContent=batch.displayError;details.append(error);}this.batchContainer.append(details);
      }
      this.rows.forEach(row=>this.describe(row));
    }
    async reset(saved){
      const epoch=++this.epoch,e=this.e,data=e.data;e.phaseAutomatic=false;e.phaseRequest++;
      this.schema=null;
      for(const entry of this.plotEditors.values()){entry.editor.epoch++;entry.editor.applied=null;entry.editor.preview=null;entry.editor.schema=null;entry.metric=null;entry.body.hidden=true;entry.inherit.hidden=true;}
      this.rows.forEach(row=>{clearTimeout(row.timer);++row.editor.epoch;row.host.remove();});this.rows=[];this.displayCache.clear();this.metricFilters=copy(saved?.rhythmRows?.metricFilters||{});this.plotFilters=copy(saved?.rhythmRows?.plotFilters||{});
      for(const view of ["image","outline","trace"])delete this.plotFilters[view];
      this.addButton.disabled=true;
      try{
        const [schema,history]=await Promise.all([e.options.api(e.url("rhythm-options")),e.options.api(e.url("rhythm-history"))]);if(epoch!==this.epoch||data!==e.data)return;
        this.schema=schema;this.history=history.analyses;e.rhythmSchema=schema;
        const attached=data.results?.rhythms?.analysis_id?data.results.rhythms:null;
        const legacyMetrics=saved?.rhythmMetrics?.length?saved.rhythmMetrics:attached?.metrics||schema.selected_metrics;
        const legacy={...schema.defaults,...attached?.resolved_options};for(const [id,key,numeric] of fields){const value=saved?.controls?.["xp-"+id];if(value!==undefined&&value!=="")legacy[key]=numeric?Number(value):value;}
        const legacyTest=saved?.controls?.["xp-rhythm-test"];if(legacyTest){legacy.fft_component_test=legacyTest==="fft_component";legacy.significance_method=legacy.fft_component_test?null:legacyTest;}
        if(!saved?.rhythmRows&&saved?.cellFilters)for(const metric of legacyMetrics)this.metricFilters[metric]={filter:copy(saved.cellFilters),scope:"all"};
        const records=saved?.rhythmRows?.rows?.length?saved.rhythmRows.rows:legacyMetrics.map((metric,index)=>({id:index+1,options:{...legacy,metrics:[metric]},analysis_id:attached?.analysis_id,shown:!!attached&&saved?.controls?.["xp-rhythm-visible"]!==false}));
        for(const record of records.slice(0,16)){
          const row=this.add(record);if(record.analysis_id){try{row.result=attached?.analysis_id===record.analysis_id?attached:await e.options.api(e.url("rhythms",{analysis_id:record.analysis_id}));}catch(error){row.error=error.message;row.shown.checked=false;}}
          if(!this.metricFilters[row.metric.value]&&saved?.cellFilters)this.metricFilters[row.metric.value]={filter:copy(saved.cellFilters),scope:"metric"};
          await this.restoreFilter(row);this.describe(row);
        }
        if(epoch!==this.epoch||data!==e.data)return;
        // At most one saved result is shown per measurement, including migrated sessions.
        const selected=new Set();for(const row of this.rows){if(selected.has(row.metric.value))row.shown.checked=false;if(row.shown.checked&&row.result)selected.add(row.metric.value);}
        this.addButton.disabled=false;this.refreshPlots();this.publish();
        this.refreshStoredOverrides();this.renderBatches();this.pollBatches();
        const jobs=await e.options.api("/api/jobs");if(epoch!==this.epoch)return;
        for(const row of this.rows){const job=jobs.find(job=>job.id===row.jobId);if(job)this.watch(row,job);}
      }catch(error){if(epoch===this.epoch){e.$("rhythm-status").textContent=error.message;console.error(error);}}
    }
    buildRow(id){
      // All rows, including the first, use the same controls, grouping and ID mapping.
      const host=document.createElement("article");host.className="xp-rhythm-row";host.dataset.row=id;host.append(...this.templates.map(el=>el.cloneNode(true)));
      const $=key=>host.querySelector(`[data-original-id="xp-${key}"]`)||host.querySelector(`#xp-${key}`),toolbar=host.querySelector(".xp-rhythm-toolbar");
      $("rhythm-measurements").hidden=true;
      const make=(key,label,type="select",values=[])=>{let input=$(key);if(input)return input;const wrap=document.createElement("label");wrap.textContent=label+" ";input=document.createElement(type==="select"?"select":type==="json"?"textarea":"input");input.id=`xp-${key}`;if(type==="select")input.replaceChildren(...values.map(v=>new Option(v.label,v.key)));else if(type==="json"){input.rows=2;input.value="{}";}else{input.type="number";input.step="any";}wrap.append(input);$("rhythm-settings").querySelector(".controls").append(wrap);return input;};
      const metric=make("analysis-metric","Measurement","select",this.schema.metrics);metric.dataset.rowMetric="true";
      make("rhythm-normalization","Normalisation","select",this.schema.normalizations);make("rhythm-view","Trace display","select",[{key:"detrended",label:"Detrended"},{key:"raw",label:"Raw baseline"}]);
      make("rhythm-normalization-config","Normalisation options (JSON)","json");make("rhythm-surrogates","Component surrogates","number");make("rhythm-block","Surrogate block (h)","number");make("rhythm-seed","Surrogate seed","number");
      make("rhythm-advanced-options","Detrending and method options (JSON)","json");
      make("rhythm-prefilter","Input filtering","select",[{key:"inherit",label:"Measurement default"},...this.schema.input_filters.map(value=>({key:value.key,label:value.label}))]);make("rhythm-median-points","Filter window, samples","number");
      const history=make("rhythm-saved-run","Saved run","select"),scope=make("rhythm-exclusion-scope","Apply exclusions to","select",[{key:"metric",label:"This measurement"},{key:"all",label:"All measurements"}]);
      const group=(className,elements)=>{const wrap=document.createElement("div");wrap.className=className;wrap.append(...elements);return wrap;};
      const configuration=group("xp-rhythm-configuration",["analysis-metric","rhythm-recipe","rhythm-detrend","rhythm-normalization"].map(key=>$(key).closest("label")));
      const runAll=document.createElement('button');runAll.type='button';runAll.textContent='Run all recordings';runAll.title='Run this measurement, method, detrending and normalisation on every loaded recording';
      $("run-rhythms").textContent='Run this recording';
      const runControls=group("xp-rhythm-run-controls",[$("run-rhythms"),runAll,$("rhythm-settings-toggle"),$("rhythm-visible").closest("label")]);
      const remove=document.createElement("button");remove.type="button";remove.textContent="Remove";remove.hidden=id===1;
      const exclusions=group("xp-rhythm-exclusion-controls",[scope.closest("label"),$("filter-toggle"),$("filter-count"),group("xp-rhythm-remove-slot",[remove])]);
      const job=group("xp-rhythm-job",[$("rhythm-progress"),$("rhythm-status")]);host.querySelector(".xp-rhythm-summary").append(job);
      toolbar.replaceChildren(configuration,group("xp-rhythm-actions",[runControls,exclusions]),$("rhythm-measurements"));
      for(const key of ["rhythm-pending","rhythm-use-shown","rhythm-status","rhythm-progress","filter-count","filter-status"])$(key).classList.add("xp-"+key);
      const ids=new Map();for(const el of host.querySelectorAll("[id]")){const original=el.id;el.dataset.originalId=original;el.id=original==="xp-analysis-metric"?`xp-analysis-metric-${id}`:id===1?original:`${original}-analysis-${id}`;ids.set(original,el.id);}
      for(const el of host.querySelectorAll("[aria-controls],[aria-labelledby],[aria-describedby],[for]"))for(const key of ["aria-controls","aria-labelledby","aria-describedby","for"]){const value=el.getAttribute(key);if(value)el.setAttribute(key,value.split(/\s+/).map(ref=>ids.get(ref)||ref).join(" "));}
      if(this.initial){++this.e.filters.epoch;this.initial.forEach(el=>el.remove());this.initial=null;}
      this.container.append(host);return {id,host,$,metric,scope,history,remove,runAll};
    }
    add(record=null){
      if(!this.schema)return null;const e=this.e,id=record?.id||Math.max(0,...this.rows.map(row=>row.id))+1;if(this.rows.length>=16)return null;
      const row={...this.buildRow(id),result:null,analysisId:record?.analysis_id,error:null,jobId:record?.job_id,batchId:record?.batch_id},{$,host,metric}=row;
      this.rows.push(row);
      const current=this.metricFor("trace"),parent=this.shown(current)||this.rows.find(other=>other!==row);
      const initial=record?.options||{...this.schema.defaults,...(parent?this.options(parent):{}),metrics:[current||this.schema.selected_metrics[0]]};metric.value=initial.metrics[0];if(!metric.value)metric.value=this.schema.selected_metrics[0];
      this.historyOptions(row);
      row.history.onchange=async()=>{if(!row.history.value)return;const epoch=this.epoch;try{const result=await e.options.api(e.url("rhythms",{analysis_id:row.history.value}));if(epoch!==this.epoch)return;row.result=result;row.analysisId=result.analysis_id;row.error=null;this.publish();}catch(error){row.error=error.message;this.describe(row);}};
      $("filter-toggle").textContent="Exclusions";
      $("filter-panel").querySelector('[id^="xp-filter-apply"]').title="Apply exclusions to displays; saved test populations remain unchanged";
      row.shown=$("rhythm-visible");row.shown.closest("label").lastChild.textContent=" Show this result in plots";row.shown.checked=record?.shown??false;
      row.facade=Object.create(e);Object.assign(row.facade,{host,data:{...e.data,results:{rhythms:null}},rhythmSchema:this.schema,$,saveSession:()=>e.saveSession(),selectedRhythmMetrics:()=>[metric.value],setRhythmMetrics:(_,chosen)=>metric.value=chosen[0],updateRhythmSettings:()=>this.describe(row)});
      for(const [key,choices] of [["rhythm-method",this.schema.methods],["rhythm-test",this.schema.tests],["rhythm-detrend",this.schema.detrends],["rhythm-recipe",[...this.schema.recipes,{key:"custom",label:"Custom method / test"}]]])$(key).replaceChildren(...choices.map(v=>new Option(v.label,v.key)));
      this.setForm(row,initial);$("rhythm-settings").open=!!record?.advanced;
      $("rhythm-settings-toggle").onclick=()=>{$("rhythm-settings").open=!$("rhythm-settings").open;$("rhythm-settings-toggle").setAttribute("aria-expanded",String($("rhythm-settings").open));e.saveSession();};
      $("rhythm-use-shown").onclick=()=>{if(row.result)this.setForm(row,row.result.resolved_options);this.describe(row);e.saveSession();};
      $("rhythm-recipe").onchange=()=>{const recipe=this.schema.recipes.find(r=>r.key===$("rhythm-recipe").value);if(recipe){$("rhythm-method").value=recipe.fit_method;$("rhythm-test").value=recipe.test;for(const [key,value] of Object.entries(recipe.required)){const f=fields.find(f=>f[1]===key);if(f)$(f[0]).value=value;}}else $("rhythm-settings").open=true;row.facade.rhythmControlsChanged();};
      for(const key of ["rhythm-test","rhythm-normalization-config","rhythm-advanced-options","rhythm-prefilter","rhythm-median-points",...fields.map(f=>f[0])])$(key).onchange=()=>row.facade.rhythmControlsChanged();
      row.editor=new window.PyMicrogliaCellFilters(e,{element:key=>$("filter-"+key)});if(id===1)e.filters=row.editor;
      row.editor.context={element:key=>$("filter-"+key),metric:()=>metric.value,analysisIds:()=>this.filterAnalysisIds(),onDraft:filter=>{this.metricFilters[metric.value]??={scope:row.scope.value,identities:null};this.metricFilters[metric.value].filter=filter;},onApply:result=>{this.metricFilters[metric.value]={filter:row.editor.save(),scope:row.scope.value,identities:result?.identities||null};if(!row.restoringFilter){this.redraw();this.syncEditors(row);}}};
      row.scope.onchange=()=>{const state=this.metricFilters[metric.value];if(state){state.scope=row.scope.value;this.redraw();}e.saveSession();};
      metric.onchange=async()=>{row.shown.checked=false;if(!this.metricFilters[metric.value])this.metricFilters[metric.value]={filter:copy(row.editor.save()),scope:"metric"};await this.restoreFilter(row);this.historyOptions(row);this.describe(row);this.publish();e.saveSession();};
      row.shown.onchange=()=>{if(row.shown.checked&&row.result)for(const other of this.rows)if(other!==row&&other.result?.metrics.includes(metric.value))other.shown.checked=false;this.publish();};
      $("run-rhythms").onclick=()=>this.run(row);
      row.runAll.onclick=()=>this.runAll(row);
      row.remove.onclick=()=>{clearTimeout(row.timer);++row.editor.epoch;this.rows=this.rows.filter(r=>r!==row);host.remove();this.publish();};
      if(!this.metricFilters[metric.value])this.metricFilters[metric.value]={filter:copy(parent?.editor.save()||emptyFilter()),scope:"metric"};
      if(!record)this.restoreFilter(row);this.describe(row);return row;
    }
    options(row){let extra;try{extra=JSON.parse(row.$("rhythm-advanced-options").value);}catch{extra={period_config:null};}const options={...extra,metrics:[row.metric.value]};for(const [id,key,numeric] of fields)options[key]=numeric?Number(row.$(id).value):row.$(id).value;const filter=row.$("rhythm-prefilter").value;options.trace_filter=filter==="inherit"?null:filter==="none"?{method:"none"}:{method:filter,window_points:Number(row.$("rhythm-median-points").value)};options.fft_component_test=row.$("rhythm-test").value==="fft_component";options.significance_method=options.fft_component_test?null:row.$("rhythm-test").value;try{options.trace_normalization_config=JSON.parse(row.$("rhythm-normalization-config").value);}catch{options.trace_normalization_config=null;}return options;}
    setForm(row,options){const config={...this.schema.defaults,...options};for(const [id,key] of fields)row.$(id).value=config[key];row.$("rhythm-test").value=config.fft_component_test?"fft_component":config.significance_method||"lomb";row.$("rhythm-normalization-config").value=JSON.stringify(config.trace_normalization_config||{});row.$("rhythm-prefilter").value=config.trace_filter?.method||"inherit";row.$("rhythm-median-points").value=config.trace_filter?.window_points||3;row.$("rhythm-advanced-options").value=JSON.stringify(Object.fromEntries((this.schema.advanced_option_names||[]).filter(key=>key!=="trace_filter").map(key=>[key,config[key]])),null,2);}
    describe(row){
      row.$("rhythm-settings-toggle").setAttribute("aria-expanded",String(row.$("rhythm-settings").open));
      const inputLabel=options=>options.trace_filter&&options.trace_filter.method!=="none"?`${this.schema.input_filters.find(value=>value.key===options.trace_filter.method)?.label||options.trace_filter.method} input (${options.trace_filter.window_points} samples)`:options.trace_filter?.method==="none"?"Raw measured input":"Measurement default input";
      const draft=this.options(row),result=row.result,valid=result?.metrics.includes(row.metric.value),plan=C.rhythmPlan(draft,this.schema),recipe=this.schema.recipes.find(r=>r.fit_method===plan.fit_method&&r.test===plan.test);row.$("rhythm-recipe").value=recipe?.key||"custom";
      const label=this.schema.metrics.find(m=>m.key===row.metric.value)?.label||row.metric.value;
      const rows=valid?result.rows.filter(r=>r.metric===row.metric.value):[],rhythmic=rows.filter(r=>r.status==="rhythmic").length,tested=rows.filter(r=>["rhythmic","not rhythmic","arrhythmic"].includes(r.status)).length;
      row.$("rhythm-applied").textContent=valid?`${row.shown.checked?"Shown":"Completed, hidden"}: ${label} · ${rhythmic} / ${rows.length} rhythmic cells (${tested} tested) · ${row.facade.rhythmPlanText(C.rhythmPlan(result.resolved_options,this.schema))} · ${inputLabel(result.resolved_options)} ? ${result.settings.trace_normalization} · ${new Date(result.created_at).toLocaleString()}`:"No completed analysis for this measurement";
      const dirty=!valid||Object.entries(draft).some(([key,value])=>JSON.stringify(value)!==JSON.stringify(key==="metrics"?[row.metric.value]:key in result.resolved_options?result.resolved_options[key]:this.schema.defaults[key]));
      row.$("rhythm-pending").hidden=!dirty;row.$("rhythm-pending").textContent=`Draft: ${row.facade.rhythmPlanText(plan)} · ${inputLabel(draft)} ? ${draft.trace_normalization}${row.facade.rhythmCompatibilityNotice?" · "+row.facade.rhythmCompatibilityNotice:""}. Press Run to apply.`;
      row.$("rhythm-use-shown").hidden=!valid||!dirty;row.shown.disabled=!valid;row.$("run-rhythms").disabled=!!row.running;
      if(row.runAll){const active=[...this.batches.values()].some(batch=>['queued','running'].includes(batch.state)&&signature(batch.params.options)===signature(draft)&&batch.recordings.some(entry=>entry.run===this.e.data.run&&entry.stem===this.e.data.stem));row.runAll.disabled=!!row.batchSubmitting||active;row.$('run-rhythms').disabled=!!row.running||!!row.batchSubmitting||active;}
      if(row.history)row.history.value=row.result?.analysis_id||"";
      if(!row.running){row.$("rhythm-progress").hidden=true;row.$("rhythm-status").textContent=row.error||"";}
    }
    async run(row){
      if(row.running)return;const e=this.e,data=e.data,epoch=this.epoch,options=this.options(row);row.running=true;row.error=null;this.describe(row);row.$("rhythm-status").textContent="Queuing analysis…";
      try{const job=await e.options.api("/api/exploration/rhythm-jobs",{method:"POST",body:JSON.stringify({run:data.run,stem:data.stem,options})});if(epoch!==this.epoch)return;row.jobId=job.id;e.saveSession();this.watch(row,job);}
      catch(error){if(epoch===this.epoch){row.running=false;row.error=error.message;this.describe(row);}}
    }
    async watch(row,job){
      clearTimeout(row.timer);const epoch=this.epoch;row.running=true;row.facade.data=this.e.data;row.facade.showRhythmProgress(job);row.$("run-rhythms").disabled=true;
      if(job.state==="completed"||job.state==="failed"){
        row.running=false;row.jobId=null;
        if(job.state==="failed")row.error=job.error;
        else try{const result=await this.e.options.api(this.e.url("rhythms",{analysis_id:job.analysis_id||job.id}));if(epoch!==this.epoch||!this.rows.includes(row))return;row.result=result;row.analysisId=result.analysis_id;
          if(result.metrics.includes(row.metric.value)){row.shown.checked=true;for(const other of this.rows)if(other!==row&&other.metric.value===row.metric.value)other.shown.checked=false;}
          row.error=null;this.history.unshift({analysis_id:result.analysis_id,metrics:result.metrics,created_at:result.created_at,resolved_options:result.resolved_options});this.rows.forEach(row=>this.historyOptions(row));this.publish();
        }catch(error){row.error=error.message;}
        this.describe(row);this.e.saveSession();return;
      }
      row.timer=setTimeout(async()=>{try{const next=await this.e.options.api(`/api/jobs/${job.id}`);if(epoch===this.epoch&&this.rows.includes(row))this.watch(row,next);}catch(error){if(epoch===this.epoch){row.$("rhythm-status").textContent=`Progress unavailable; saved result retained. ${error.message}`;row.timer=setTimeout(()=>this.watch(row,job),2000);}}},1000);
    }
    shown(metric){return this.rows.find(row=>row.shown?.checked&&row.result?.metrics.includes(metric)&&row.metric.value===metric);}
    result(metric){return this.shown(metric)?.result||null;}
    historyOptions(row){if(!row.history)return;const selected=row.result?.analysis_id||"";row.history.replaceChildren(new Option("Choose a completed run", ""),...[...new Map(this.history.filter(result=>result.metrics.includes(row.metric.value)).map(result=>[result.analysis_id,result])).values()].map(result=>new Option(`${new Date(result.created_at).toLocaleString()} · ${result.resolved_options.fit_method} · ${result.resolved_options.detrend} · ${result.resolved_options.trace_normalization||"minmax"}`,result.analysis_id)));row.history.value=selected;}
    traceMetric(metric){return metric==="integrated_density"&&this.e.val("trace-smoothing")==="3"&&this.shown("integrated_density_mean3")?"integrated_density_mean3":metric;}
    analysisIds(){return Object.fromEntries((this.e.sourceData?.metrics||[]).map(m=>[m.key,this.result(m.key)?.analysis_id||null]));}
    filterAnalysisIds(){return Object.fromEntries((this.e.sourceData?.metrics||[]).map(m=>[m.key,(this.result(m.key)||this.rows.filter(row=>row.metric.value===m.key&&row.result?.metrics.includes(m.key)).sort((a,b)=>b.result.created_at.localeCompare(a.result.created_at))[0]?.result)?.analysis_id||null]));}
    async refreshStoredOverrides(){const epoch=this.epoch,source=this.e.sourceData;for(const [view,states] of Object.entries(this.plotFilters))for(const [metric,state] of Object.entries(states)){if(!state.enabled||!state.filter?.applied)continue;try{const result=await this.e.options.api("/api/exploration/filter-preview",{method:"POST",body:JSON.stringify({run:source.run,stem:source.stem,...state.filter.applied,analysis_ids:this.filterAnalysisIds()})});if(epoch!==this.epoch)return;if(this.plotFilters[view]?.[metric]!==state)continue;state.identities=result.identities;this.redraw();}catch(error){if(epoch===this.epoch)state.error=error.message;}}}
    publish(){
      const e=this.e;if(!e.data)return;const metrics=this.rows.filter(row=>row.shown.checked&&row.result?.metrics.includes(row.metric.value)).map(row=>row.metric.value),rows=metrics.flatMap(metric=>this.result(metric).rows.filter(r=>r.metric===metric));
      e.data.results.rhythms={rows,metrics,preview:false,analysis_id:JSON.stringify(this.analysisIds()),meaning:"Shown completed analyses; each measurement uses its selected saved run.",source:"Selected analysis rows"};e.rhythmIndexSource=null;e.phaseAutomatic=false;e.phaseRequest++;
      for(const [key,choices] of [["clock-metric",metrics],["tectonics-metric",metrics]]){const old=e.val(key);e.setOptions(key,choices.map(metric=>[metric,e.data.metrics.find(m=>m.key===metric)?.label||metric]),old);}
      for(const key of ["map-layer","barcode-order"]){const select=e.$(key);for(const option of [...select.options])if(option.value.startsWith("phase:"))option.remove();for(const metric of metrics)select.add(new Option("Tested phase · "+(e.data.metrics.find(m=>m.key===metric)?.label||metric),"phase:"+metric));}
      e.comparisons.refresh();this.rows.forEach(row=>this.describe(row));this.rows.forEach(row=>row.editor.refreshApplied());this.plotEditors.forEach(editor=>editor.editor.refreshApplied());e.outlineFilters?.refreshApplied();this.redraw();
    }
    async restoreFilter(row){const state=this.metricFilters[row.metric.value];row.scope.value=state?.scope||"metric";row.restoringFilter=true;try{await row.editor.reset(state?.filter||emptyFilter());}finally{row.restoringFilter=false;}}
    syncEditors(source){for(const row of this.rows)if(row!==source&&row.metric.value===source.metric.value)this.restoreFilter(row);}
    metricFor(view){const e=this.e,base=e.comparisons.base(view);if(base==="barcode"){const metric=C.metricKey(e.comparisons.value("barcode-metric",view),e.comparisons.value("barcode-aggregation",view),e.data);return metric==="integrated_density"&&e.comparisons.value("barcode-smoothing",view)==="3"&&this.shown("integrated_density_mean3")?"integrated_density_mean3":metric;}if(base==="tissue-map")return e.comparisons.value("tectonics-metric",view)||this.metricFor("trace");if(view==="clock")return e.val("clock-metric")||this.metricFor("trace");if(base==="distribution")return C.metricKey(e.comparisons.value("distribution-metric",view),e.comparisons.value("distribution-aggregation",view),e.data);return this.traceMetric(C.metricKey(e.val("trace-metric"),e.val("trace-aggregation"),e.data));}
    allowed(view,metric,identity){
      if(["trace","trace-grid"].includes(view))metric=this.traceMetric(metric);
      const override=this.plotFilters[view]?.[metric];if(override?.enabled)return !override.identities||override.identities.includes(identity);
      const local=this.metricFilters[metric];if(local?.identities&&!local.identities.includes(identity))return false;
      return !Object.values(this.metricFilters).some(state=>state.scope==="all"&&state.identities&&!state.identities.includes(identity));
    }
    withCohort(view,draw){
      const e=this.e;if(!e.data||!this.schema)return draw();const metric=this.metricFor(view),data=e.data,original={cells:data.cells,identities:data.identities,cohort:e.cohortIds,view:e.plotView};
      const editor=this.plotEditors.get(view);if(editor?.metric&&editor.metric!==metric){editor.editor.epoch++;editor.metric=null;editor.editor.applied=null;editor.body.hidden=true;editor.inherit.hidden=true;}this.refreshPlotLabel(view);
      const metrics=["trace","trace-grid"].includes(view)?e.traceMetrics():[metric],ids=e.sourceData.identities.filter(id=>metrics.some(metric=>this.allowed(view,metric,id)));e.plotView=view;e.cohortIds=new Set(ids);data.identities=ids;data.cells=e.sourceData.cells.filter(cell=>e.cohortIds.has(cell.identity));
      try{if(!ids.includes(e.state.identity)&&(view==="outline"||view==="image"&&e.val("image-mode")==="cell")){for(const canvas of view==="outline"?["outline-history"]:["cell-image","cell-change"])e.blank(e.$(canvas),"Selected cell is excluded from this plot");return;}return draw();}finally{data.cells=original.cells;data.identities=original.identities;e.cohortIds=original.cohort;e.plotView=original.view;}
    }
    redraw(){this.revision++;const e=this.e;e.gridPreparedKey=null;e.draw();e.saveSession();this.refreshPlotLabels();}
    refreshPlots(){
      const e=this.e;for(const panel of e.host.querySelectorAll("article.panel")){const view=[...e.comparisons.views.values()].find(v=>v.panel===panel)?.id||({"xp-trace-panel":"trace","xp-trace-grid-panel":"trace-grid","xp-events-panel":"events","xp-clock-panel":"clock","xp-map-panel":"tissue-map","xp-network-panel":"network","xp-distribution-panel":"distribution","xp-image-panel":"image","xp-outline-panel":"outline"})[panel.id];if(!view||["image","outline","trace"].includes(view)||this.plotEditors.has(view)&&this.plotEditors.get(view).panel===panel)continue;
        const editorHost=document.createElement("section"),toggle=document.createElement("button"),count=document.createElement("span"),body=this.filterTemplate.cloneNode(true),inherit=document.createElement("button");toggle.type="button";toggle.textContent="Exclusions";toggle.title="Use the main measurement exclusions, or apply different rules to this plot";count.className="xp-plot-cohort";inherit.type="button";inherit.textContent="Use main exclusions";editorHost.className="xp-plot-exclusions";
        const mapping=new Map();body.querySelectorAll("[id]").forEach(el=>{const key=el.id.replace("xp-filter-","");el.id=`xp-plot-filter-${view}-${key}`;mapping.set(key,el);});body.id=`xp-plot-filter-${view}-panel`;mapping.set("panel",body);mapping.set("toggle",toggle);mapping.set("count",count);editorHost.append(inherit,body);e.plotHeader(panel).actions.append(toggle,count);panel.append(editorHost);body.hidden=true;inherit.hidden=true;
        const sourceNote=document.createElement("span");sourceNote.className="xp-analysis-source";panel.append(sourceNote);
        const entry={panel,toggle,count,body,inherit,sourceNote,metric:null};const editor=new window.PyMicrogliaCellFilters(e,{element:key=>mapping.get(key),metric:()=>entry.metric||this.metricFor(view),analysisIds:()=>this.filterAnalysisIds(),onApply:result=>{if(entry.initializing)return;const metric=entry.metric||this.metricFor(view);this.plotFilters[view]??={};this.plotFilters[view][metric]={enabled:true,identities:result?.identities||null,filter:editor.save()};this.redraw();}});entry.editor=editor;this.plotEditors.set(view,entry);
        toggle.onclick=async()=>{const metric=this.metricFor(view);if(entry.metric!==metric){entry.metric=metric;const override=this.plotFilters[view]?.[metric];entry.initializing=true;try{await editor.reset(copy(override?.filter||this.metricFilters[metric]?.filter||emptyFilter()));}finally{entry.initializing=false;}if(!override?.enabled){editor.applied=null;editor.savedApplied=null;}body.hidden=false;await editor.load();}else{body.hidden=!body.hidden;if(!body.hidden&&!editor.schema)await editor.load();}inherit.hidden=body.hidden;toggle.setAttribute("aria-expanded",String(!body.hidden));this.refreshPlotLabels();};
        inherit.onclick=()=>{const metric=this.metricFor(view);delete this.plotFilters[view]?.[metric];entry.metric=null;body.hidden=true;inherit.hidden=true;this.redraw();};
      }
      this.refreshPlotLabels();
    }
    refreshPlotLabels(){if(!this.e.data)return;for(const view of this.plotEditors.keys())this.refreshPlotLabel(view);}
    refreshPlotLabel(view){const entry=this.plotEditors.get(view);if(!entry||!this.e.data)return;const metric=this.metricFor(view),label=this.e.data.metrics.find(m=>m.key===metric)?.label||metric,ids=this.e.sourceData.identities,kept=ids.filter(id=>this.allowed(view,metric,id)).length,override=this.plotFilters[view]?.[metric]?.enabled,result=this.result(metric);const comparisonState=this.e.groupComparison?.states.get(view),comparison=comparisonState?.preview?.counts||comparisonState?.lastCounts;entry.count.textContent=`${override?"Plot override":"Main"}: ${comparison?comparison.cells:kept}/${comparison?comparison.total_cells:ids.length}`;entry.toggle.title=`${label} exclusions. ${override?"This plot overrides the main rules.":"Inherited from the main measurement rules."}`;const traceDisplay=["trace","trace-grid"].includes(view)?(this.e.val("trace-values")==="raw"?" Trace display: original measured values.":this.displayCache.get(result?.analysis_id+":"+metric)?.message?" Trace display: raw fallback; "+this.displayCache.get(result?.analysis_id+":"+metric).message:" Trace display: saved preprocessing when available."):"";entry.sourceNote.textContent=(result?`Rhythm source: ${label} · ${new Date(result.created_at).toLocaleString()} · ${result.resolved_options.fit_method} · ${result.resolved_options.detrend} · ${result.resolved_options.trace_normalization}. Exclusions change the display; ${result.rows.filter(r=>r.metric===metric).length} cells were analysed.`:`${label}: no completed rhythm result selected.`)+traceDisplay;}
    displaySeries(item){
      const e=this.e;if(e.val("trace-values")==="raw")return item;const metric=this.traceMetric(item.metric),result=this.result(metric);if(!result)return item;
      const key=result.analysis_id+":"+metric;let cache=this.displayCache.get(key);if(!cache){cache={cells:new Map(),pending:new Set(),failed:new Set(),queued:new Set()};this.displayCache.set(key,cache);}
      const saved=cache.cells.get(item.identity);if(saved){const positions=new Map(saved.indices.map((index,i)=>[index,i]));return {...item,metric,values:item.indices.map(index=>saved.values[positions.get(index)]??null),fitted:item.indices.map(index=>saved.fitted[positions.get(index)]??null),label:(e.data.metrics.find(m=>m.key===metric)?.label||item.label)+` · ${result.settings.trace_view}, ${result.settings.trace_normalization}`,analysis_id:result.analysis_id};}
      if(cache.failed.has(item.identity))return {...item,label:item.label+" · Raw fallback: "+(cache.error||cache.message||"saved display trace unavailable")};
      if(!cache.pending.has(item.identity)){cache.queued.add(item.identity);if(!cache.timer)cache.timer=setTimeout(()=>this.fetchDisplay(result,metric,cache),0);}
      return {...item,values:item.values.map(()=>null),label:item.label+" · Loading shown analysis trace"};
    }
    async fetchDisplay(result,metric,cache){
      cache.timer=null;const ids=[...cache.queued].slice(0,128);ids.forEach(id=>{cache.queued.delete(id);cache.pending.add(id);});if(!ids.length)return;const epoch=this.epoch;
      try{const response=await this.e.options.api(this.e.url("rhythm-traces",{metric,analysis_id:result.analysis_id,identities:ids.join(",")}));if(epoch!==this.epoch)return;cache.message=response.message;response.cells.forEach(cell=>cache.cells.set(cell.identity,cell));ids.forEach(id=>{if(!cache.cells.has(id))cache.failed.add(id);});}
      catch(error){ids.forEach(id=>cache.failed.add(id));cache.error=error.message;}
      finally{ids.forEach(id=>cache.pending.delete(id));if(epoch===this.epoch){this.e.gridPreparedKey=null;this.e.drawTrace();this.e.drawCellCards();this.e.drawTraceGrid();if(cache.queued.size)cache.timer=setTimeout(()=>this.fetchDisplay(result,metric,cache),0);}}
    }
  }
  window.PyMicrogliaRhythmRows=RhythmRows;
})();
