/* Selected-cell comparison page. Scientific calculations live in the backend. */
(() => {
  "use strict";
  const C=window.PyMicrogliaExplorationCore,copy=value=>JSON.parse(JSON.stringify(value));
  const esc=value=>String(value??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[c]);
  const num=value=>C.finite(value)?Number(value.toPrecision(4)).toString():"Unavailable";
  const KEY="pm-rhythm-audit-v1";
  class RhythmAudit {
    constructor(host,options){
      this.host=host;this.options=options;this.epoch=0;this.data=null;this.result=null;this.running=false;this.cursor=null;this.styles={};this.folds={};this.view={};
      try{this.saved=JSON.parse(localStorage.getItem(KEY)||"null");}catch{}
      this.styles=this.saved?.styles||{};this.folds=this.saved?.folds||{};this.view=this.saved?.view||{};
      this.host.classList.add("xp-workbench","ra-page");this.build();
      new ResizeObserver(()=>{if(this.visible)this.draw();}).observe(host);
      new MutationObserver(()=>{if(this.visible)this.draw();}).observe(document.documentElement,{attributes:true,attributeFilter:["data-pm-theme"]});
    }
    $(id){return this.host.querySelector(`#ra-${id}`);}
    build(){this.host.innerHTML=`
      <div class="ra-heading"><h1>Rhythm audit</h1><span id="ra-status" role="status" aria-live="polite">Choose a saved recording.</span></div>
      <div class="controls ra-selection"><label>Recording <select id="ra-recording" title="Choose a saved recording"></select></label><label>Cell <select id="ra-cell" title="Choose one saved cell, including excluded cells"></select></label><label>Measurement <select id="ra-metric"></select></label><button id="ra-run" type="button" disabled>Run audit</button><button id="ra-cancel" type="button" hidden title="Stop after the current comparison completes">Cancel audit</button></div>
      <div class="ra-state"><span id="ra-exclusion"></span><span id="ra-pending" role="status"></span><progress id="ra-progress" hidden></progress></div>
      <details class="settings ra-comparison-settings" id="ra-config" open><summary>Comparison settings</summary><div class="ra-choices"><fieldset id="ra-detrends"><legend>Detrending rows</legend></fieldset><fieldset id="ra-methods"><legend>Period-estimator columns</legend></fieldset></div>
        <div class="controls"><label>Rhythmicity test <select id="ra-test" title="This same test supplies rhythmicity evidence for each estimator column"></select></label><label>Normalisation (display) <select id="ra-normalization" title="Scale the displayed trace; estimation and testing use unnormalised inputs"></select></label></div>
        <fieldset class="ra-filter-settings"><legend>Additional input filters</legend><div class="controls"><span>None (unfiltered)</span><button id="ra-add-filter" type="button" title="Add a filter and window length to compare">+ Add filter</button></div><div id="ra-input-filters"></div></fieldset>
        <details id="ra-advanced"><summary>Advanced analysis settings</summary><div class="controls" id="ra-advanced-controls"></div><label class="ra-json">Normalisation options (JSON)<textarea id="ra-normalization-config" rows="2">{}</textarea></label></details>
      </details>
      <div class="controls ra-history"><label>Completed audit <select id="ra-history"><option value="">Choose a completed audit</option></select></label><button id="ra-use-shown" type="button" disabled>Restore shown settings</button><span id="ra-shown"></span></div>
      <div id="ra-results"><p id="ra-empty">Choose a cell and measurement, then press Run audit.</p></div>`;
      this.$("add-filter").onclick=()=>{this.addFilter({method:"median",window_points:3});this.changed();};
      this.$("run").onclick=()=>this.run();this.$("cancel").onclick=()=>this.cancel();
      this.$("recording").onchange=()=>this.options.onRecording(this.$("recording").value);
      this.$("history").onchange=()=>{if(this.$("history").value)this.loadResult(this.$("history").value);};
      this.$("use-shown").onclick=()=>{if(this.result){this.setForm({...this.result.options,input_filters:this.filtersForResult().map(value=>value.recipe)});this.changed();}};
      this.host.addEventListener("change",event=>{if(event.target.closest("#ra-config,.ra-selection"))this.changed();});
      this.host.addEventListener("toggle",event=>{if(event.target.id)this.save();},true);
    }
    async open(data,selection=null){
      this.visible=true;this.host.hidden=false;
      if(!data){this.data=null;this.$("run").disabled=true;this.$("status").textContent="Choose a saved measurement run.";return;}
      const same=this.data?.run===data.run&&this.data?.stem===data.stem;
      if(same&&this.schema){if(selection){this.setSelection(selection);this.changed();}this.draw();return;}
      const epoch=++this.epoch;clearTimeout(this.timer);this.running=false;this.jobId=null;this.result=null;this.schema=null;this.data=data;
      this.build();
      this.$("status").textContent="Loading audit settings…";this.$("run").disabled=true;
      this.$("cancel").hidden=true;this.$("progress").hidden=true;this.$("use-shown").disabled=true;this.$("shown").textContent="";
      this.$("results").innerHTML='<p id="ra-empty">Choose a cell and measurement, then press Run audit.</p>';
      try{
        const [schema,history]=await Promise.all([this.options.api(this.url("audit-options")),this.options.api(this.url("audit-history"))]);
        if(epoch!==this.epoch)return;const stored=this.saved?.run===data.run&&this.saved?.stem===data.stem?copy(this.saved):null;this.schema=schema;this.history=history.audits;
        this.menu("recording",data.recordings.map(row=>({key:row.stem,label:row.stem})),data.stem);
        this.menu("cell",schema.identities.map(id=>({key:id,label:`Cell ${id}`})),selection?.identity||this.options.explorer()?.state.identity);
        this.menu("metric",schema.metrics,selection?.metric||schema.selected_metrics[0]);
        this.menu("test",schema.tests.filter(row=>row.key!=="fft_component"),"lomb");this.menu("normalization",schema.normalizations,"none");
        this.checklist("detrends",schema.detrends,schema.audit_defaults.detrend_methods);this.checklist("methods",schema.methods,schema.audit_defaults.period_methods);
        this.advanced(schema);this.installSharedControls();
        this.setForm({...schema.defaults,...schema.audit_defaults,...stored?.options,identity:Number(this.$("cell").value),metric:this.$("metric").value});
        if(selection)this.setSelection(selection);else if(stored?.options)this.setForm(stored.options);
        this.historyMenu();this.$("run").disabled=false;this.$("status").textContent="Ready";this.changed();
        if(stored?.audit_id)await this.loadResult(stored.audit_id);
        if(epoch!==this.epoch)return;
        const jobs=await this.options.api("/api/jobs");if(epoch!==this.epoch)return;
        const job=jobs.find(job=>job.id===stored?.job_id)||jobs.find(job=>job.kind==="rhythm_audit"&&["queued","running"].includes(job.state)&&job.params.run===data.run&&job.params.stem===data.stem);
        if(job)this.watch(job);
      }catch(error){if(epoch===this.epoch){this.$("status").textContent=error.message;this.$("run").disabled=!this.schema;}}
    }
    close(){this.visible=false;this.host.hidden=true;}
    url(endpoint,extra={}){return `/api/exploration/${endpoint}?${new URLSearchParams({run:this.data.run,stem:this.data.stem,...extra})}`;}
    menu(id,rows,value){this.$(id).replaceChildren(...rows.map(row=>new Option(row.label,row.key)));if(value!=null)this.$(id).value=String(value);if(!this.$(id).value)this.$(id).selectedIndex=0;}
    checklist(id,rows,values){const host=this.$(id);host.querySelectorAll("label").forEach(el=>el.remove());for(const row of rows){const label=document.createElement("label"),input=document.createElement("input");input.type="checkbox";input.value=row.key;input.checked=values.includes(row.key);label.title=`Include ${row.label} in this audit`;label.append(input,document.createTextNode(row.label));host.append(label);}}
    advanced(schema){
      const host=this.$("advanced-controls");host.replaceChildren();this.advancedFields=[];
      const fixed=[...["period_min_hours","period_max_hours","min_observations","min_cycles","rhythmic_alpha","multiple_testing"]];
      const labels={period_min_hours:"Minimum period (h)",period_max_hours:"Maximum period (h)",min_observations:"Minimum observations",min_cycles:"Minimum cycles",rhythmic_alpha:"Significance threshold",multiple_testing:"Correction"};
      const keys=[...fixed,...Object.keys(schema.scientific_options).filter(key=>(key.startsWith("detrend_")&&key!=="detrend_methods")||key==="period_config")];
      for(const key of keys){
        const spec=schema.scientific_options[key]||{},label=document.createElement("label"),choices=key==="multiple_testing"?["bh","bonferroni","sidak","none"]:spec.allowed||[];
        const input=document.createElement(choices.length?"select":key==="period_config"?"textarea":"input");input.id="ra-option-"+key;input.dataset.option=key;input.dataset.kind=key==="period_config"?"json":key==="multiple_testing"?"str":key==="min_observations"?"int":["period_min_hours","period_max_hours","min_cycles","rhythmic_alpha"].includes(key)?"float":spec.type||"float";
        if(choices.length)input.replaceChildren(...choices.map(value=>new Option(spec.choice_labels?.[value]||value,value)));else if(input.tagName==="INPUT"){input.type="number";input.step=input.dataset.kind==="int"?"1":"any";if(spec.minimum!=null)input.min=spec.minimum;if(spec.maximum!=null)input.max=spec.maximum;}
        label.textContent=(labels[key]||spec.label||key.replaceAll("_"," ").replace("detrend ",""))+" ";label.title=spec.description||label.textContent;label.append(input);host.append(label);this.advancedFields.push(input);
      }
    }
    installSharedControls(){
      const e=this.options.explorer();if(!e)return;
      this.facade=Object.create(e);Object.assign(this.facade,{host:this.host,data:this.data,styles:this.styles,styleHistory:{},measurementTabs:{},metricPickers:new Map(),comparisons:null,$:id=>this.host.querySelector(`#xp-${id}`),draw:()=>this.draw(),saveSession:()=>this.save()});
      // The same grouped measurement picker used by every existing trace card.
      this.$("metric").id="xp-audit-metric";
      this.facade.updateMetricPicker("audit-metric");this.metricElement=this.host.querySelector("#xp-audit-metric");
      this.metricElement.id="ra-metric";
      this.facade.$=id=>id==="audit-metric"?this.metricElement:this.host.querySelector(`#xp-${id}`);
      this.facade.style=id=>{const base=e.style.call(this.facade,id),dark=document.documentElement.dataset.pmTheme!=="light";return {...base,background:this.styles[id]?.background||(dark?"#101923":"#ffffff"),accent:this.styles[id]?.accent||(dark?"#4fd1c5":"#167b78")};};
    }
    setSelection(selection){if(selection.identity!=null)this.$("cell").value=selection.identity;if(selection.metric&&this.schema.metrics.some(row=>row.key===selection.metric))this.$("metric").value=selection.metric;this.facade?.syncMetricPickers();}
    setForm(config){
      for(const [id,key] of [["cell","identity"],["metric","metric"],["test","significance_method"],["normalization","trace_normalization"]])if(config[key]!=null)this.$(id).value=config[key];
      for(const [id,key] of [["detrends","detrend_methods"],["methods","period_methods"]])if(config[key])for(const input of this.$(id).querySelectorAll("input"))input.checked=config[key].includes(input.value);
      for(const input of this.advancedFields){const key=input.dataset.option,value=config[key];input.value=input.dataset.kind==="json"?JSON.stringify(value||{}):value??"";}
      this.$("input-filters").replaceChildren();for(const recipe of config.input_filters||[])if(recipe.method!=="none")this.addFilter(recipe);
      this.$("normalization-config").value=JSON.stringify(config.trace_normalization_config||{});this.facade?.syncMetricPickers();
    }
    addFilter(recipe){
      const row=document.createElement("div");row.className="controls ra-filter-row";
      const options=(this.schema?.input_filters||[]).filter(value=>value.window_required);
      row.innerHTML=`<label>Filter <select aria-label="Additional input filter">${options.map(value=>`<option value="${esc(value.key)}">${esc(value.label)}</option>`).join("")}</select></label><label>Window, samples <input aria-label="Filter window in samples" type="number" min="3" max="101" step="2"></label><button type="button" title="Remove this filter from the next audit" aria-label="Remove input filter">Remove</button>`;
      row.querySelector("select").value=recipe.method;row.querySelector("input").value=recipe.window_points||3;
      row.querySelector("button").onclick=()=>{row.remove();this.changed();};this.$("input-filters").append(row);
    }
    filtersForResult(){
      return [...new Set((this.result?.results||[]).map(row=>row.preprocessor))].map(key=>{
        const row=this.result.results.find(value=>value.preprocessor===key),recipe=row.trace_filter_json?JSON.parse(row.trace_filter_json):key==="median"?{method:"median",window_points:this.result.options.median_window_points||3}:{method:"none"};
        return {key,recipe,label:row.filter_label||(recipe.method==="none"?"None (unfiltered)":`${this.schema.input_filters.find(value=>value.key===recipe.method)?.label||recipe.method} (${recipe.window_points} samples)`)};
      });
    }
    filterLabel(key){return this.filtersForResult().find(value=>value.key===key)?.label||key;}
    filterColor(key){const index=this.filtersForResult().findIndex(value=>value.key===key);return index<=0?(this.facade?.style("audit-grid").accent||"#167b78"): ["#5599e6","#e4a048","#bf83e4","#ed779d"][(index-1)%4];}
    request(){const config={identity:Number(this.$("cell").value),metric:this.$("metric").value,detrend_methods:[...this.$("detrends").querySelectorAll("input:checked")].map(el=>el.value),period_methods:[...this.$("methods").querySelectorAll("input:checked")].map(el=>el.value),significance_method:this.$("test").value,trace_normalization:this.$("normalization").value,input_filters:[{method:"none"},...[...this.$("input-filters").children].map(row=>({method:row.querySelector("select").value,window_points:Number(row.querySelector("input").value)}))],trace_normalization_config:JSON.parse(this.$("normalization-config").value)};
      for(const input of this.advancedFields){const value=input.value;config[input.dataset.option]=input.dataset.kind==="json"?JSON.parse(value):["int","float"].includes(input.dataset.kind)?value===""?null:Number(value):value;}
      config.input_filters=config.input_filters.filter((recipe,index,all)=>all.findIndex(value=>JSON.stringify(value)===JSON.stringify(recipe))===index);
      return config;
    }
    changed(){
      if(!this.schema)return;
      try{const draft=this.request(),dirty=!this.result||Object.entries(draft).some(([key,value])=>JSON.stringify(value)!==JSON.stringify(key==="input_filters"?this.filtersForResult().map(value=>value.recipe):this.result.options[key]));this.$("pending").textContent=dirty?this.result?"Settings changed · shown audit stays unchanged until Run audit":"Draft · press Run audit":"Shown audit matches these settings";}
      catch{this.$("pending").textContent="Complete the settings before running";}
      const rows=this.options.explorer()?.rhythmAnalyses,metric=this.$("metric").value,identity=Number(this.$("cell").value),local=rows?.metricFilters[metric];
      const allowed=(!local?.identities||local.identities.includes(identity))&&!Object.values(rows?.metricFilters||{}).some(state=>state.scope==="all"&&state.identities&&!state.identities.includes(identity));
      this.$("exclusion").textContent=allowed===false?"Excluded by this measurement’s main rules · available here for inspection":"";
      this.save();
    }
    save(){if(!this.data||!this.schema)return;let options;try{options=this.request();}catch{return;}for(const el of this.host.querySelectorAll("details[id]"))this.folds[el.id]=el.open;
      this.saved={schema_version:2,run:this.data.run,stem:this.data.stem,options,audit_id:this.result?.audit_id||null,job_id:this.jobId||null,styles:this.styles,folds:this.folds,view:this.view};try{localStorage.setItem(KEY,JSON.stringify(this.saved));}catch{this.$("status").textContent="Browser settings could not be saved; completed audits remain on disk.";}}
    async run(){if(this.running||!this.schema)return;const epoch=this.epoch;try{const invalid=this.host.querySelector("#ra-config input:invalid");if(invalid)throw Error(`Check ${invalid.closest("label").textContent.trim()}.`);const options=this.request();if(!options.detrend_methods.length||!options.period_methods.length)throw Error("Choose at least one detrending row and period estimator.");this.running=true;this.$("run").disabled=true;this.$("status").textContent="Queuing audit…";const job=await this.options.api("/api/exploration/audit-jobs",{method:"POST",body:JSON.stringify({run:this.data.run,stem:this.data.stem,options})});if(epoch!==this.epoch)return;this.watch(job);}catch(error){if(epoch===this.epoch){this.running=false;this.$("run").disabled=false;this.$("status").textContent=error.message;}}}
    async cancel(){if(!this.jobId)return;try{await this.options.api(`/api/exploration/audit-jobs/${this.jobId}/cancel`,{method:"POST",body:"{}"});this.$("status").textContent="Cancellation requested · finishing the current comparison";this.$("cancel").disabled=true;}catch(error){this.$("status").textContent=error.message;}}
    async watch(job){
      clearTimeout(this.timer);const epoch=this.epoch;this.jobId=job.id;const active=["queued","running"].includes(job.state);this.running=active||job.state==="completed";this.$("run").disabled=this.running;this.$("cancel").hidden=!active;this.$("cancel").disabled=!!job.cancel_requested;
      const progress=job.progress||{};this.$("progress").hidden=!active;this.$("progress").max=progress.total||1;this.$("progress").value=progress.completed||0;
      this.$("status").textContent=job.cancel_requested&&active?"Cancellation requested · finishing the current comparison":active?`${job.state==="queued"?"Queued":"Comparing"} · ${progress.completed||0}/${progress.total||0}${progress.detrend?" · "+this.label("detrends",progress.detrend):""}`:job.error||"Loading completed audit…";
      if(!active){this.jobId=null;if(job.state==="completed"){await this.loadResult(job.audit_id);if(epoch!==this.epoch)return;this.$("status").textContent=job.reused?"Completed audit reused":"Audit completed";this.history=(await this.options.api(this.url("audit-history"))).audits;if(epoch!==this.epoch)return;this.historyMenu();}this.running=false;this.$("run").disabled=false;this.save();return;}
      this.save();this.timer=setTimeout(async()=>{try{const next=await this.options.api(`/api/jobs/${job.id}`);if(epoch===this.epoch)this.watch(next);}catch(error){if(epoch===this.epoch){this.$("status").textContent="Progress unavailable · "+error.message;this.timer=setTimeout(()=>this.watch(job),2000);}}},700);
    }
    historyMenu(){const selected=this.result?.audit_id||"";this.$("history").replaceChildren(new Option("Choose a completed audit",""),...(this.history||[]).map(row=>new Option(`Cell ${row.options.identity} · ${this.label("metrics",row.options.metric)} · ${new Date(row.created_at).toLocaleString()}`,row.audit_id)));this.$("history").value=selected;}
    label(collection,key){return this.schema?.[collection]?.find(row=>row.key===key)?.label||key;}
    async loadResult(id){const epoch=this.epoch,request=this.resultRequest=(this.resultRequest||0)+1;try{const result=await this.options.api(this.url("audit",{audit_id:id}));if(epoch!==this.epoch||request!==this.resultRequest)return;this.result=result;this.renderResults();this.$("use-shown").disabled=false;this.$("shown").textContent=`Shown: Cell ${result.options.identity} · ${this.label("metrics",result.options.metric)} · ${new Date(result.created_at).toLocaleString()}`;this.$("history").value=id;this.changed();}catch(error){if(epoch===this.epoch)this.$("status").textContent=error.message;}}
    panel(id,title,body,settings=""){return `<article class="panel ra-panel" data-plot-type="matrix" data-style-key="${id}"><div class="panel-heading"><h3>${esc(title)}</h3></div><details class="settings" id="ra-settings-${id}"><summary>Plot settings</summary>${settings||'<button type="button" data-shared-time-settings title="Edit the time range and interval guides shared by the audit plots">Time range and interval guides</button>'}</details>${body}<details class="xp-export" id="ra-export-${id}"><summary>Export</summary><div class="xp-export-actions ra-downloads"></div></details></article>`;}
    renderResults(){
      const result=this.result,config=result.options;this.cursor=null;this.selected=null;
      const guides=`<div class="controls"><label>Show input <select id="ra-input-view"><option value="both">All selected inputs</option>${this.filtersForResult().map(value=>`<option value="${esc(value.key)}">${esc(value.label)}</option>`).join("")}</select></label><div class="xp-guide-controls"><label>Guides every (h) <input id="ra-guide-every" type="number" min="0" step="any" placeholder="Off"></label><label>Guide offset (h) <input id="ra-guide-offset" type="number" step="any" value="0"></label></div><label>Time start (h) <input id="ra-time-min" type="number" step="any" placeholder="Auto"></label><label>Time end (h) <input id="ra-time-max" type="number" step="any" placeholder="Auto"></label></div>`;
      const columns=config.period_methods.length,tiles=[];let index=0;
      const heads=config.period_methods.map(method=>`<div class="ra-column"><strong>${esc(this.label("methods",method))}</strong><span>Test: ${esc(this.label("tests",config.significance_method))}</span></div>`).join("");
      let grid=`<div class="ra-grid-header"></div>${heads}`;
      for(const detrend of config.detrend_methods){grid+=`<div class="ra-row-label">${esc(this.label("detrends",detrend))}</div>`;for(const method of config.period_methods){const rows=result.results.filter(row=>row.detrend===detrend&&row.estimator===method),id=`audit-tile-${index++}`;
        grid+=`<div class="ra-tile" data-detrend="${esc(detrend)}" data-method="${esc(method)}"><canvas id="xp-${id}" height="170" aria-label="${esc(this.label("detrends",detrend)+", "+this.label("methods",method))}"></canvas>${rows.map(row=>`<button type="button" class="ra-result" data-preprocessor="${row.preprocessor}"><strong><span class="ra-filter-swatch" style="background:${this.filterColor(row.preprocessor)}" aria-hidden="true"></span>${esc(this.filterLabel(row.preprocessor))}</strong><span>${num(row.estimated_period_hours)}${C.finite(row.estimated_period_hours)?" h":""} · ${esc(row.estimate_status!=="ok"?row.estimate_status:row.rhythm_status)} · p ${num(row.p_value)} · q ${num(row.q_value)}</span></button>`).join("")}</div>`;
        tiles.push({id,detrend,method,rows});}}
      this.tiles=tiles;
      this.$("results").innerHTML=this.panel("audit-raw","Raw trace",'<canvas id="xp-audit-raw" height="200" aria-label="Original measured cell trace"></canvas>')+this.panel("audit-grid","Detrending and rhythm comparison",`<canvas id="xp-audit-grid" hidden height="170" data-primary-canvas></canvas><div class="ra-grid-scroll"><div class="ra-grid" style="--ra-columns:${columns}">${grid}</div></div>`,guides)+`<section id="ra-detail" hidden>${this.panel("audit-detail","Selected comparison",'<canvas id="xp-audit-detail" height="220"></canvas><div id="ra-detail-summary"></div><div class="controls"><button id="ra-promote" type="button">Use this setup</button><span id="ra-promote-status"></span></div><details id="ra-diagnostics"><summary>Components and spectral diagnostics</summary><canvas id="xp-audit-spectrum" height="200" hidden></canvas><div id="ra-components"></div></details><details id="ra-evidence"><summary>Recorded evidence and settings</summary><pre id="ra-evidence-json"></pre></details>')}</section>`;
      for(const canvas of this.$("results").querySelectorAll("canvas"))canvas.dataset.baseHeight=canvas.height;
      if(this.facade){this.facade.plotEditors([...this.$("results").querySelectorAll("[data-plot-type]")]);}
      for(const button of this.$("results").querySelectorAll("[data-shared-time-settings]"))button.onclick=()=>{this.$("settings-audit-grid").open=true;this.$("settings-audit-grid").scrollIntoView({block:"center"});};
      for(const panel of this.$("results").querySelectorAll(".ra-panel")){
        const id=panel.dataset.styleKey,downloads=panel.querySelector(".ra-downloads");
        for(const file of result.artifacts||[]){if(id==="audit-raw"&&!['input.csv','traces.csv'].includes(file.name))continue;const link=document.createElement("a");link.href="/api/artifacts/"+file.token;link.download=file.name;link.textContent=file.name==="reproduce.py"?"Reproduce in Python":file.name;downloads.append(link);}
        if(id==="audit-grid"){
          const button=document.createElement("button");button.type="button";button.textContent="Configure native audit figure";button.title="Export through the registered audit renderer. Its figure uses unnormalised traces; the saved tables retain this page's display normalisation.";
          button.onclick=()=>{const scientific=Object.fromEntries(Object.entries(config).filter(([key])=>key in this.schema.scientific_options));this.options.explorer().options.onAction("period_method_audit",{...scientific,run:result.run,stem:result.stem,view:"method_grid",identity:config.identity,metrics:[config.metric],detrend:config.detrend_methods[0],fit_method:config.period_methods[0],detrend_methods:config.detrend_methods,period_methods:config.period_methods,input_filters:this.filtersForResult().map(value=>value.recipe)});};downloads.prepend(button);
        }
      }
      for(const [id,key,fallback] of [["input-view","input","both"],["guide-every","guides",""],["guide-offset","offset","0"],["time-min","min",""],["time-max","max",""]]){this.$(id).value=this.view[key]??fallback;if(id==="input-view"&&!this.$(id).value){this.$(id).value="both";this.view.input="both";}this.$(id).onchange=()=>{const interval=Number(this.$("guide-every").value),span=this.timeBounds()[1]-this.timeBounds()[0];this.$("guide-every").setCustomValidity(interval>0&&span/interval>1000?"Choose an interval with at most 1000 guides":"");if(!this.$("guide-every").reportValidity())return;this.view[key]=this.$(id).value;this.draw();this.save();};}
      for(const tile of this.$("results").querySelectorAll(".ra-tile"))for(const button of tile.querySelectorAll(".ra-result"))button.onclick=()=>this.inspect(tile.dataset.detrend,tile.dataset.method,button.dataset.preprocessor);
      for(const canvas of this.$("results").querySelectorAll("canvas")){canvas.onpointermove=event=>{const rect=canvas.getBoundingClientRect(),[lo,hi]=this.timeBounds();this.cursor=lo+(event.clientX-rect.left-45)/Math.max(1,rect.width-57)*(hi-lo);if(!this.cursorFrame)this.cursorFrame=requestAnimationFrame(()=>{this.cursorFrame=null;this.draw();});};canvas.onpointerleave=()=>{this.cursor=null;this.draw();};}
      this.$("promote").onclick=()=>this.promote();
      for(const details of this.$("results").querySelectorAll("details[id]"))if(details.id in this.folds)details.open=this.folds[details.id];
      this.$("config").open=false;
      this.draw();
    }
    trace(detrend,method,preprocessor){return this.result.traces.filter(row=>row.display_panel==="fit"&&row.detrend===detrend&&row.estimator===method&&row.preprocessor===preprocessor);}
    timeBounds(){const hours=this.result?.traces.filter(row=>row.display_panel==="raw_reference").map(row=>row.hours)||[0,1],range=C.range(hours),lo=this.view.min!==""&&this.view.min!=null?Number(this.view.min):range[0],hi=this.view.max!==""&&this.view.max!=null?Number(this.view.max):range[1];return hi>lo?[lo,hi]:range;}
    draw(){if(!this.visible||!this.result||!this.host.querySelector("#xp-audit-raw"))return;
      this.chart("audit-raw",[{rows:this.result.traces.filter(row=>row.display_panel==="raw_reference"),field:"raw_value"}],"audit-raw");
      for(const tile of this.tiles||[]){const preprocessors=(this.view.input||"both")==="both"?this.filtersForResult().map(value=>value.key):[this.view.input],series=preprocessors.map(preprocessor=>({rows:this.trace(tile.detrend,tile.method,preprocessor),field:"display_value",color:this.filterColor(preprocessor),dashed:preprocessor!=="raw"}));this.chart(tile.id,series,"audit-grid");}
      if(this.selected){const rows=this.trace(this.selected.detrend,this.selected.estimator,this.selected.preprocessor);this.chart("audit-detail",[{rows,field:"display_value"}],"audit-detail");const spectra=this.result.spectra.filter(row=>row.detrend===this.selected.detrend&&row.estimator===this.selected.estimator&&row.preprocessor===this.selected.preprocessor);if(spectra.length)this.chart("audit-spectrum",[{rows:spectra.map(row=>({...row,hours:row.period_hours,value:row.power})),field:"value"}],"audit-detail",true);}
    }
    chart(id,series,styleId,spectrum=false){
      const canvas=this.host.querySelector(`#xp-${id}`);if(!canvas||canvas.hidden)return;const style=this.facade?.style(styleId)||{height:180,fontSize:12,lineWidth:1.5,background:"#ffffff",accent:"#167b78"},w=Math.max(1,canvas.getBoundingClientRect().width),h=id==="audit-spectrum"?200:style.height,ratio=devicePixelRatio||1;canvas.style.height=h+"px";canvas.width=Math.round(w*ratio);canvas.height=Math.round(h*ratio);const ctx=canvas.getContext("2d");ctx.scale(ratio,ratio);ctx.fillStyle=style.background;ctx.fillRect(0,0,w,h);ctx.font=`${style.fontSize}px system-ui`;const rgb=[1,3,5].map(i=>parseInt(style.background.slice(i,i+2),16)),ink=rgb.reduce((a,v)=>a+v,0)>435?"#27313b":"#dce5ed";
      const [lo,hi]=spectrum?C.range(series.flatMap(item=>item.rows.map(row=>row.hours))):this.timeBounds(),values=series.flatMap(item=>item.rows.map(row=>row[item.field])),[bottom,top]=C.range(values),left=45,right=w-12,up=12,down=h-28,x=value=>left+(value-lo)/(hi-lo)*(right-left),y=value=>down-(value-bottom)/(top-bottom)*(down-up);
      ctx.strokeStyle=ink;ctx.lineWidth=.6;ctx.beginPath();ctx.moveTo(left,up);ctx.lineTo(left,down);ctx.lineTo(right,down);ctx.stroke();ctx.fillStyle=ink;ctx.textAlign="right";ctx.fillText(num(top),left-4,up+9,40);ctx.fillText(num(bottom),left-4,down,40);ctx.textAlign="center";for(let i=0;i<=4;i++){const t=lo+(hi-lo)*i/4;ctx.fillText(num(t),x(t),h-10,Math.max(18,(right-left)/4));}
      ctx.save();ctx.beginPath();ctx.rect(left,up,right-left,down-up);ctx.clip();
      if(!spectrum){const every=Number(this.view.guides),offset=Number(this.view.offset)||0;if(every>0&&(hi-lo)/every<=1000){ctx.strokeStyle=ink;ctx.globalAlpha=.25;ctx.setLineDash([4,4]);for(let t=offset+Math.ceil((lo-offset)/every)*every;t<=hi;t+=every){ctx.beginPath();ctx.moveTo(x(t),up);ctx.lineTo(x(t),down);ctx.stroke();}ctx.globalAlpha=1;}}
      for(const item of series){ctx.strokeStyle=item.color||style.accent;ctx.lineWidth=style.lineWidth;ctx.setLineDash(item.dashed?[5,3]:[]);ctx.globalAlpha=1;let started=false;ctx.beginPath();const steps=item.rows.slice(1).map((row,i)=>row.hours-item.rows[i].hours).filter(value=>value>0).sort((a,b)=>a-b),interval=steps[Math.floor(steps.length/2)]||Infinity;let previous=null;for(const row of item.rows){const value=row[item.field];if(!C.finite(value)||!C.finite(row.hours)){started=false;previous=null;continue;}if(previous!=null&&row.hours-previous>1.5*interval)started=false;if(started)ctx.lineTo(x(row.hours),y(value));else ctx.moveTo(x(row.hours),y(value));started=true;previous=row.hours;}ctx.stroke();}
      ctx.globalAlpha=1;ctx.setLineDash([]);if(!spectrum&&this.cursor>=lo&&this.cursor<=hi){ctx.strokeStyle=ink;ctx.globalAlpha=.5;ctx.beginPath();ctx.moveTo(x(this.cursor),up);ctx.lineTo(x(this.cursor),down);ctx.stroke();}ctx.restore();
      if(!values.some(C.finite)){ctx.fillStyle=ink;ctx.textAlign="center";ctx.fillText("Trace unavailable",w/2,h/2,w-24);}canvas.dataset.drawn="true";
    }
    inspect(detrend,method,preprocessor){const row=this.result.results.find(row=>row.detrend===detrend&&row.estimator===method&&row.preprocessor===preprocessor);if(!row)return;this.selected=row;this.$("detail").hidden=false;
      const trace=this.trace(detrend,method,preprocessor),normalizationReason=trace.find(value=>value.normalization_message)?.normalization_message;
      this.$("detail-summary").textContent=`${this.label("detrends",detrend)} · ${this.label("methods",method)} · ${this.filterLabel(preprocessor)} input · ${num(row.estimated_period_hours)} h · ${row.rhythm_status}. ${row.message||""}${row.period_underdetermined?" Period support limited by the observed cycles.":""}${row.period_at_search_edge?" Period lies at the search boundary.":""} Native fitted series unavailable; showing detrended observations.${normalizationReason?" Display normalisation unavailable: "+normalizationReason:""}`;
      const components=this.result.components.filter(value=>value.detrend===detrend&&value.estimator===method&&value.preprocessor===preprocessor),spectra=this.result.spectra.filter(value=>value.detrend===detrend&&value.estimator===method&&value.preprocessor===preprocessor);this.host.querySelector("#xp-audit-spectrum").hidden=!spectra.length;
      this.$("components").innerHTML=components.length?`<table><thead><tr><th>Component</th><th>Period (h)</th><th>Amplitude</th><th>Phase (h)</th><th>Relative amplitude error</th></tr></thead><tbody>${components.map(value=>`<tr><td>${value.component_index}${value.selected?" · selected":""}</td><td>${num(value.period_hours)}</td><td>${num(value.amplitude)}</td><td>${num(value.phase_hours)}</td><td>${num(value.relative_amplitude_error)}</td></tr>`).join("")}</tbody></table><p>Component significance was not tested by this audit.</p>`:"No component diagnostics returned.";
      this.$("evidence-json").textContent=JSON.stringify({result:row,correction_family:this.result.correction_family,normalization_scope:this.result.normalization_scope,settings:this.result.options},null,2);
      const transfer=this.transfer(row);this.$("promote").disabled=!!transfer.reason;this.$("promote-status").textContent=transfer.reason||"Adds a configured main analysis row; run it separately.";
      for(const tile of this.host.querySelectorAll(".ra-tile"))tile.classList.toggle("is-selected",tile.dataset.detrend===detrend&&tile.dataset.method===method);this.draw();this.$("detail").scrollIntoView({block:"nearest",behavior:"smooth"});
    }
    transfer(row){const config=this.result.options,e=this.options.explorer(),allowed=e?.rhythmAnalyses?.schema?.detrends.map(value=>value.key)||[];let detrend=row.detrend;
      const extra={};if(["poly3","poly6"].includes(detrend)){extra.detrend_polynomial_degree=detrend==="poly6"?6:3;detrend="polynomial";}
      if(!allowed.includes(detrend))return {reason:"This detrending choice is not available in the main analysis controls."};
      const shared=Object.fromEntries(Object.entries(config).filter(([key])=>key in e.rhythmAnalyses.schema.defaults));
      return {options:{...shared,...extra,trace_filter:this.filtersForResult().find(value=>value.key===row.preprocessor)?.recipe||{method:"none"},metrics:[config.metric],fit_method:row.estimator,significance_method:row.significance_method,fft_component_test:false,detrend,trace_view:"detrended"}};
    }
    async promote(){if(!this.selected)return;const transfer=this.transfer(this.selected);if(transfer.reason)return;try{await this.options.onUse(transfer.options);this.$("promote-status").textContent="Configured analysis row added.";}catch(error){this.$("promote-status").textContent=error.message;}}
  }
  window.PyMicrogliaRhythmAudit={mount:(host,options)=>new RhythmAudit(host,options)};
})();
