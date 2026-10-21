"use strict";
(() => {
  const $ = id => document.getElementById(id);
  const state = {map:null, page:"rhythms", workspace:"micro", action:null, jobs:[], config:null,
                 motionPage:"frames", motionSource:"", motionActions:null, motionInfo:null,
                 motionFrame:1, results:null, recentRuns:[], quicklook:null,
                 quicklookMessage:"Choose a saved measurement run to see its plots.",runLoadId:0,
                  explorationCatalogue:null,explorer:null,explorationData:null,eventBrowser:null};
  const sessionEdits=new Set(["swap_identities","reassign_identities","delete_between_frames",
                              "remove_identities","apply_plan","apply_batch"]);
  const escapeHtml = value => String(value ?? "").replace(/[&<>"']/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"})[char]);
  async function api(path, options={}) {
    const response=await fetch(path,{...options,headers:{"Content-Type":"application/json",...(options.headers||{})}});
    let result; try{result=await response.json();}catch{result={message:await response.text()};}
    if(!response.ok) throw new Error(result.detail||result.message||`Request failed (${response.status})`);
    return result;
  }
  function setWorkspace(name) {
    if(name==="events"){name="micro";state.page="events";localStorage.setItem("pm-page","events");}
    state.workspace=name;
    document.documentElement.dataset.pmWorkspace=name;
    localStorage.setItem("pm-workspace",name);
    document.querySelectorAll(".top-tabs button").forEach(button=>button.classList.toggle("active",button.dataset.workspace===name));
    $("micro-sidebar").hidden=name!=="micro";
    $("motion-sidebar").hidden=name!=="motion";
    $("auto-workspace").hidden=name!=="auto";
    $("motion-workspace").hidden=name!=="motion";
    $("micro-workspace").hidden=name!=="micro";
    $("events-sidebar").hidden=name!=="micro"||state.page!=="events";
    $("events-workspace").hidden=name!=="micro"||state.page!=="events";
    $("main").style.gridColumn=name==="auto"?"1 / -1":"";
    if(name!=="micro")state.explorer?.pause();
    if(name==="auto") openAuto();
    if(name==="motion") openMotion();
    if(name==="micro")renderPage();else state.eventBrowser?.close();
  }
  async function ensureExplorationRun(run,stem){
    if(state.explorer?.data?.run===run&&state.explorer?.data?.stem===stem)return;
    $("run-path").value=run;await selectRun(false);
    if(state.explorer?.data?.stem!==stem){const data=await api(`/api/exploration?${new URLSearchParams({run,stem})}`);state.explorationData=data;await state.explorer.setData(data);}
  }
  function setMicroPage(page){
    const returning=state.page==="events"&&page!=="events",selected=state.eventBrowser?.data;
    state.page=page;localStorage.setItem("pm-page",page);renderNav();renderPage();
    const jump=()=>state.explorer?.ready.then(()=>{const e=state.explorer;e.$(page)?.scrollIntoView({block:"start"});requestAnimationFrame(()=>e.draw());});
    if(returning&&selected)ensureExplorationRun(selected.run,selected.stem).then(jump).catch(error=>{$("run-context").textContent=error.message;});
    else if(!["events","rhythm-audit"].includes(page))jump();
  }
  function openRhythmAudit(selection=null){
    if(!state.auditBrowser){
      const host=document.createElement("section");host.id="rhythm-audit-host";host.hidden=true;$("exploration-host").after(host);
      state.auditBrowser=window.PyMicrogliaRhythmAudit.mount(host,{api,explorer:()=>state.explorer,
        onRecording:async stem=>{const data=await api(`/api/exploration?${new URLSearchParams({run:$("run-path").value.trim(),stem})}`);state.explorationData=data;await state.explorer.setData(data);await state.auditBrowser.open(data);},
        onUse:async options=>{
          const e=state.explorer;await e.ready;
          if(!e.rhythmAnalyses?.schema)await e.loadRhythmOptions(e.session);
          const row=e.rhythmAnalyses.add({options,shown:false,advanced:true});
          if(!row)throw Error("The main view already has the maximum number of analysis rows.");
          e.rhythmAnalyses.restoreFilter(row);e.saveSession();setMicroPage("follow");row.host.scrollIntoView({block:"center"});
        }});window.pyMicrogliaRhythmAudit=state.auditBrowser;
    }
    const data=state.explorer?.sourceData||state.explorationData;
    state.auditBrowser.open(data,selection);
  }
  function openEvents(){
    if(!state.eventBrowser){state.eventBrowser=window.PyMicrogliaEvents.mount({api,
      cohort:data=>{const explorer=state.explorer;return explorer?.sourceData?.run===data?.run&&explorer?.sourceData?.stem===data?.stem?explorer.cohortIds:null;},
      onInspect:async selection=>{await ensureExplorationRun(selection.run,selection.stem);
        state.page="follow";localStorage.setItem("pm-page","follow");setWorkspace("micro");renderNav();state.explorer?.select({identity:selection.identity,frame:selection.frame});state.explorer?.$("follow")?.scrollIntoView({block:"start"});},
      onTheme:setting=>{localStorage.setItem("pm-extras-theme",setting);updateExtrasTheme();},
      onRecording:data=>{$("run-path").value=data.run;localStorage.setItem("pm-run-folder",data.run);},
      onJob:()=>refreshJobs()});window.pyMicrogliaEvents=state.eventBrowser;}
    const run=$("run-path").value.trim(),stem=state.eventBrowser.data?.run===run?state.eventBrowser.data.stem:state.explorer?.data?.run===run?state.explorer.data.stem:undefined;
    state.eventBrowser.open(run,stem);
  }
  function openAuto() {
    const url=state.config?.auto_url;
    $("auto-unavailable").hidden=!!url;
    $("auto-frame").hidden=!url;
    if(url && $("auto-frame").src!==url) $("auto-frame").src=url;
  }
  function renderNav() {
    const nav=$("micro-nav");nav.innerHTML="";
    for(const group of state.explorationCatalogue?.groups||[]){
      const button=document.createElement("button");button.textContent=group.title;button.dataset.page=group.key;
      button.classList.toggle("active",group.key===state.page||state.page==="rhythms"&&group.key==="follow");
      button.addEventListener("click",()=>setMicroPage(group.key));nav.append(button);
    }
    const pipelineDivider=document.createElement("span");pipelineDivider.className="nav-divider";pipelineDivider.textContent="RUN ON DEMAND";nav.append(pipelineDivider);
    const pipelineButton=document.createElement("button");pipelineButton.textContent="Pipelines";
    pipelineButton.dataset.page="pipelines";pipelineButton.classList.toggle("active",state.page==="pipelines");
    pipelineButton.onclick=()=>setMicroPage("pipelines");nav.append(pipelineButton);
    const resultButton=document.createElement("button");resultButton.textContent="Results and export";
    resultButton.dataset.page="results";resultButton.classList.toggle("active",state.page==="results");
    resultButton.addEventListener("click",()=>setMicroPage("results"));nav.append(resultButton);
    const allButton=document.createElement("button");allButton.textContent="All methods";
    allButton.classList.toggle("active",state.page==="all_methods");
    allButton.onclick=()=>setMicroPage("all_methods");nav.append(allButton);
  }
  function section(title,note=""){
    const block=document.createElement("section");block.innerHTML=`<div class="section-heading"><h2>${escapeHtml(title)}</h2><small>${escapeHtml(note)}</small></div>`;
    const grid=document.createElement("div");grid.className="card-grid";block.append(grid);$("micro-content").append(block);return grid;
  }
  function figureCard(item) {
    const card=document.createElement("article");card.className="card";
    card.innerHTML=`<div class="card-top"><span class="eyebrow">FIGURE</span><span class="pill ${item.saved?"saved":""}">${item.saved?"saved pipeline":"on demand"}</span></div><h3>${escapeHtml(item.title)}</h3><p>${escapeHtml(item.views.join(" · "))}</p><div class="card-bottom"><small>${item.views.length} ${item.views.length===1?"view":"views"}</small><button class="button ghost" type="button">Set up →</button></div>`;
    card.querySelector("button").onclick=()=>showAction(item.key);
    return card;
  }
  function actionCard(name,kind="ANALYSIS") {
    const row=state.map.inventory.find(item=>item.name===name && item.kind==="action");
    const card=document.createElement("article");card.className="card";
    card.innerHTML=`<div class="card-top"><span class="eyebrow">${kind}</span></div><h3>${escapeHtml(name.replaceAll("_"," "))}</h3><p>${escapeHtml(row?.summary||"Open the live action settings")}</p><div class="card-bottom"><small>Live method</small><button class="button ghost" type="button">Set up →</button></div>`;
    card.querySelector("button").onclick=()=>showAction(name);return card;
  }
  function renderPage() {
    const content=$("micro-content");content.innerHTML="";
    const events=state.page==="events";
    const audit=state.page==="rhythm-audit";
    if(!audit)state.auditBrowser?.close();
    document.documentElement.dataset.pmPage=state.page;
    $("micro-page-head").hidden=events||audit;$("micro-toolbar").hidden=events;
    $("events-sidebar").hidden=!events||state.workspace!=="micro";$("events-workspace").hidden=!events||state.workspace!=="micro";
    if(events){$("exploration-host").hidden=true;content.hidden=true;state.explorer?.pause();if(state.workspace==="micro")openEvents();return;}
    state.eventBrowser?.close();
    if(audit){$("exploration-host").hidden=true;content.hidden=true;state.explorer?.pause();if(state.workspace==="micro")openRhythmAudit();return;}
    const onDemand=["results","pipelines","all_methods"].includes(state.page);
    $("exploration-host").hidden=onDemand;content.hidden=!onDemand;
    if(onDemand)state.explorer?.pause();
    if(state.page==="results") {$("page-title").textContent="Results and export";$("page-question").textContent="Review saved evidence and output files.";renderResults();return;}
    if(state.page==="pipelines") {renderPipelines();return;}
    if(state.page==="all_methods") {renderAllMethods();return;}
    $("page-title").textContent="A cell in context.";$("page-question").textContent="Watch it move. Follow its signal. See where it belongs.";
    if(!state.explorer){state.explorer=window.PyMicrogliaExploration.mount($("exploration-host"),{api,catalogue:state.explorationCatalogue,onAction:showAction,
      onAudit:selection=>{state.page="rhythm-audit";localStorage.setItem("pm-page",state.page);renderNav();renderPage();openRhythmAudit(selection);},
      onSelection:selection=>{const source=state.explorer?.data?.frames?.[selection.frame]?.source_frame_index;
        const frame=Number.isInteger(source)?source+1:selection.frame+1;
        $("exploration-readout").textContent=`${selection.stem} \u00b7 Cell ${selection.identity} \u00b7 source frame ${frame}`;},
      onRecording:data=>{state.explorationData=data;state.quicklook=data;$("run-context").textContent=`${data.total_cells||0} measured cells`;},
      onCohort:()=>{if(state.eventBrowser?.visible)state.eventBrowser.renderList();}});window.pyMicrogliaExplorer=state.explorer;}
  }
  function renderPipelines(){
    $("page-title").textContent="Pipelines";$("page-question").textContent="Run a complete workflow when you choose; quick plots on the other pages use saved measurements.";
    const workflows=state.map.inventory.filter(item=>item.kind==="pipeline");
    const grid=section("Complete workflows",`${workflows.length} on-demand pipelines`);
    workflows.forEach(row=>grid.append(pipelineCard(row.name)));
  }
  function renderAllMethods(){
    $("page-title").textContent="All methods";$("page-question").textContent="Every registered action, including preparation and tracking calls, in one searchable index.";
    const search=document.createElement("input");search.className="method-search";search.placeholder="Search actions by name or description";
    $("micro-content").append(search);
    const groups=new Map();
    for(const row of state.map.inventory.filter(item=>item.kind==="action")){
      if(!groups.has(row.home))groups.set(row.home,[]);groups.get(row.home).push(row);
    }
    for(const [home,rows] of groups){const grid=section(home.replaceAll("/"," / "),`${rows.length} actions`);
      for(const row of rows){const card=actionCard(row.name,"ACTION");card.dataset.search=(row.name+" "+row.summary).toLowerCase();grid.append(card);}
    }
    const figures=new Map();for(const row of state.map.inventory.filter(r=>r.kind==="figure view"&&!r.alias_of)){if(!figures.has(row.name))figures.set(row.name,{key:row.name,title:row.title,views:[],saved:row.requires_saved_pipeline});figures.get(row.name).views.push(row.view);}
    const exports=section("All saved figures and composite reports","Configure and export through the existing analysis methods");
    for(const item of figures.values()){const card=figureCard(item);card.dataset.search=(item.key+" "+item.title+" "+item.views.join(" ")).toLowerCase();exports.append(card);}
    search.oninput=()=>{const value=search.value.trim().toLowerCase();
      for(const card of $("micro-content").querySelectorAll("[data-search]"))card.hidden=!card.dataset.search.includes(value);
    };
  }
  function pipelineCard(name){
    const card=document.createElement("article");card.className="card";
    const title=name==="native_red_green"?"Full recordings: original Red and Green":name.replaceAll("_"," ");
    card.innerHTML=`<div class="card-top"><span class="eyebrow">PIPELINE</span></div><h3>${escapeHtml(title)}</h3><p>Run this complete workflow with its own saved inputs and settings.</p><div class="card-bottom"><small>Live pipeline</small><button class="button ghost">Set up →</button></div>`;
    card.querySelector("button").onclick=()=>showAction(name,{},true);return card;
  }
  function parseValue(input,type) {
    if(type==="bool")return input.checked;
    const raw=input.value.trim();if(!raw)return undefined;
    if(type==="Any" && /^[\[{]/.test(raw)){try{return JSON.parse(raw);}catch{throw Error(`${input.name} needs valid JSON`);}}
    if(type==="int") {const value=Number(raw);if(!Number.isInteger(value))throw Error(`${input.name} needs a whole number`);return value;}
    if(type==="float") {const value=Number(raw);if(!Number.isFinite(value))throw Error(`${input.name} needs a number`);return value;}
    if(["list","mapping","tuple"].includes(type)){try{return JSON.parse(raw);}catch{throw Error(`${input.name} needs valid JSON`);}}
    return raw;
  }
  function field(spec) {
    const wrapper=document.createElement(spec.item_choices?"div":"label");wrapper.className="field";
    wrapper.title=spec.description||"";
    if(spec.type==="bool")wrapper.classList.add("checkbox");
    const title=document.createElement("span");title.textContent=(spec.label||spec.name.replaceAll("_"," "))+(spec.required?" *":"");
    let input;
    if(spec.item_choices){input=document.createElement("input");input.type="hidden";
    }else if(spec.choices?.length){input=document.createElement("select");
      if(!spec.required){const blank=document.createElement("option");blank.value="";blank.textContent="Use default";input.append(blank);}
      for(const choice of spec.choices){const option=document.createElement("option");option.value=String(choice);option.textContent=spec.choice_labels?.[choice]||String(choice);input.append(option);}
    }else if(["list","mapping","tuple"].includes(spec.type)){input=document.createElement("textarea");wrapper.classList.add("wide");
    }else{input=document.createElement("input");input.type=spec.type==="bool"?"checkbox":["int","float"].includes(spec.type)?"number":"text";if(spec.type==="float")input.step="any";}
    input.name=spec.name;input.dataset.type=spec.type;
    if(spec.required)input.required=true;
    if(spec.type==="bool")input.checked=Boolean(spec.default);
    else if(spec.default!==null && spec.default!==undefined && spec.default!=="")input.value=typeof spec.default==="object"?JSON.stringify(spec.default):String(spec.default);
    if(spec.name==="run" && $("run-path").value.trim())input.value=$("run-path").value.trim();
    if(Object.hasOwn(state.actionPreset||{},spec.name)){
      const selected=state.actionPreset[spec.name];
      if(spec.type==="bool")input.checked=Boolean(selected);
      else input.value=typeof selected==="object"?JSON.stringify(selected):String(selected);
    }
    if(spec.type==="bool")wrapper.append(input,title);else wrapper.append(title,input);
    if(spec.item_choices){
      const choices=document.createElement("div");choices.className="field-checks";
      const selected=JSON.parse(input.value||"[]");
      for(const choice of spec.item_choices){
        const label=document.createElement("label"),check=document.createElement("input"),text=document.createElement("span");
        check.type="checkbox";check.value=choice;check.checked=selected.includes(choice);check.dataset.choiceFor=spec.name;
        text.textContent=spec.choice_labels?.[choice]||choice;
        check.onchange=()=>{input.value=JSON.stringify([...choices.querySelectorAll("input:checked")].map(c=>c.value));input.dispatchEvent(new Event("input",{bubbles:true}));};
        label.append(check,text);choices.append(label);
      }
      wrapper.append(choices);
    }
    const help=document.createElement("small");help.textContent=spec.description||"";wrapper.append(help);return wrapper;
  }
  async function showAction(name,preset={},pipeline=false) {
    try{
      const spec=await api(`/${pipeline?"api/pipelines":"api/actions"}/${encodeURIComponent(name)}`);state.action=spec;
      state.actionPreset=preset;
      $("action-title").textContent=name==="native_red_green"?"Full recordings: original Red and Green":name.replaceAll("_"," ");$("action-summary").textContent=spec.summary;
      $("action-message").textContent=spec.pending?"This action is not installed in the current package.":"";
      const container=$("action-fields");container.innerHTML="";
      const primary=document.createElement("div");primary.className="field-list";
      const advanced=document.createElement("details");advanced.className="advanced";advanced.innerHTML="<summary>More settings</summary>";
      const advancedList=document.createElement("div");advancedList.className="field-list";
      const isCellGrid=["cell_grid","cell_analysis_grid"].includes(name);
      const gridPrimary=new Set(spec.params.filter(p=>p.grid_primary_order!==null&&p.grid_primary_order!==undefined).map(p=>p.name));
      for(const param of spec.params){if(param.name==="claim")continue;(param.required||Object.hasOwn(preset,param.name)||gridPrimary.has(param.name)?primary:advancedList).append(field(param));}
      if(spec.claim_required||!spec.display_only){const claim=field({name:"claim",type:"str",required:spec.claim_required,description:"What should this analysis show?",default:""});primary.append(claim);}
      container.append(primary);advanced.append(advancedList);container.append(advanced);
      if(isCellGrid){
        const entrySpec=spec.params.find(p=>p.name==="analysis");
        const selector=container.querySelector('[name="analysis"]');
        const local=document.createElement("div");local.className="field-list cell-grid-settings";
        const analysisFold=document.createElement("details");analysisFold.className="advanced cell-grid-analysis";
        analysisFold.innerHTML="<summary>Analysis settings</summary>";analysisFold.append(local);primary.append(analysisFold);
        const inputGroup=(host,group,key,param,value)=>{
          const type={object:"mapping",array:"list",number:"float",integer:"int",boolean:"bool"}[param.type]||param.type;
          const wrapper=field({...param,type,name:`cell-grid-${group}-${key}`,label:param.label||key.replaceAll("_"," "),default:value});
          const input=wrapper.querySelector("input,select,textarea");input.dataset.gridGroup=group;input.dataset.gridKey=key;
          host.append(wrapper);
        };
        const preprocessing=spec.params.find(p=>p.name==="preprocessing");
        const presentation=spec.params.find(p=>p.name==="grid_presentation");
        if(presentation?.properties){
          let stored={};try{const saved=JSON.parse(localStorage.getItem("pm-cell-grid-presentation-v1")||"{}");if(saved.schema_version===1)stored=saved.settings||{};}catch{/* Ignore a damaged optional display preference. */}
          const initial=preset.grid_presentation??stored;
          const fold=document.createElement("details");fold.className="advanced cell-grid-analysis";
          fold.innerHTML="<summary>Grid arrangement and overlays</summary>";
          const strip=document.createElement("div");strip.className="field-list";fold.append(strip);primary.append(fold);
          for(const [key,param] of Object.entries(presentation.properties)){
            inputGroup(strip,"grid_presentation",key,param,initial[key]??param.default);
            if(param.trace_only)strip.querySelector(`[data-grid-key="${key}"]`).dataset.traceOnly="true";
            if(param.fft_only)strip.querySelector(`[data-grid-key="${key}"]`).dataset.fftOnly="true";
          }
          const raw=container.querySelector('[name="grid_presentation"]');raw.closest("label").style.display="none";raw.disabled=true;
          strip.addEventListener("input",()=>{try{const settings=Object.fromEntries([...strip.querySelectorAll('[data-grid-group="grid_presentation"]')].map(i=>[i.dataset.gridKey,parseValue(i,i.dataset.type)]));localStorage.setItem("pm-cell-grid-presentation-v1",JSON.stringify({schema_version:1,settings}));}catch{/* Validation reports incomplete edits. */}});
          strip.addEventListener("change",()=>strip.dispatchEvent(new Event("input")));
        }
        if(preprocessing?.properties){
          const initial=preset.preprocessing??preprocessing.default;
          const strip=document.createElement("div");strip.className="field-list cell-grid-preprocessing";primary.append(strip);
          for(const [key,param] of Object.entries(preprocessing.properties))
            inputGroup(strip,"preprocessing",key,param,initial?.[key]??param.default);
          const rawRecipe=container.querySelector('[name="preprocessing"]');
          rawRecipe.closest("label").style.display="none";
        }
        const grouped=group=>Object.fromEntries([...container.querySelectorAll(`[data-grid-group="${group}"]`)]
          .filter(input=>!input.disabled).map(input=>[input.dataset.gridKey,parseValue(input,input.dataset.type)]).filter(([,value])=>value!==undefined));
        const saved=new Map();let previous=null;
        const updatePairing=()=>{
          const selected=selector.value||entrySpec.default;
          const views=container.querySelector('[name="analysis_views"]');
          const keys=[selected,...(views&&!views.disabled?JSON.parse(views.value||"[]"):[])];
          const paired=keys.some(key=>entrySpec.analysis_entries?.[key]?.paired);
          const reference=container.querySelector('[name="reference_metric"]');
          reference.required=paired;reference.disabled=!paired;
          reference.closest("label").hidden=!paired;
          reference.closest("label").style.display=paired?"":"none";
          const components=container.querySelector('[data-fft-only="true"]');
          if(components){
            components.disabled=!keys.some(key=>entrySpec.analysis_entries?.[key]?.fft_components);
            components.closest(".field").style.display=components.disabled?"none":"";
          }
        };
        container.querySelector('[name="analysis_views"]')?.addEventListener("input",updatePairing);
        const update=()=>{
          const entry=entrySpec.analysis_entries?.[selector.value||entrySpec.default];
          if(!entry)return;
          for(const input of container.querySelectorAll('[data-trace-only="true"]')){
            input.disabled=entry.action!=="trace";input.closest(".field").style.display=input.disabled?"none":"";
          }
          const views=container.querySelector('[name="analysis_views"]');
          if(views){views.disabled=entry.action==="trace";views.closest(".field").style.display=views.disabled?"none":"";}
          const raw=container.querySelector('[name="analysis_parameters"]');
          if(previous){try{saved.set(previous,{...JSON.parse(raw.value||"{}"),...grouped("analysis_parameters")});}catch{/* Validation reports invalid raw input. */}}
          local.replaceChildren();
          const restored=saved.get(entry.action)||(previous===null?preset.analysis_parameters||{}:{});
          raw.value=JSON.stringify(restored);previous=entry.action;
          updatePairing();
          const bound=new Set(["recording","recordings","wavelet_values","wavelet_values_b","wavelet_hours","wavelet_result","series_a","series_b","hours","channels","config"]);
          const params=entry.parameters.filter(p=>!bound.has(p.name));
          for(const param of params)inputGroup(local,"analysis_parameters",param.name,param,
            Object.hasOwn(restored,param.name)?restored[param.name]:param.default);
          const recordingSettings=container.querySelector('[name="analysis_settings"]');
          recordingSettings.disabled=!entry.parameters.some(p=>p.name==="config");
          recordingSettings.closest("label").hidden=recordingSettings.disabled;
          recordingSettings.closest("label").style.display=recordingSettings.disabled?"none":"";
          const appearance=container.querySelector('[name="analysis_display"]');
          appearance.disabled=entry.action==="trace";
          appearance.closest("label").style.display=appearance.disabled?"none":"";
          const help=container.querySelector('[name="analysis_parameters"]').closest("label").querySelector("small");
          help.textContent=params.length?params.map(p=>`${p.name}${p.required?" (required)":""}: ${p.description}`).join("\n"):"No extra action parameters are needed.";
          if(["wavelet_reconstruct","wavelet_phases"].includes(entry.action))help.textContent+=" Supply wavelet_options for the preceding spectrum when needed.";
          help.style.whiteSpace="pre-line";
        };
        selector.addEventListener("change",update);update();
      }
      $("action-dialog").showModal();
      if($("run-path").value.trim() && state.map.inventory.some(row=>row.kind==="figure view"&&row.name===name)){
        try{const found=await api(`/api/availability/${encodeURIComponent(name)}?run=${encodeURIComponent($("run-path").value.trim())}`);
          const ready=Object.entries(found.views).filter(([,reason])=>reason===null).map(([view])=>view);
          const missing=Object.entries(found.views).filter(([,reason])=>reason!==null);
          $("action-message").textContent=`Saved inputs: ${ready.length?ready.join(", "):"none ready"}.`+
            (missing.length?`\nUnavailable: ${missing.map(([view,reason])=>`${view}: ${reason}`).join("; ")}`:"");
        }catch(error){$("action-message").textContent=`Saved input check: ${error.message}`;}
      }
    }catch(error){alert(error.message);}
  }
  function actionPayload() {
    const params={};let claim="";
    for(const input of $("action-fields").querySelectorAll("input,select,textarea")){
      if(input.disabled||input.dataset.choiceFor)continue;
      if(input.dataset.gridGroup){
        const value=parseValue(input,input.dataset.type);
        if(value!==undefined){params[input.dataset.gridGroup]??={};params[input.dataset.gridGroup][input.dataset.gridKey]=value;}
        continue;
      }
      const value=parseValue(input,input.dataset.type);
      if(input.name==="claim")claim=value||"";else if(value!==undefined)
        params[input.name]=input.name==="preprocessing"||input.name==="analysis_parameters"?{...value,...params[input.name]}:value;
    }
    return {action:state.action.name,params,claim,request:`PyMicroglia workbench: ${state.action.name}`};
  }
  async function checkAction(run) {
    try{
      const payload=actionPayload();const result=await api(state.action.pipeline?"/api/pipelines/validate":"/api/validate",{method:"POST",body:JSON.stringify(payload)});
      if(result.problems?.length||result.unknown_params?.length){$("action-message").textContent=result.message||JSON.stringify(result.problems);return;}
      $("action-message").textContent="Settings accepted.";
      if(!run)return;
      const started=await api(state.action.pipeline?"/api/pipeline-jobs":"/api/jobs",{method:"POST",body:JSON.stringify(payload)});
      $("action-dialog").close();showJobs();await refreshJobs();watchJob(started.id);
    }catch(error){$("action-message").textContent=error.message;}
  }
  async function refreshJobs(){try{state.jobs=await api("/api/jobs");$("job-count").textContent=state.jobs.length;
    $("job-list").innerHTML=state.jobs.length?state.jobs.map(job=>`<div class="job ${escapeHtml(job.state)}"><strong>${escapeHtml(job.action)}</strong> · ${escapeHtml(job.state)}<small>${escapeHtml(job.error||(job.state==="running"&&job.progress?`${job.progress.stage}: ${job.progress.completed}/${job.progress.total}`:job.record?.run_id)||(job.changed_pixels!==undefined?`${job.changed_pixels} pixels changed · ${job.identity_presence}`:job.id))}</small></div>`).join(""):"<p>No jobs yet.</p>";
  }catch(error){$("job-list").textContent=error.message;}}
  function showJobs(){$("job-tray").hidden=false;}
  function watchJob(id){const timer=setInterval(async()=>{await refreshJobs();const job=state.jobs.find(item=>item.id===id);if(!job||["completed","failed"].includes(job.state)){clearInterval(timer);
    if(job?.state==="completed"&&job.measure_params){setWorkspace("micro");state.page="results";renderNav();renderPage();showAction("measure",job.measure_params);}
    else if(job?.state==="completed"&&job.measurement_run){setWorkspace("micro");state.page="rhythms";$("run-path").value=job.measurement_run;renderNav();selectRun(false);}
    else if(job?.state==="completed")renderPage();
  }},1000);}
  function renderResults(){
    const grid=section("Saved evidence",state.results?.folder||"Select a measurement run above");
    if(!state.results){grid.append(Object.assign(document.createElement("div"),{className:"empty-card",textContent:"Choose a saved measurement run to inspect its figures and outputs."}));}
    else if(!state.results.artifacts.length){grid.append(Object.assign(document.createElement("div"),{className:"empty-card",textContent:"No viewable files found at the run root or one folder below it. Recorded outputs can also be opened from Recent runs."}));}
    else for(const item of state.results.artifacts){
      const card=document.createElement("article");card.className="card artifact-card";
      const url=`/api/artifacts/${encodeURIComponent(item.token)}`;
      const image=["svg","png","jpg","jpeg","webp","gif"].includes(item.kind);
      const video=["mp4","webm"].includes(item.kind);
      card.innerHTML=`<div class="card-top"><span class="eyebrow">${escapeHtml(item.kind.toUpperCase())}</span></div>`+
        (image?`<img class="artifact-preview" src="${url}" alt="${escapeHtml(item.name)}">`:video?`<video class="artifact-preview" src="${url}" controls preload="metadata"></video>`:`<div class="artifact-placeholder">${escapeHtml(item.kind.toUpperCase())}</div>`)+
        `<h3>${escapeHtml(item.name)}</h3><div class="card-bottom"><small>Saved output</small><a class="button secondary" href="${url}" target="_blank" rel="noopener">Open</a></div>`;
      grid.append(card);
    }
    const recent=section("Recent recorded analyses","Select a run to inspect its declared outputs");
    for(const row of state.recentRuns.slice(0,20)){
      const id=row.run_id||row.id;if(!id)continue;
      const card=document.createElement("article");card.className="card";
      card.innerHTML=`<span class="eyebrow">SAVED RUN</span><h3>${escapeHtml(row.action||id)}</h3><p>${escapeHtml(row.claim||row.project||"")}</p><div class="card-bottom"><small>${escapeHtml(id)}</small><button class="button ghost">Inspect →</button></div>`;
      card.querySelector("button").onclick=()=>openRecord(id);recent.append(card);
    }
  }
  async function openRecord(id){try{const found=await api(`/api/runs/${encodeURIComponent(id)}`);state.results={folder:`Recorded analysis ${id}`,artifacts:found.artifacts};state.page="results";renderNav();renderPage();}catch(error){alert(error.message);}}
  async function selectRun(openFiles=false){
    const folder=$("run-path").value.trim(),loadId=++state.runLoadId;localStorage.setItem("pm-run-folder",folder);
    if(!folder){state.results=null;state.quicklook=null;state.quicklookMessage="Choose a saved measurement run to see its plots.";
      state.explorationData=null;state.explorer?.setData(null);$("run-context").textContent="No run selected";if(openFiles)state.page="results";renderNav();renderPage();return;}
    state.quicklook=null;state.quicklookMessage="Loading measured plots...";renderPage();
    let previousStem="",previousSession=null;try{previousSession=JSON.parse(localStorage.getItem("pm-exploration-session-v1")||"null");if(previousSession?.version===1&&previousSession.run===folder)previousStem=previousSession.stem||"";}catch{/* Ignore an older browser setting. */}
    const [preview,saved]=await Promise.allSettled([
      api(`/api/exploration?run=${encodeURIComponent(folder)}${previousStem?`&stem=${encodeURIComponent(previousStem)}`:""}`),
      api("/api/results/select",{method:"POST",body:JSON.stringify({folder})})]);
    if(loadId!==state.runLoadId)return;
    if(preview.status==="fulfilled"){state.quicklook=preview.value;state.explorationData=preview.value;state.quicklookMessage="";}
    else {state.quicklook=null;state.explorationData=null;state.quicklookMessage=preview.reason.message;}
    if(saved.status==="fulfilled"){state.results=saved.value;$("run-context").textContent=`${saved.value.artifacts.length} saved files | ${state.quicklook?.total_cells||0} measured cells`;}
    else {state.results=null;$("run-context").textContent=saved.reason.message;}
    if(openFiles)state.page="results";renderNav();renderPage();
    if(!state.explorer){const previous=state.page;state.page="follow";renderPage();state.page=previous;renderPage();}
    state.explorer.options.message=state.quicklookMessage;await state.explorer.setData(state.explorationData);
    if(loadId!==state.runLoadId)return;
    if(previousSession?.version===1&&previousSession.comparisonRun===folder){
      const source=state.explorer.groupComparison.design?.recordings.find(row=>row.run===previousSession.run&&row.stem===previousSession.stem);
      if(source)await state.explorer.switchRecording(source);
    }
    if(state.page==="rhythm-audit")openRhythmAudit();
  }
  async function openMotion(){
    if(!state.config?.motion_url){$("motion-content").innerHTML="<div class='empty-card'>Motion service is not connected. Start the Motion service and reopen with its URL.</div>";return;}
    if(!state.motionActions){try{state.motionActions=(await api("/api/motion/actions")).actions;}catch(error){$("motion-content").textContent=error.message;return;}}
    if(!$("motion-source").options.length || $("motion-source").options.length===1)await refreshMotionSources();
    if(state.motionSource && !state.motionInfo){
      if([...$("motion-source").options].some(item=>item.value===state.motionSource))await chooseMotionSource(state.motionSource);
      else {state.motionSource="";localStorage.removeItem("pm-motion-source");}
    }
    renderMotionPage();
  }
  async function refreshMotionSources(){
    try{const listed=await api("/api/motion/sources");const select=$("motion-source");
      select.innerHTML="<option value=''>Choose source</option>";
      for(const row of listed.sources||[]){const option=document.createElement("option");option.value=row.source;option.textContent=`${row.name} · ${row.kind}`;select.append(option);}
      if(state.motionSource)select.value=state.motionSource;
      $("motion-source-status").textContent=`${listed.sources?.length||0} saved sources`;
    }catch(error){$("motion-source-status").textContent=error.message;}
  }
  async function chooseMotionSource(value){
    state.motionSource=value;state.motionInfo=null;
    localStorage.setItem("pm-motion-source",value);
    if(value){try{state.motionInfo=await api(`/api/motion/source?source=${encodeURIComponent(value)}`);state.motionFrame=1;
        $("motion-status").textContent=`${state.motionInfo.frames} frames · ${state.motionInfo.identities} identities`;
      }catch(error){$("motion-status").textContent=error.message;}}
    renderMotionPage();
  }
  function motionSection(title,text){
    const node=document.createElement("section");node.className="motion-panel card";
    node.innerHTML=`<span class="eyebrow">SAVED MOTION EVIDENCE</span><h2>${escapeHtml(title)}</h2><p>${escapeHtml(text)}</p>`;
    $("motion-content").append(node);return node;
  }
  function motionActionButtons(container,names){
    const row=document.createElement("div");row.className="motion-action-list";
    for(const name of names){if(!state.motionActions[name])continue;const button=document.createElement("button");button.className="button secondary";button.textContent=name.replaceAll("_"," ");button.title=state.motionActions[name].summary;button.onclick=()=>showMotionAction(name);row.append(button);}
    container.append(row);
  }
  function renderMotionPage(){
    document.querySelectorAll("[data-motion-page]").forEach(button=>button.classList.toggle("active",button.dataset.motionPage===state.motionPage));
    const titles={frames:"Review frames",continuity:"Continuity and gaps",correct:"Correct identities",handoff:"Outline handoff",history:"History and export"};
    $("motion-heading").textContent=titles[state.motionPage];$("motion-content").innerHTML="";
    if(!state.motionSource){motionSection("Choose a saved Motion source","The source can be an editing bundle, saved session or Motion package.");return;}
    const info=state.motionInfo||{};
    $("motion-subtitle").textContent=`${info.stem||"Saved track"} · ${info.frames||"?"} frames · ${info.identities||"?"} identities`;
    if(state.motionPage==="frames"){
      const panel=motionSection("Original frame with saved identities","The image and labels come from the selected Motion session.");
      panel.insertAdjacentHTML("beforeend",`<div class="motion-viewer"><img id="motion-image" alt="Saved Motion review frame"></div><div class="motion-transport"><button id="motion-prev" class="button secondary">‹</button><input id="motion-frame" type="range" min="1" max="${Number(info.frames)||1}" value="${state.motionFrame}"><button id="motion-next" class="button secondary">›</button><strong id="motion-frame-label"></strong></div><div id="motion-frame-cells" class="cell-list"></div>`);
      $("motion-frame").oninput=e=>setMotionFrame(Number(e.target.value));
      $("motion-prev").onclick=()=>setMotionFrame(state.motionFrame-1);
      $("motion-next").onclick=()=>setMotionFrame(state.motionFrame+1);
      setMotionFrame(state.motionFrame);
    }else if(state.motionPage==="continuity"){
      const panel=motionSection("Identity persistence","Select an identity in the current frame to inspect its occupied frame runs and gaps.");
      panel.innerHTML+=`<div class="motion-transport"><label>Frame <input id="continuity-frame" type="number" min="1" max="${Number(info.frames)||1}" value="${state.motionFrame}"></label><button id="load-continuity" class="button secondary">Show cells</button></div><div id="motion-frame-cells" class="cell-list"></div><pre id="motion-track-detail" class="evidence-json"></pre>`;
      $("load-continuity").onclick=()=>{state.motionFrame=Number($("continuity-frame").value);loadFrameCells();};loadFrameCells();
    }else if(state.motionPage==="correct"){
      const panel=motionSection("Save a correction as a new session","The selected source remains unchanged. Each successful edit becomes the next selected session.");
      const edits=Object.keys(state.motionActions).filter(name=>sessionEdits.has(name));
      motionActionButtons(panel,edits);
      const helpers=motionSection("Inspect or preview","Read source evidence, propose edits from feedback marks, or preview a saved plan.");
      motionActionButtons(helpers,Object.keys(state.motionActions).filter(name=>!state.motionActions[name].mutates));
      const feedback=motionSection("Feedback notes","Add or retract feedback marks, or render review files. These actions do not change cell identities.");
      motionActionButtons(feedback,Object.keys(state.motionActions).filter(name=>state.motionActions[name].mutates&&!sessionEdits.has(name)));
    }else if(state.motionPage==="handoff"){
      const panel=motionSection("Identity stays with Motion","Mask refinement changes the pixels measured for each saved cell identity. It does not rename or rerun tracking.");
      const button=document.createElement("button");button.className="button primary";button.textContent="Open mask change figure";button.onclick=()=>{setWorkspace("micro");state.page="cell_explorer";renderNav();renderPage();showAction("mask_change_map");};panel.append(button);
      const source=document.createElement("p");source.textContent=`Selected identity session: ${state.motionSource}`;panel.append(source);
      const exportPanel=motionSection("Measure this corrected history","Create aligned copies of Motion's saved labels and registered signal, with a handoff receipt. The selected session is preserved.");
      const exportButton=document.createElement("button");exportButton.className="button secondary";exportButton.textContent="Prepare measurement inputs";
      exportButton.onclick=async()=>{exportButton.disabled=true;exportButton.textContent="Preparing inputs…";
        try{const handoff=await api("/api/motion/handoff",{method:"POST",body:JSON.stringify({source:state.motionSource})});
          $("run-context").textContent=`Motion handoff: ${handoff.identities} identities`;
          showRefinement(handoff);
        }catch(error){exportButton.disabled=false;exportButton.textContent="Prepare measurement inputs";alert(error.message);}
      };exportPanel.append(exportButton);
    }else if(state.motionPage==="history"){
      const panel=motionSection("Saved correction history","Every step is recorded by Motion. Select an earlier source above to branch from it.");
      const list=document.createElement("ol");list.className="history-list";
      for(const row of info.history||[]){const item=document.createElement("li");item.textContent=`Step ${row.step}: ${row.edit} · ${row.user||"unknown user"}`;list.append(item);}
      if(!list.children.length){const item=document.createElement("li");item.textContent="Automatic tracks; no saved corrections";list.append(item);}panel.append(list);
      motionActionButtons(panel,["build_review","render_movie"]);
    }
  }
  async function setMotionFrame(value){
    state.motionFrame=Math.max(1,Math.min(Number(state.motionInfo?.frames)||1,value));
    if(!$("motion-image"))return;
    $("motion-frame").value=state.motionFrame;$("motion-frame-label").textContent=`Frame ${state.motionFrame} / ${state.motionInfo.frames}`;
    $("motion-image").src=`/api/motion/frame?source=${encodeURIComponent(state.motionSource)}&number=${state.motionFrame}`;
    await loadFrameCells();
  }
  async function loadFrameCells(){
    if(!$("motion-frame-cells"))return;
    try{const found=await api("/api/motion/action",{method:"POST",body:JSON.stringify({action:"frame_cells",params:{source:state.motionSource,frame:state.motionFrame}})});
      const cells=found.cells||found.identities||found;
      const values=Array.isArray(cells)?cells:Object.values(cells);
      $("motion-frame-cells").innerHTML="";
      for(const cell of values){const id=typeof cell==="object"?(cell.identity??cell.id):cell;if(id===undefined)continue;
        const button=document.createElement("button");button.className="button secondary";button.textContent=`Cell ${id}`;button.onclick=()=>showMotionTrack(id);$("motion-frame-cells").append(button);}
      if(!$("motion-frame-cells").children.length)$("motion-frame-cells").textContent=JSON.stringify(found);
    }catch(error){$("motion-frame-cells").textContent=error.message;}
  }
  async function showMotionTrack(identity){
    try{const result=await api("/api/motion/action",{method:"POST",body:JSON.stringify({action:"track",params:{source:state.motionSource,identity}})});
      if($("motion-track-detail"))$("motion-track-detail").textContent=JSON.stringify(result,null,2);
      else alert(JSON.stringify(result,null,2));
    }catch(error){alert(error.message);}
  }
  async function showMotionAction(name){
    try{const spec=await api(`/api/motion/actions/${encodeURIComponent(name)}`);state.motionAction=spec;
      $("motion-action-title").textContent=name.replaceAll("_"," ");$("motion-action-summary").textContent=spec.summary;
      $("motion-action-message").textContent=spec.destructive?"This removes labels in the new session. The selected source is preserved.":"";
      const container=$("motion-fields");container.innerHTML="";
      for(const param of spec.params){if(param.name==="source")continue;container.append(field(param));}
      $("motion-form").querySelector("button[type=submit]").textContent=sessionEdits.has(name)?"Save new session":spec.mutates?"Save output":"Run read-only action";
      $("motion-dialog").showModal();
    }catch(error){alert(error.message);}
  }
  async function runMotionAction(){
    try{const params=state.motionAction.params.some(row=>row.name==="source")?{source:state.motionSource}:{};
      for(const input of $("motion-fields").querySelectorAll("input,select,textarea")){const value=parseValue(input,input.dataset.type);if(value!==undefined)params[input.name]=value;}
      const result=await api("/api/motion/action",{method:"POST",body:JSON.stringify({action:state.motionAction.action,params})});
      $("motion-dialog").close();
      if(result.session){await refreshMotionSources();$("motion-source").value=result.session;await chooseMotionSource(result.session);}
      else{const box=motionSection("Action result",state.motionAction.summary);const pre=document.createElement("pre");pre.className="evidence-json";pre.textContent=JSON.stringify(result,null,2);box.append(pre);}
    }catch(error){$("motion-action-message").textContent=error.message;}
  }
  function showRefinement(handoff){
    state.handoff=handoff;
    $("refinement-method").value="single_frame";
    $("refinement-options").value="{}";
    $("refinement-weights").value=localStorage.getItem("pm-refinement-weights")||"";
    $("refinement-remove").checked=false;
    $("refinement-message").textContent=state.config?.mask_weights_configured||$("refinement-weights").value?"":"Select the accepted model.pt file for the single-frame method.";
    $("refinement-dialog").showModal();
  }
  async function runRefinement(){
    try{
      const chosen=$("refinement-method").value;
      if(chosen==="off"){
        $("refinement-dialog").close();setWorkspace("micro");state.page="results";renderNav();renderPage();
        showAction("measure",state.handoff.measure_params);return;
      }
      const method=chosen==="custom"?$("refinement-custom").value.trim():chosen;
      const options=JSON.parse($("refinement-options").value||"{}");
      if(!options||Array.isArray(options)||typeof options!=="object")throw Error("Method options must be a JSON object");
      if($("refinement-remove").checked)options.policy="intersection";
      const weights=$("refinement-weights").value.trim()||null;
      if(method==="single_frame"&&!weights&&!state.config?.mask_weights_configured)throw Error("Select the trained single-frame model.pt file first");
      if(weights)localStorage.setItem("pm-refinement-weights",weights);
      const job=await api("/api/refinements",{method:"POST",body:JSON.stringify({manifest:state.handoff.manifest,method,options,weights})});
      $("refinement-dialog").close();showJobs();await refreshJobs();watchJob(job.id);
    }catch(error){$("refinement-message").textContent=error.message;}
  }
  async function initialize(){
    if(localStorage.getItem("pm-workspace")==="events"){localStorage.setItem("pm-workspace","micro");localStorage.setItem("pm-page","events");}
    if(localStorage.getItem("pm-workspace")==="micro"&&localStorage.getItem("pm-page")==="events"){$("run-path").value=new URL(location.href).searchParams.get("run")||localStorage.getItem("pm-run-folder")||"";state.page="events";setWorkspace("micro");}
    try{[state.config,state.map,state.explorationCatalogue]=await Promise.all([api("/api/config"),api("/api/catalogue"),api("/api/exploration/catalogue")]);
      const lastPage=localStorage.getItem("pm-page");
      if(lastPage && ["results","all_methods","pipelines","follow","tissue","compare","events","rhythm-audit"].includes(lastPage))state.page=lastPage;
      const url=new URL(window.location.href),requestedRun=url.searchParams.get("run");
      if(requestedRun){url.searchParams.delete("run");history.replaceState(null,"",url);}
      const savedRunFolder=requestedRun||localStorage.getItem("pm-run-folder")||"";
      if(requestedRun)localStorage.setItem("pm-run-folder",savedRunFolder);
      $("run-path").value=savedRunFolder;
      state.motionSource=localStorage.getItem("pm-motion-source")||"";
      renderNav();renderPage();await refreshJobs();
      try{state.recentRuns=(await api("/api/runs")).runs||[];}catch{/* The audit index is optional. */}
      const lastWorkspace=localStorage.getItem("pm-workspace");
       if(["micro","motion","auto"].includes(lastWorkspace))setWorkspace(lastWorkspace);
       if(savedRunFolder && $("run-path").value.trim()===savedRunFolder&&!(state.workspace==="micro"&&state.page==="events"))selectRun(false);
    }catch(error){$("micro-content").innerHTML=`<div class="empty-card">${escapeHtml(error.message)}</div>`;}
  }
  document.querySelectorAll(".top-tabs button").forEach(button=>button.onclick=()=>setWorkspace(button.dataset.workspace));
  function updateExtrasTheme(){const setting=localStorage.getItem("pm-extras-theme")||"auto";document.documentElement.dataset.pmTheme=setting==="auto"?(matchMedia("(prefers-color-scheme: light)").matches?"light":"dark"):setting;$("micro-theme").value=setting;}
  $("micro-theme").onchange=()=>{localStorage.setItem("pm-extras-theme",$("micro-theme").value);updateExtrasTheme();};
  matchMedia("(prefers-color-scheme: light)").addEventListener("change",()=>{if((localStorage.getItem("pm-extras-theme")||"auto")==="auto")updateExtrasTheme();});
  updateExtrasTheme();
  $("open-results").onclick=()=>selectRun(true);
  $("run-path").onchange=()=>selectRun(false);
  window.addEventListener("pywebviewready",()=>{$("browse-results").hidden=false;$("browse-weights").hidden=false;});
  $("browse-results").onclick=async()=>{const folder=await window.pywebview?.api?.choose_folder($("run-path").value.trim());
    if(folder){$("run-path").value=folder;selectRun(false);}};
  $("browse-weights").onclick=async()=>{const file=await window.pywebview?.api?.choose_model($("refinement-weights").value.trim());
    if(file){$("refinement-weights").value=file;$("refinement-message").textContent="";}};
  $("close-action").onclick=()=>$("action-dialog").close();
  $("validate-action").onclick=()=>checkAction(false);
  $("action-form").onsubmit=event=>{event.preventDefault();checkAction(true);};
  $("job-toggle").onclick=()=>{$("job-tray").hidden=!$("job-tray").hidden;refreshJobs();};
  $("close-jobs").onclick=()=>$("job-tray").hidden=true;
  $("motion-source").onchange=e=>chooseMotionSource(e.target.value);
  $("motion-refresh").onclick=refreshMotionSources;
  document.querySelectorAll("[data-motion-page]").forEach(button=>button.onclick=()=>{state.motionPage=button.dataset.motionPage;renderMotionPage();});
  $("close-motion-action").onclick=()=>$("motion-dialog").close();
  $("motion-form").onsubmit=event=>{event.preventDefault();runMotionAction();};
  $("close-refinement").onclick=()=>$("refinement-dialog").close();
  $("refinement-method").onchange=()=>{$("refinement-custom-field").hidden=$("refinement-method").value!=="custom";
    $("refinement-form").querySelector("button[type=submit]").textContent=$("refinement-method").value==="off"?"Continue to measurements":"Prepare refined masks";};
  $("refinement-form").onsubmit=event=>{event.preventDefault();runRefinement();};
  initialize();
})();
