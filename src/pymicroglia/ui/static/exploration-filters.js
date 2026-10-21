/* A single registry-driven editor for nested cell-cohort expressions. */
(() => {
  "use strict";
  const copy=value=>JSON.parse(JSON.stringify(value)),escape=value=>String(value??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"})[c]);
  class CellFilters {
    constructor(explorer,context={}){
      this.context=context;
      this.explorer=explorer;this.schema=null;this.epoch=0;this.tree={op:"OR",children:[]};this.applied=null;this.preview=null;this.mode="exclude";this.unknown="keep";this.decisions=null;this.greyColour="#aaaaaa";
      this.$("toggle").onclick=()=>{const body=this.$("panel");body.hidden=!body.hidden;this.$("toggle").setAttribute("aria-expanded",String(!body.hidden));if(!body.hidden&&!this.schema)this.load();};
      this.$("tree").addEventListener("click",event=>this.edit(event));
      this.$("tree").addEventListener("change",event=>this.change(event));
      this.$("mode").onchange=()=>{this.mode=this.$("mode").value;this.dirty();};
      this.$("unknown").onchange=()=>{this.unknown=this.$("unknown").value;this.dirty();};
      this.$("preview").onclick=()=>this.evaluate(false);this.$("apply").onclick=()=>this.evaluate(true);
      this.$("off").onclick=()=>this.off();this.$("clear").onclick=()=>{this.tree={op:"OR",children:[]};this.render();this.dirty();};
      this.$("download").onclick=()=>this.download("json");this.$("script").onclick=()=>this.download("py");
      this.$("import").onchange=async event=>{const file=event.target.files[0];if(!file)return;try{const record=JSON.parse(await file.text());if(record.schema_version!==1)throw Error("Unsupported filter file version");this.tree=copy(record.expression);this.mode=record.mode;this.unknown=record.unknown;this.decisions=record.decisions||null;this.greyColour=record.grey_colour||"#aaaaaa";await this.load();this.render();this.dirty();await this.evaluate(false);}catch(error){this.$("status").textContent=error.message;}event.target.value="";};
      this.installDecisionControls();
      this.$("recipes").onclick=event=>{const button=event.target.closest("[data-filter-recipe]");if(!button||!this.schema)return;const preset={nonrhythmic:["rhythmic","eq",false],gap:["longest_gap_frames","gte",3],missing:["missing_pct","gt",10],merge:["has_merge","eq",true],near:["min_distance_px","lt",20]}[button.dataset.filterRecipe];if(!preset)return;const rule={field:preset[0],cmp:preset[1],value:preset[2]};if(this.field(rule).metric)rule.metric=this.defaultMetric();if(this.tree.op!=="AND"&&this.tree.op!=="OR"&&this.tree.op!=="XOR")this.tree={op:"OR",children:[this.tree]};this.tree.children.push(rule);this.render();this.dirty();};
    }
    static mountGrey(explorer){
      const panel=explorer.$("filter-panel").cloneNode(true);
      panel.id="xp-grey-filter-panel";panel.setAttribute("aria-label","Grey outline filter builder");
      panel.querySelectorAll("[id]").forEach(el=>el.id=el.id.replace("xp-filter-","xp-grey-filter-"));
      const toolbar=document.createElement("div");toolbar.className="controls";
      toolbar.innerHTML='<button id="xp-grey-filter-toggle" type="button" aria-controls="xp-grey-filter-panel" aria-expanded="false" title="Show cells failing configurable rules in grey; retain all measurements and cells">Grey outlines</button><span id="xp-grey-filter-count" class="xp-filter-count">All cells</span>';
      explorer.host.querySelector(".transport").after(toolbar,panel);
      const el=key=>explorer.$("grey-filter-"+key);
      el("mode").options[0].textContent="Grey matching cells";el("mode").options[1].textContent="Colour only matching cells";
      el("mode").title="Choose which cells retain their original outline colour; other cells remain visible in grey";
      el("unknown").options[0].textContent="Keep original colour";el("unknown").options[1].textContent="Show in grey";
      el("apply").title="Apply outline colours to images and playback without changing the cohort";
      el("off").title="Restore original outline colours, retaining the draft rules";
      const editor=new CellFilters(explorer,{grey:true,element:el,analysisIds:()=>explorer.rhythmAnalyses?.filterAnalysisIds(),onApply:result=>explorer.applyOutlineFilter(result)});
      for(const [action,label] of [["tracked_cell_image","Export outlined image"],["tracked_cell_video","Export outlined video"]]){
        const button=document.createElement("button");button.type="button";button.textContent=label;button.title="Configure a native export using the applied outline filter";
        button.onclick=()=>{const result=editor.applied;if(!result){el("status").textContent="Apply an outline filter before exporting";return;}explorer.options.onAction?.(action,{...result.render_sources,cell_filter:result,filter_grey_colour:result.grey_colour,...(action==="tracked_cell_image"?{frame_index:explorer.state.frame}:{})});};
        panel.querySelector(".controls").append(button);
      }
      return editor;
    }
    installDecisionControls(){
      this.$("panel").querySelectorAll("[data-decision-controls]").forEach(el=>el.remove());
      const label=document.createElement("label");label.dataset.decisionControls="true";label.textContent="Import decisions ";label.title="Per-cell pass/fail CSV: identity and passed or passes_fixed_circle, optionally recording";
      const input=document.createElement("input");input.type="file";input.accept=".csv,text/csv";input.setAttribute("aria-label","Import per-cell decisions");label.append(input);
      this.decisionsNote=document.createElement("span");this.decisionsNote.dataset.decisionControls="true";this.$("panel").querySelector(".xp-filter-shortcuts").append(label,this.decisionsNote);
      input.onchange=async()=>{const file=input.files[0],source=this.explorer.sourceData;if(!file||!source)return;const epoch=++this.epoch;try{const decisions=await this.explorer.options.api("/api/exploration/filter-decisions",{method:"POST",body:JSON.stringify({text:await file.text(),stem:source.stem,filename:file.name})});if(epoch!==this.epoch||source!==this.explorer.sourceData)return;this.decisions=decisions;await this.load();this.tree={field:"review_passed",cmp:"eq",value:true};this.mode="keep";this.render();this.dirty();}catch(error){if(epoch===this.epoch)this.$("status").textContent=error.message;}input.value="";};
      if(this.context.grey){const grey=document.createElement("label");grey.textContent="Grey colour ";grey.title="Outline and identity colour for cells that fail the applied rules";this.greyInput=document.createElement("input");this.greyInput.type="color";this.greyInput.value=this.greyColour;this.greyInput.setAttribute("aria-label","Filtered cell outline colour");grey.append(this.greyInput);this.$("panel").querySelector(".controls").append(grey);this.greyInput.onchange=()=>{this.greyColour=this.greyInput.value;this.dirty();};}
    }
    $(id){return this.context.element?.(id)||this.explorer.$("filter-"+id);}
    field(node){return this.schema?.fields.find(f=>f.key===node.field);}
    defaultMetric(){if(this.context.metric)return this.context.metric();const tested=this.explorer.sourceData.results?.rhythms?.metrics||[];return tested[0]||this.schema.metrics[0]?.key||"";}
    save(){const applied=this.applied||this.savedApplied;return {schema_version:1,expression:copy(this.tree),mode:this.mode,unknown:this.unknown,decisions:this.decisions,grey_colour:this.greyColour,applied:applied?{expression:copy(applied.expression),mode:applied.mode,unknown:applied.unknown,decisions:applied.decisions||null,grey_colour:applied.grey_colour||this.greyColour,analysis_ids:applied.analysis_ids}:null,open:!this.$("panel").hidden};}
    async reset(saved){
      ++this.epoch;this.$("toggle").disabled=!this.explorer.data;this.$("preview").disabled=true;this.$("off").disabled=true;this.schema=null;this.preview=null;this.applied=null;this.$("evidence").replaceChildren();this.$("apply").disabled=true;this.$("download").disabled=true;this.$("script").disabled=true;
      this.tree=copy(saved?.schema_version===1?saved.expression:{op:"OR",children:[]});this.mode=saved?.mode||"exclude";this.unknown=saved?.unknown||"keep";this.decisions=saved?.decisions||null;this.greyColour=saved?.grey_colour||"#aaaaaa";this.savedApplied=saved?.applied||null;
      this.$("panel").hidden=!saved?.open;this.$("toggle").setAttribute("aria-expanded",String(!!saved?.open));this.$("status").textContent="";this.$("count").textContent="All cells";
      if(saved?.open||this.savedApplied){await this.load();if(this.savedApplied){const pending={expression:this.tree,mode:this.mode,unknown:this.unknown,decisions:this.decisions,grey_colour:this.greyColour};await this.evaluate(true,this.savedApplied);this.tree=pending.expression;this.mode=pending.mode;this.unknown=pending.unknown;this.decisions=pending.decisions;this.greyColour=pending.grey_colour;this.render();this.savedApplied=null;}}
    }
    async load(){
      const epoch=++this.epoch,data=this.explorer.data;if(!data)return;
      this.$("status").textContent="Loading filter choices…";this.$("preview").disabled=true;
      try{const schema=await this.explorer.options.api(this.explorer.url("filter-options"));if(epoch!==this.epoch)return;this.schema=schema;this.$("preview").disabled=false;this.$("apply").disabled=false;this.render();this.$("status").textContent="";}
      catch(error){if(epoch===this.epoch)this.$("status").textContent=error.message;}
    }
    at(path){let node=this.tree;for(const index of path)node=node.children[index];return node;}
    replace(path,node){if(!path.length)this.tree=node;else this.at(path.slice(0,-1)).children[path.at(-1)]=node;}
    rule(){return {field:"longest_gap_frames",cmp:"gte",value:3};}
    render(){
      if(!this.schema)return;
      this.$("mode").value=this.mode;this.$("unknown").value=this.unknown;if(this.greyInput)this.greyInput.value=this.greyColour;if(this.decisionsNote)this.decisionsNote.textContent=this.decisions?`${Object.keys(this.decisions.values).length} imported decisions`:"";
      const opts=(values,selected)=>Object.entries(values).map(([key,label])=>`<option value="${escape(key)}" ${String(selected)===key?"selected":""}>${escape(label)}</option>`).join("");
      const render=(node,path=[])=>{
        const encoded=path.join("."),attrs=`data-path="${encoded}"`,buttons=`<button type="button" data-edit="not" ${attrs} title="${node.op==="NOT"?"Remove NOT":"Invert this rule or group; unknown stays unknown"}">${node.op==="NOT"?"Undo NOT":"NOT"}</button>${path.length?`<button type="button" data-edit="remove" ${attrs} title="Remove this rule or group">Remove</button>`:""}`;
        if(node.op){const controls=node.op==="NOT"?"<strong>NOT</strong>":`<label>Match <select data-change="op" ${attrs} title="AND: all; OR: any; XOR: odd number of matching rules">${opts({AND:"AND — all",OR:"OR — any",XOR:"XOR — odd count"},node.op)}</select></label><button type="button" data-edit="rule" ${attrs} title="Add a condition within this group">+ Rule</button><button type="button" data-edit="group" ${attrs} title="Add a nested group with its own logical operator">+ Group</button>`;return `<fieldset class="xp-filter-group" ${attrs}><legend>${node.op==="NOT"?"Negated condition":"Condition group"}</legend><div class="xp-filter-row">${controls}${buttons}</div>${node.children.map((child,index)=>render(child,[...path,index])).join("")}${!node.children.length?'<span class="xp-filter-empty">Add a rule or use a shortcut below.</span>':""}</fieldset>`;}
        const field=this.field(node);if(!field)return `<div class="xp-filter-row">Unknown field: ${escape(node.field)} ${buttons}</div>`;
        const groups=[...new Set(this.schema.fields.map(f=>f.group))],fields=groups.map(g=>`<optgroup label="${escape(g)}">${this.schema.fields.filter(f=>f.group===g).map(f=>`<option value="${escape(f.key)}" ${f.key===node.field?"selected":""}>${escape(f.label)}</option>`).join("")}</optgroup>`).join("");
        const metric=field.metric?`<select aria-label="Measurement for ${escape(field.label)}" data-change="metric" ${attrs} title="Saved measurement and test result used by this rule">${this.schema.metrics.map(m=>`<option value="${escape(m.key)}" ${m.key===node.metric?"selected":""}>${escape(m.label)}${m.unit?" ("+escape(m.unit)+")":""}</option>`).join("")}</select>`:"";
        const boolean=field.kind==="bool",range=["between","not_between"].includes(node.cmp),values=range?node.value:[node.value],value=boolean?`<select data-change="value" ${attrs} aria-label="Condition value" title="Match true or false">${opts({true:"Yes",false:"No"},node.value)}</select>`:(range?values:[node.value]).map((v,i)=>`<input type="number" step="any" aria-label="${range?(i?"Upper":"Lower"):"Threshold"}" data-change="value" data-bound="${i}" ${attrs} value="${escape(v)}" title="${escape(field.help)}">`).join(range?"<span>to</span>":"");
        return `<div class="xp-filter-row xp-filter-rule"><select data-change="field" ${attrs} aria-label="Cell criterion" title="${escape(field.help)}">${fields}</select>${metric}<select data-change="cmp" ${attrs} aria-label="Comparison" title="Comparison uses exact bounds; between includes both endpoints">${opts(field.comparators,node.cmp)}</select>${value}${buttons}</div>`;
      };
      this.$("tree").innerHTML=render(this.tree);
    }
    edit(event){const button=event.target.closest("[data-edit]");if(!button)return;const path=button.dataset.path?button.dataset.path.split(".").map(Number):[],node=this.at(path);switch(button.dataset.edit){case "rule":node.children.push(this.rule());break;case "group":node.children.push({op:"AND",children:[this.rule()]});break;case "not":this.replace(path,node.op==="NOT"?node.children[0]:{op:"NOT",children:[node]});break;case "remove":this.at(path.slice(0,-1)).children.splice(path.at(-1),1);break;}this.render();this.dirty();}
    change(event){const el=event.target;if(!el.dataset.change)return;const node=this.at(el.dataset.path?el.dataset.path.split(".").map(Number):[]);switch(el.dataset.change){case "field":node.field=el.value;node.cmp="eq";node.value=this.field(node).kind==="bool"?true:0;if(this.field(node).metric)node.metric=this.defaultMetric();else delete node.metric;break;case "cmp":node.cmp=el.value;if(["between","not_between"].includes(node.cmp))node.value=Array.isArray(node.value)?node.value:[node.value,node.value];else if(Array.isArray(node.value))node.value=node.value[0];break;case "value":{const value=this.field(node).kind==="bool"?el.value==="true":el.value===""?null:Number(el.value);if(Array.isArray(node.value))node.value[Number(el.dataset.bound)]=value;else node.value=value;break;}default:node[el.dataset.change]=el.value;}this.render();this.dirty();}
    dirty(){++this.epoch;this.preview=null;this.$("preview").disabled=!this.schema;this.$("apply").disabled=!this.schema;this.$("download").disabled=!this.applied;this.$("script").disabled=!this.applied;if(this.applied)this.evidence(this.applied,"Applied");else this.$("evidence").replaceChildren();this.$("status").textContent="Pending filter edits — Preview or Apply";this.context.onDraft?.(this.save());this.explorer.saveSession();}
    async evaluate(apply=false,override=null){
      if(!this.explorer.data)return;const epoch=++this.epoch,request=copy(override||{expression:this.tree,mode:this.mode,unknown:this.unknown,decisions:this.decisions,grey_colour:this.greyColour}),source=this.explorer.sourceData;
      this.$("status").textContent="Checking exact frames and saved evidence…";this.$("preview").disabled=true;this.$("apply").disabled=true;
      try{const result=await this.explorer.options.api("/api/exploration/filter-preview",{method:"POST",body:JSON.stringify({run:source.run,stem:source.stem,...request,...(this.context.analysisIds?{analysis_ids:this.context.analysisIds()}: {})})});if(epoch!==this.epoch||source!==this.explorer.sourceData)return;result.grey_colour=request.grey_colour||this.greyColour;this.preview=result;this.$("download").disabled=false;this.$("script").disabled=false;
        this.$("status").textContent=`${apply?"Applied":"Preview"}: ${result.included} / ${result.total} cells kept · ${result.excluded} ${this.context.grey?"grey":"excluded"} · ${result.unknown_count} unknown${result.unknown_count?" ("+request.unknown+")":""}`;
        this.evidence(result,apply?"Applied":"Preview");if(apply){this.applied=result;this.apply(result);this.$("count").textContent=this.context.grey?`${result.excluded} grey / ${result.total} cells`:`${result.included} / ${result.total} cells`;this.$("off").disabled=false;this.explorer.saveSession();}
      }catch(error){if(epoch===this.epoch)this.$("status").textContent=error.message;}
      finally{if(epoch===this.epoch){this.$("preview").disabled=false;this.$("apply").disabled=false;}}
    }
    evidence(result,label="Preview"){
      const name=rule=>this.field(rule)?.label||rule.field,format=value=>value===null?"Unknown":typeof value==="number"?String(Number(value.toPrecision(5))):String(value);
      this.$("evidence").innerHTML=`<table><caption>${escape(label)} rules · ${escape(result.mode)} matching cells · unknown: ${escape(result.unknown)}</caption><thead><tr><th>Cell</th><th>Outcome</th><th>Evidence for each rule</th></tr></thead><tbody>${result.rows.map(row=>`<tr><td>${row.identity}</td><td>${row.included?"Keep":this.context.grey?"Grey":"Exclude"}${row.match===null?" · unknown":""}</td><td>${row.evidence.map(e=>`${escape(name(e.rule))}${e.rule.metric?" ["+escape(e.rule.metric)+"]":""} ${escape(this.field(e.rule)?.comparators[e.rule.cmp]||e.rule.cmp)} ${escape(JSON.stringify(e.rule.value))}: ${e.no_neighbour?"no other cell":escape(format(e.actual))} (${e.match===null?"unknown":e.match?"matches":"does not match"})`).join("; ")}</td></tr>`).join("")}</tbody></table>`;
    }
    off(){++this.epoch;this.applied=null;this.savedApplied=null;this.preview=null;this.apply(null);this.$("count").textContent="All cells";this.$("status").textContent="Filtering off — all cells restored";this.$("off").disabled=true;this.$("preview").disabled=false;this.$("apply").disabled=!this.schema;this.$("download").disabled=true;this.$("script").disabled=true;this.explorer.saveSession();}
    apply(result){if(this.context.onApply)this.context.onApply(result);else this.explorer.applyCohort(result);}
    refreshApplied(){if(this.applied)this.evaluate(true,{expression:this.applied.expression,mode:this.applied.mode,unknown:this.applied.unknown,decisions:this.applied.decisions,grey_colour:this.applied.grey_colour,analysis_ids:this.applied.analysis_ids});}
    download(type){const result=this.preview||this.applied;if(!result)return;const data=type==="py"?result.equivalent_script:JSON.stringify(result,null,2),url=URL.createObjectURL(new Blob([data],{type:type==="py"?"text/x-python":"application/json"})),link=document.createElement("a");link.href=url;link.download=`${result.stem}_cell_filter.${type}`;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
  }
  window.PyMicrogliaCellFilters=CellFilters;
})();
