/* Recording assignments and native comparisons through the shared plot cards. */
(() => {
  'use strict';
  const copy=x=>JSON.parse(JSON.stringify(x)),esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  class GroupComparison {
    constructor(explorer){
      this.e=explorer;this.epoch=0;this.states=new Map();this.selected=new Set();this.design=null;this.schema=null;
      this.saveQueue=Promise.resolve();this.rhythmHistories=new Map();this.rhythmPending=new Set();
      const saveSession=explorer.saveSession.bind(explorer);explorer.saveSession=(...args)=>{saveSession(...args);this.checkpoint();};
      const $=id=>explorer.$('comparison-'+id);
      $('assign').onclick=()=>this.assignCondition();
      $('select-all').onclick=()=>{this.selected=new Set(this.design.recordings.map(row=>row.id));this.renderAssignments();};
      $('clear-selection').onclick=()=>{this.selected.clear();this.renderAssignments();};
      for(const key of ['target','cell-recording','cell-ids','pattern','ignore-case'])$(''+key).addEventListener(key==='cell-ids'||key==='pattern'?'input':'change',()=>this.previewSelection());
      $('condition').oninput=()=>this.assignmentButton();
      $('add-recordings').onclick=()=>this.addRecordings();
      $('recordings').onchange=event=>{const input=event.target,row=this.design.recordings.find(r=>r.id===input.dataset.recording);if(!row)return;if(input.dataset.field==='selected'){input.checked?this.selected.add(row.id):this.selected.delete(row.id);this.previewSelection();return;}if(input.dataset.field.startsWith('factor:'))row.factors[input.dataset.field.slice(7)]=input.value;else if(input.dataset.field.startsWith('analysis:')){row.analysis_ids??={};const metric=input.dataset.field.slice(9);if(input.value)row.analysis_ids[metric]=input.value;else delete row.analysis_ids[metric];}else row[input.dataset.field]=input.type==='checkbox'?input.checked:input.value;this.changed();};
      $('condition-list').onchange=event=>{const input=event.target,row=this.design.conditions.find(r=>r.id===input.dataset.condition);if(!row)return;row[input.dataset.field]=input.type==='checkbox'?input.checked:input.value;this.renderAssignments();this.changed();};
      $('factor-names').onchange=()=>{this.design.factor_names=$('factor-names').value.split(',').map(v=>v.trim()).filter(Boolean);this.renderAssignments();this.changed();};
      $('show-samples').onchange=()=>{this.design.show_samples=$('show-samples').checked;this.renderAssignments();this.changed();};
      $('show-rhythms').onchange=()=>{this.design.show_rhythms=$('show-rhythms').checked;this.renderAssignments();this.changed();};
      $('download-design').onclick=async()=>{try{const design=await explorer.options.api('/api/exploration/comparison-design',{method:'POST',body:JSON.stringify({run:this.run,design:this.save()})});this.download(JSON.stringify(design,null,2),'comparison-design.json','application/json');}catch(error){$('design-status').textContent=error.message;}};
      $('condition-list').onclick=event=>{const button=event.target.closest('[data-condition-order]');if(!button)return;const index=this.design.conditions.findIndex(row=>row.id===button.dataset.condition),next=index+Number(button.dataset.conditionOrder);if(next<0||next>=this.design.conditions.length)return;[this.design.conditions[index],this.design.conditions[next]]=[this.design.conditions[next],this.design.conditions[index]];this.renderAssignments();this.changed();};
      $('import-design').onchange=async event=>{const file=event.target.files[0];if(!file)return;try{const incoming=JSON.parse(await file.text());this.design=await explorer.options.api('/api/exploration/comparison-workspace',{method:'PUT',body:JSON.stringify({run:this.run,design:incoming})});this.restorePlots();this.renderAssignments();this.changed();}catch(error){$('design-status').textContent=error.message;}event.target.value='';};
      this.resizeObserver=new ResizeObserver(entries=>{const width=Math.round(entries[0].contentRect.width);if(width===this.lastWidth)return;this.lastWidth=width;clearTimeout(this.resizeTimer);this.resizeTimer=setTimeout(()=>{if(this.design)this.drawAll();},120);});
      this.resizeObserver.observe(explorer.$('compare'));
    }
    element(key,view='distribution'){return this.e.comparisons.element('distribution-'+key,view);}
    async addRecordings(){
      if(!this.design||this.addingRecordings)return;
      const field=this.e.$('comparison-add-run'),button=this.e.$('comparison-add-recordings'),status=this.e.$('comparison-design-status'),runs=field.value.split(/\r?\n/).map(value=>value.trim()).filter(Boolean),epoch=this.epoch;
      if(!runs.length){status.textContent='Enter saved measurement folders, one per line.';return;}
      this.addingRecordings=true;button.disabled=true;status.textContent='Loading recordings…';
      try{
        const design=await this.e.options.api('/api/exploration/comparison-recordings',{method:'POST',body:JSON.stringify({runs,design:this.save()})});
        if(epoch!==this.epoch)return;this.design=design;field.value='';this.renderAssignments();this.changed();
      }catch(error){if(epoch===this.epoch)status.textContent=error.message;}
      finally{this.addingRecordings=false;button.disabled=false;}
    }
    selection(){
      const $=key=>this.e.$('comparison-'+key),mode=$('target').value;
      return {mode,recording_ids:mode==='cells'?[$('cell-recording').value].filter(Boolean):[...this.selected],ids:$('cell-ids').value,pattern:$('pattern').value,ignore_case:$('ignore-case').checked};
    }
    assignmentButton(){this.e.$('comparison-assign').disabled=this.assigning||!this.selectionPreview?.cells||!this.e.$('comparison-condition').value.trim();}
    previewSelection(){
      if(!this.design)return;const $=key=>this.e.$('comparison-'+key),selection=this.selection(),token=this.selectionToken=(this.selectionToken||0)+1,epoch=this.epoch;
      clearTimeout(this.selectionTimer);this.selectionPreview=null;this.assignmentButton();
      for(const mode of ['cell','recording','pattern'])$(mode+'-targets').hidden=selection.mode!==({cell:'cells',recording:'recordings',pattern:'pattern'}[mode]);
      for(const input of $('recordings').querySelectorAll('[data-field=selected]')){input.disabled=selection.mode!=='recordings';input.checked=selection.mode==='recordings'&&this.selected.has(input.dataset.recording);}
      for(const row of $('recordings').rows)row.cells[0].hidden=selection.mode==='cells';
      $('recordings').tHead.rows[0].cells[0].textContent=selection.mode==='pattern'?'Matches':'Select';
      $('selection-details').hidden=true;
      if(selection.mode==='cells'&&!selection.ids.trim()||selection.mode==='pattern'&&!selection.pattern.trim()||selection.mode==='recordings'&&!selection.recording_ids.length){$('selection-status').textContent=selection.mode==='cells'?'Enter cell IDs to select cells.':selection.mode==='pattern'?'Enter a regular expression to match recording names.':'Select recordings in the table.';return;}
      $('selection-status').textContent='Checking selection…';
      this.selectionTimer=setTimeout(async()=>{
        try{
          const preview=await this.e.options.api('/api/exploration/comparison-selection',{method:'POST',body:JSON.stringify({design:this.design,selection})});
          if(token!==this.selectionToken||epoch!==this.epoch)return;this.selectionPreview=preview;this.assignmentButton();
          const missing=preview.targets.reduce((sum,row)=>sum+row.missing_ids.length,0);
          $('selection-status').textContent=`${preview.cells} cells in ${preview.recordings} recording${preview.recordings===1?'':'s'} selected${missing?`; ${missing} IDs not found`:''}.`;
          $('selection-details').hidden=!preview.targets.length;
          $('selection-list').innerHTML=preview.targets.map(row=>`<div><strong>${esc(row.stem)}</strong>: ${row.identities.length} cells<details><summary>Cell IDs</summary>${esc(row.identities.join(', '))}${row.missing_ids.length?`<p>Not found: ${esc(row.missing_ids.join(', '))}</p>`:''}</details></div>`).join('');
          if(selection.mode==='pattern')for(const input of $('recordings').querySelectorAll('[data-field=selected]'))input.checked=preview.targets.some(row=>row.recording_id===input.dataset.recording&&row.identities.length);
        }catch(error){if(token===this.selectionToken&&epoch===this.epoch){$('selection-status').textContent=error.message;this.assignmentButton();}}
      },180);
    }
    async assignCondition(){
      if(!this.selectionPreview?.cells||this.assigning)return;const $=key=>this.e.$('comparison-'+key),epoch=this.epoch,selection=this.selection(),signature=this.selectionPreview.signature,label=$('condition').value.trim();
      this.assigning=true;this.assignmentButton();$('selection-status').textContent='Setting condition…';
      try{
        clearTimeout(this.saveTimer);await this.saveQueue.catch(()=>{});
        const response=await this.e.options.api('/api/exploration/comparison-assign',{method:'POST',body:JSON.stringify({run:this.run,design:this.save(),selection,expected_signature:signature,condition:label})});
        if(epoch!==this.epoch)return;this.design=response.design;this.renderAssignments();this.changed();
        $('selection-status').textContent=`${response.selection.cells} cells set to ${response.condition}.`;
      }catch(error){if(epoch===this.epoch)$('selection-status').textContent=error.message;}
      finally{this.assigning=false;this.assignmentButton();}
    }
    value(key,view){return this.element(key,view).value;}
    views(){return [...this.e.comparisons.views.values()].filter(view=>view.base==='distribution');}
    async load(data){
      const epoch=++this.epoch;clearTimeout(this.saveTimer);
      for(const state of this.states.values())if(!state.preview)state.key=null;
      if(this.design?.recordings.some(row=>row.run===data.run&&row.stem===data.stem)){this.restorePlots();this.renderAssignments();this.drawAll();return;}
      if(this.run===data.run&&this.design){this.renderAssignments();this.drawAll();return;}
      this.design=null;this.run=data.run;this.states.clear();this.selected.clear();
      this.e.$('comparison-design-status').textContent='Loading assignments…';
      try{
        const [schema,design]=await Promise.all([this.schema||this.e.options.api('/api/exploration/comparison-options'),this.e.options.api('/api/exploration/comparison-workspace?'+new URLSearchParams({run:data.run}))]);
        if(epoch!==this.epoch)return;this.schema=schema;this.design=design;this.loading=true;
        try{const draft=JSON.parse(localStorage.getItem('pm-comparison-draft:'+data.run)||'null');if(draft?.pending)this.design=draft.design;}catch{/* Browser recovery is optional. */}
        this.restorePlots();this.renderAssignments();this.loading=false;this.lastSnapshot=JSON.stringify(design);this.e.$('comparison-design-status').textContent='';this.checkpoint();this.drawAll();
      }catch(error){if(epoch===this.epoch)this.e.$('comparison-design-status').textContent=error.message;}
    }
    restorePlots(){
      const wanted=new Set(this.design.layout?.plots||[]);for(const view of this.e.comparisons.copies('distribution'))if(!wanted.has(view.id))this.e.comparisons.remove(view.id);
      if(this.design.layout?.plots){for(const id of this.design.layout.plots)this.e.comparisons.add('distribution',id);const group=this.e.comparisons.groups.get('distribution');group.columns.value=String(this.design.layout.columns||1);this.e.comparisons.layout(group);}
      for(const view of this.views()){
        this.bindPanel(view);const saved=this.design.plots?.[view.id];if(!saved)continue;
        for(const [key,value] of Object.entries(saved.controls||{})){const element=this.element(key,view.id);if(!element)continue;if(key==='groups'){this.states.get(view.id).savedGroups=value;continue;}if(element.multiple)for(const option of element.options)option.selected=value.includes(option.value);else element.value=value;}
        if(saved.appearance)this.e.styles[view.id]={...saved.appearance};
        this.states.get(view.id).resultId=saved.result_id||null;
        this.states.get(view.id).completedTest=saved.completed_test||null;
        this.states.get(view.id).jobId=saved.job_id||null;
        if(saved.exclusions){this.e.rhythmAnalyses.plotFilters[view.id]=copy(saved.exclusions);}
        this.e.syncMetricPickers();
        if(saved.result_id)this.loadResult(view.id,saved.result_id);
        if(saved.job_id)this.watch(view.id,saved.job_id);
      }
      this.e.rhythmAnalyses.refreshPlots();
    }
    removed(id){const state=this.states.get(id);if(state){clearTimeout(state.timer);state.request++;this.states.delete(id);}}
    bindPanel(view){
      if(!this.schema)return;const id=view.id;let state=this.states.get(id);if(state&&state.panel===view.panel)return;
      state={panel:view.panel,request:0,key:null,resultId:null};this.states.set(id,state);
      const set=(key,rows,preferred)=>{const el=this.element(key,id),value=preferred??el.value;el.replaceChildren(...rows.map(row=>new Option(row.label,row.key)));if(rows.some(r=>r.key===value))el.value=value;};
      set('view',this.schema.views,'samples');set('normalization',this.schema.normalizations,'none');set('detrend',this.schema.detrends.map(key=>({key,label:key.replaceAll('_',' ')})),'none');
      set('filter',Object.entries(this.schema.filters).map(([key,value])=>({key,label:value.label})),'none');set('test',this.schema.tests,'mannwhitney');set('correction',this.schema.corrections.map(key=>({key,label:key.replaceAll('_',' ')})),'benjamini_hochberg');
      this.element('run',id).onclick=()=>this.runTest(id);
      this.element('history',id).onchange=async()=>{const identifier=this.value('history',id);if(!identifier)return;state.resultId=identifier;await this.loadResult(id,identifier);state.key=null;this.draw(id);this.changed(false);};
      view.panel.querySelector('[data-comparison-export]').onclick=()=>this.export(id);
      for(const control of view.panel.querySelectorAll('select,input,textarea')){
        if(control.classList.contains('xp-plot-control')||control.closest('.xp-plot-exclusions'))continue;
        control.addEventListener('change',()=>{if(control===this.element('group',id)&&control.value==='rhythmicity')this.design.show_rhythms=true;this.draw(id);if(control===this.element('metric',id)||control===this.element('aggregation',id)||control===this.element('group',id))this.renderAssignments();this.changed(false);});
      }
      const canvas=this.e.$(id);canvas.onclick=event=>this.selectPoint(id,event);
      this.element('caption',id).textContent='';
      this.refreshGroups(id);this.refreshHistory(id);
    }
    renderAssignments(loadRhythms=true){
      if(!this.design)return;const e=this.e,$=id=>e.$('comparison-'+id),conditions=this.design.conditions;
      e.syncRecordingPicker();
      $('source-count').textContent=`${this.design.recordings.length} recording${this.design.recordings.length===1?'':'s'} loaded together`;
      $('condition-choices').replaceChildren(...['Unassigned',...conditions.map(row=>row.label)].map(label=>new Option(label,label)));
      const previous=$('cell-recording').value,currentRecording=this.design.recordings.find(row=>row.run===e.data.run&&row.stem===e.data.stem);
      $('cell-recording').replaceChildren(...this.design.recordings.map(row=>new Option(row.stem,row.id)));
      $('cell-recording').value=this.design.recordings.some(row=>row.id===previous)?previous:currentRecording?.id||this.design.recordings[0].id;
      $('factor-names').value=(this.design.factor_names||[...new Set(this.design.recordings.flatMap(r=>Object.keys(r.factors||{})))]).join(', ');
      $('show-samples').checked=!!this.design.show_samples;
      $('show-rhythms').checked=!!this.design.show_rhythms;
      const rhythmMetrics=this.design.show_rhythms?[...new Set(this.views().map(view=>this.options(view.id).metric))]:[];
      const factors=$('factor-names').value.split(',').map(v=>v.trim()).filter(Boolean),extra=this.design.show_samples?['subject','slice','pair']:[];
      $('recordings').querySelector('thead').innerHTML='<tr><th>Select</th><th>Include</th><th>Recording</th><th>Condition</th>'+extra.map(key=>`<th>${key==='pair'?'Pair':key==='slice'?'Slice':'Mouse / subject'}</th>`).join('')+factors.map(key=>`<th>${esc(key)}</th>`).join('')+rhythmMetrics.map(metric=>`<th>${esc(metric.replaceAll('_',' '))} rhythm run</th>`).join('')+'</tr>';
      $('recordings').querySelector('tbody').innerHTML=this.design.recordings.map(row=>{
        row.factors??={};const attrs=key=>`data-recording="${esc(row.id)}" data-field="${esc(key)}"`;
        const label=id=>conditions.find(condition=>condition.id===id)?.label||'Unassigned',overrides=Object.entries(row.cell_conditions||{}),byCondition=new Map();
        for(const [identity,condition] of overrides){if(!byCondition.has(condition))byCondition.set(condition,[]);byCondition.get(condition).push(identity);}
        const cellLabels=overrides.length?`<details><summary>${overrides.length} cells have their own condition</summary>${[...byCondition].map(([condition,ids])=>`<div>${esc(label(condition))}: ${esc(ids.join(', '))}</div>`).join('')}</details>`:'';
        return `<tr><td><input type="checkbox" ${attrs('selected')} ${this.selected.has(row.id)?'checked':''} title="Select this recording for bulk assignment"></td><td><input type="checkbox" ${attrs('include')} ${row.include?'checked':''} title="Include this recording in comparisons"></td><td title="${esc(row.run)}">${esc(row.stem)}</td><td>${esc(label(row.condition))}${cellLabels}</td>${extra.map(key=>`<td><input ${attrs(key)} value="${esc(row[key])}" aria-label="${esc(key)} for ${esc(row.stem)}" title="${key==='pair'?'Same pair identifier joins conditions':key==='subject'?'Recordings sharing this identifier can be aggregated':'Slice identifier'}"></td>`).join('')}${factors.map(key=>`<td><input ${attrs('factor:'+key)} value="${esc(row.factors[key])}" aria-label="${esc(key)} for ${esc(row.stem)}" title="${esc(key)} assignment"></td>`).join('')}${rhythmMetrics.map(metric=>`<td><select ${attrs('analysis:'+metric)} aria-label="${esc(metric)} completed rhythm run for ${esc(row.stem)}">${this.rhythmOptions(row,metric)}</select></td>`).join('')}</tr>`;
      }).join('');
      $('condition-list').innerHTML=conditions.map(c=>`<div class="controls"><label>Label <input data-condition="${esc(c.id)}" data-field="label" value="${esc(c.label)}" title="Condition display label"></label><label>Colour <input type="color" data-condition="${esc(c.id)}" data-field="colour" value="${esc(c.colour)}" title="Condition colour in every plot"></label><label><input type="checkbox" data-condition="${esc(c.id)}" data-field="control" ${c.control?'checked':''} title="Mark this condition as a control">Control</label><button type="button" data-condition="${esc(c.id)}" data-condition-order="-1" title="Move this condition earlier in plots">Earlier</button><button type="button" data-condition="${esc(c.id)}" data-condition-order="1" title="Move this condition later in plots">Later</button></div>`).join('');
      for(const view of this.views()){
        const select=this.element('group',view.id),value=select.value;select.replaceChildren(...[{key:'condition',label:'Condition'},{key:'recording',label:'Recording'},{key:'subject',label:'Subject'},{key:'all',label:'One population'},{key:'rhythmicity',label:'Completed rhythm analysis'},...factors.map(key=>({key:'factor:'+key,label:key}))].map(row=>new Option(row.label,row.key)));if([...select.options].some(o=>o.value===value))select.value=value;this.refreshGroups(view.id);
      }
      // Assignments are shared with the active recording's navigation and plots.
      const current=currentRecording,labelFor=cell=>conditions.find(c=>c.id===(current?.cell_conditions?.[String(cell.identity)]??current?.condition))?.label||'Unassigned';
      for(const cell of e.sourceData.cells){cell.group=labelFor(cell);cell.subject=current?.subject||'';}
      if(e.exact?.cell){e.exact.cell.group=labelFor(e.exact.cell);e.exact.cell.subject=current?.subject||'';}
      this.previewSelection();
      if(loadRhythms&&rhythmMetrics.length)this.loadRhythmChoices();
    }
    rhythmOptions(row,metric){
      const history=this.rhythmHistories.get(row.run+'\n'+row.stem),selected=row.analysis_ids?.[metric]||'';
      const options=(history?.analyses||[]).filter(item=>item.metrics.includes(metric));
      return `<option value="">${row.run===this.e.data.run&&row.stem===this.e.data.stem?'Use main selection':'Choose completed run'}</option>`+options.map(item=>`<option value="${esc(item.analysis_id)}" ${item.analysis_id===selected?'selected':''}>${esc(new Date(item.created_at).toLocaleString())} · ${esc(item.resolved_options?.fit_method||item.resolved_options?.method||'Rhythm analysis')}</option>`).join('');
    }
    async loadRhythmChoices(){
      const epoch=this.epoch;for(const row of this.design.recordings){const key=row.run+'\n'+row.stem;if(this.rhythmHistories.has(key)||this.rhythmPending.has(key))continue;this.rhythmPending.add(key);
        try{const history=await this.e.options.api('/api/exploration/rhythm-history?'+new URLSearchParams({run:row.run,stem:row.stem}));this.rhythmHistories.set(key,history);}catch(error){this.rhythmHistories.set(key,{analyses:[]});this.e.$('comparison-design-status').textContent=error.message;}finally{this.rhythmPending.delete(key);}if(epoch!==this.epoch)return;this.renderAssignments(false);
      }
    }
    refreshGroups(id,rows=null){
      if(!this.design)return;const select=this.element('groups',id),state=this.states.get(id),selected=new Set(state?.savedGroups||[...select.selectedOptions].map(o=>o.value)),group=this.value('group',id);
      rows??=group==='condition'?this.design.conditions.map(c=>({id:c.id,label:c.label})):group==='all'?[{id:'All cells',label:'All cells'}]:[];
      select.replaceChildren(...rows.map(row=>{const option=new Option(row.label,row.id);option.selected=selected.has(row.id);return option;}));
      if(rows.length&&state?.savedGroups){state.savedGroups=null;state.key=null;queueMicrotask(()=>this.draw(id));}
    }
    controls(id){
      const keys=['metric','aggregation','view','group','unit','time','summary','sample-summary','start','end','groups','normalization','detrend','filter','filter-points','band','bins','bandwidth','normalization-config','detrend-config','test','test-unit','correction','alpha'];
      return Object.fromEntries(keys.map(key=>{const el=this.element(key,id);return [key,el.multiple?[...el.selectedOptions].map(o=>o.value):el.value];}));
    }
    options(id){
      const v=this.controls(id),number=key=>v[key]===''?null:Number(v[key]);
      return {view:v.view,metric:window.PyMicrogliaExplorationCore.metricKey(v.metric,v.aggregation,this.e.data),group_by:v.group,
        unit:v.view==='cells'?'cell':v.unit,summary:v.summary,aggregate:v['sample-summary'],start:number('start'),end:number('end'),
        frame:v.time==='frame'?this.e.state.frame:null,groups:v.groups,normalization:v.normalization,detrend:v.detrend,
        trace_filter:v.filter==='none'?{method:'none'}:{method:v.filter,window_points:Number(v['filter-points'])},band:v.band,bins:Number(v.bins),bandwidth:number('bandwidth'),
        normalization_config:JSON.parse(v['normalization-config']||'{}'),detrend_config:JSON.parse(v['detrend-config']||'{}')};
    }
    test(id){return {test:this.value('test',id),unit:this.value('test-unit',id),aggregate:this.value('sample-summary',id),correction:this.value('correction',id),alpha:Number(this.value('alpha',id)),groups:[...this.element('groups',id).selectedOptions].map(o=>o.value)};}
    exclusions(id,metric){
      const rhythm=this.e.rhythmAnalyses,override=rhythm.plotFilters[id]?.[metric];
      if(override?.enabled)return override.filter?.applied?[copy(override.filter.applied)]:[];
      return Object.entries(rhythm.metricFilters).filter(([key,state])=>key===metric||state.scope==='all').map(([,state])=>state.filter?.applied).filter(Boolean).map(copy);
    }
    request(id){
      const options=this.options(id),design=copy(this.design),e=this.e,current=design.recordings.find(r=>r.run===e.data.run&&r.stem===e.data.stem);
      if(current)current.analysis_ids={...e.rhythmAnalyses.filterAnalysisIds(),...current.analysis_ids};
      const canvas=e.$(id),width=Math.max(400,Math.round(canvas.getBoundingClientRect().width));
      return {run:this.run,design,options,exclusions:this.exclusions(id,options.metric),appearance:{...e.style(id),colourOverride:e.styles[id]?.accent||null},width,result_id:this.states.get(id)?.resultId||null};
    }
    save(){
      if(!this.design)return null;const design=copy(this.design);design.plots={};
      for(const view of this.views()){const state=this.states.get(view.id);if(!state)continue;design.plots[view.id]={controls:this.controls(view.id),appearance:this.e.styles[view.id]||{},result_id:state.resultId,completed_test:state.completedTest,job_id:state.jobId,exclusions:this.e.rhythmAnalyses.plotFilters[view.id]||{}};}
      const group=this.e.comparisons.groups.get('distribution');design.layout={columns:Number(group.columns.value),plots:this.e.comparisons.copies('distribution').map(v=>v.id)};return design;
    }
    changed(redraw=true){
      if(!this.design)return;this.design=this.save();this.e.saveSession();if(redraw)this.drawAll();
    }
    checkpoint(){
      if(!this.design||this.loading||this.e.restoring)return;const design=this.save(),snapshot=JSON.stringify(design);if(snapshot===this.lastSnapshot)return;this.lastSnapshot=snapshot;
      this.e.$('comparison-design-status').textContent='Unsaved changes';
      try{localStorage.setItem('pm-comparison-draft:'+this.run,JSON.stringify({pending:true,design}));}catch{/* Disk persistence remains available. */}
      clearTimeout(this.saveTimer);this.saveTimer=setTimeout(()=>{this.saveTimer=null;this.persist(this.epoch,this.run,design,snapshot);},400);
    }
    persist(epoch,run,design,snapshot){
      this.e.$('comparison-design-status').textContent='Saving…';
      this.saveQueue=this.saveQueue.catch(()=>{}).then(async()=>{try{await this.e.options.api('/api/exploration/comparison-workspace',{method:'PUT',body:JSON.stringify({run,design})});if(run===this.run&&snapshot===this.lastSnapshot){this.e.$('comparison-design-status').textContent='Saved';try{localStorage.setItem('pm-comparison-draft:'+run,JSON.stringify({pending:false,design}));}catch{}}}catch(error){if(run===this.run&&snapshot===this.lastSnapshot)this.e.$('comparison-design-status').textContent=error.message;}});return this.saveQueue;
    }
    drawAll(){for(const view of this.views())this.draw(view.id);}
    async draw(id){
      if(!this.design||!this.e.data||this.e.mediaBusy&&this.e.data.identities.length)return;const view=this.e.comparisons.views.get(id);if(!view)return;this.bindPanel(view);const state=this.states.get(id);
      const unit=this.element('unit',id),cells=this.value('view',id)==='cells';
      const kind=this.value('view',id),show=(key,visible)=>{this.element(key,id).closest('label').hidden=!visible;};
      show('band',kind==='time_course');show('bins',kind==='histogram');show('bandwidth',kind==='violin');
      show('filter-points',this.value('filter',id)!=='none');show('normalization-config',this.value('normalization',id)!=='none');show('detrend-config',this.value('detrend',id)!=='none');
      show('summary',kind!=='time_course');
      if(cells&&!unit.disabled){state.lastSampleUnit=unit.value;unit.value='cell';unit.disabled=true;}
      else if(!cells&&unit.disabled){unit.disabled=false;unit.value=state.lastSampleUnit||'movie';}
      let payload;try{payload=this.request(id);}catch(error){this.element('caption',id).textContent=error.message;return;}
      // Workspace plot snapshots do not affect scientific inputs or rendering.
      payload.design.plots={};payload.design.layout={};const key=JSON.stringify(payload);
      if(state.key===key){if(state.preview)this.paint(id,state.preview);this.resultStatus(id);return;}
      state.key=key;state.preview=null;const token=++state.request,epoch=this.epoch;
      this.element('caption',id).textContent='Drawing native comparison…';
      this.element('aggregation-label',id).hidden=this.value('metric',id)!=='light';
      try{
        const preview=await this.e.options.api('/api/exploration/comparison-preview',{method:'POST',body:JSON.stringify(payload)});
        if(epoch!==this.epoch||token!==state.request||!view.panel.isConnected)return;
        state.preview=preview;state.image=null;state.displayedRequest=payload;await this.paint(id,preview);
        const c=preview.counts;this.element('caption',id).textContent=`${c.finite_cells}/${c.cells} cells · ${c.recordings} recordings · ${c.samples} sample points`;
        this.element('diagnostics',id).textContent=[...preview.warnings,`${c.observations} full saved observations; ${c.excluded_cells} excluded cells.`,...preview.inputs.map(row=>`${row.stem}: ${row.path}`)].join('\n');
        if(this.value('group',id)!=='condition')this.refreshGroups(id,preview.groups);
        this.resultStatus(id);
      }catch(error){if(epoch===this.epoch&&token===state.request){state.key=null;this.element('caption',id).textContent=error.message;this.e.blank(this.e.$(id),error.message);}}
    }
    async paint(id,preview){
      const state=this.states.get(id);if(!state)return;
      state.lastCounts=preview.counts;
      if(!state.image){const image=new Image(),url=URL.createObjectURL(new Blob([preview.image],{type:'image/svg+xml'}));try{await new Promise((resolve,reject)=>{image.onload=resolve;image.onerror=()=>reject(Error('Native figure image could not be displayed'));image.src=url;});state.image=image;}finally{URL.revokeObjectURL(url);}}
      if(state.preview!==preview)return;const canvas=this.e.$(id);canvas.width=preview.width;canvas.height=preview.height;canvas.dataset.logicalWidth=preview.width;canvas.dataset.logicalHeight=preview.height;canvas.style.height=preview.height+'px';canvas.getContext('2d').drawImage(state.image,0,0,preview.width,preview.height);this.e.hits[id]=preview.hits;
      const selected=preview.hits.find(hit=>hit.run===this.e.data.run&&hit.stem===this.e.data.stem&&hit.identity===this.e.state.identity);
      if(selected){const context=canvas.getContext('2d');context.beginPath();context.arc(selected.x,selected.y,6,0,Math.PI*2);context.strokeStyle=this.e.style(id).accent;context.lineWidth=2;context.stroke();}
      const editor=this.e.rhythmAnalyses.plotEditors.get(id),metric=this.options(id).metric;
      if(editor)editor.count.textContent=`${this.e.rhythmAnalyses.plotFilters[id]?.[metric]?.enabled?'Plot override':'Main'}: ${preview.counts.cells}/${preview.counts.total_cells}`;
    }
    async selectPoint(id,event){
      const point=this.e.point(id,event),hit=(this.e.hits[id]||[]).reduce((best,row)=>{const distance=Math.hypot(row.x-point.x,row.y-point.y);return distance<best.distance?{...row,distance}:best;},{distance:20});if(hit.identity==null)return;
      await this.e.switchRecording(hit,{identity:hit.identity});
    }
    async runTest(id){
      const state=this.states.get(id);this.element('run',id).disabled=true;try{const payload=this.request(id);payload.test=this.test(id);const job=await this.e.options.api('/api/exploration/comparison-jobs',{method:'POST',body:JSON.stringify(payload)});state.jobId=job.id;this.changed(false);this.watch(id,job.id);}catch(error){this.element('result-status',id).textContent=error.message;this.element('run',id).disabled=false;}
    }
    async watch(id,jobId){
      const state=this.states.get(id),epoch=this.epoch;if(!state)return;
      try{const job=await this.e.options.api('/api/jobs/'+jobId);if(epoch!==this.epoch||state.jobId!==jobId)return;
        if(['queued','running'].includes(job.state)){this.element('result-status',id).textContent=job.state==='queued'?'Comparison queued':'Running native comparison…';state.timer=setTimeout(()=>this.watch(id,jobId),700);return;}
        state.jobId=null;this.element('run',id).disabled=false;
        if(job.state==='failed')throw Error(job.error);state.resultId=job.result.id;state.result=job.result;state.completedTest=job.result.request.test;state.key=null;this.showResults(id);this.refreshHistory(id);this.changed(false);this.draw(id);
      }catch(error){if(epoch===this.epoch){state.jobId=null;this.element('run',id).disabled=false;this.element('result-status',id).textContent=error.message;this.changed(false);}}
    }
    async loadResult(id,identifier){try{const done=await this.e.options.api('/api/exploration/comparison-result?'+new URLSearchParams({run:this.run,result_id:identifier}));const state=this.states.get(id);if(!state||state.resultId!==identifier)return;state.result=done;state.completedTest=done.request.test;this.showResults(id);this.resultStatus(id);}catch(error){this.element('result-status',id).textContent=error.message;}}
    async refreshHistory(id){const epoch=this.epoch;try{const result=await this.e.options.api('/api/exploration/comparison-history?'+new URLSearchParams({run:this.run}));if(epoch!==this.epoch||!this.e.comparisons.views.has(id))return;const select=this.element('history',id),current=this.states.get(id)?.resultId;select.replaceChildren(new Option('Choose a completed comparison',''),...result.comparisons.map(row=>new Option(`${new Date(row.created_at).toLocaleString()} · ${row.metric.replaceAll('_',' ')} · ${this.schema.tests.find(test=>test.key===row.test.test)?.label||row.test.test}`,row.id)));select.value=current||'';}catch(error){this.element('result-status',id).textContent=error.message;}}
    resultStatus(id){const state=this.states.get(id);if(!state?.result||state.jobId)return;const stale=state.preview?.stale||JSON.stringify(state.completedTest)!==JSON.stringify(this.test(id));this.element('result-status',id).textContent=stale?'Settings changed · showing the previous completed comparison':'Showing completed comparison';}
    showResults(id){const state=this.states.get(id),rows=state.result.statistics,labels=new Map(this.design.conditions.map(c=>[c.id,c.label])),counts=value=>String(value||'').split('+').map(part=>{const [key,n]=part.split('=');return `${labels.get(key)||key}: ${n}`;}).join(' · '),format=value=>typeof value==='number'?Number(value.toPrecision(5)).toString():value??'';
      this.element('test-results',id).innerHTML='<table><thead><tr><th>Test</th><th>Unit</th><th>Group counts</th><th>p</th><th>Corrected p</th><th>Effect</th></tr></thead><tbody>'+rows.map(row=>`<tr><td>${esc(this.schema.tests.find(test=>test.key===row.test)?.label||row.test)}</td><td>${esc({movie:'Recording',subject:'Subject',cell:'Cell'}[row.unit]||row.unit)}</td><td>${esc(counts(row.n_per_group))}</td><td>${esc(format(row.p_value) || 'Not tested')}</td><td>${esc(format(row.p_corrected))}</td><td>${esc(format(row.effect))}</td></tr>${row.note?`<tr><td colspan="6">${esc(row.note)}</td></tr>`:''}`).join('')+'</tbody></table>';}
    async export(id){
      const state=this.states.get(id),format=this.value('export-format',id);if(!state?.displayedRequest||!state.preview)return;const preview=state.preview,payload={...state.displayedRequest,expected_input_key:preview.input_key};
      try{if(format==='json'){const completed=payload.result_id?await this.e.options.api('/api/exploration/comparison-result?'+new URLSearchParams({run:payload.run,result_id:payload.result_id})):null;this.download(JSON.stringify({schema_version:1,request:payload,preview:{...preview,image:undefined},result:completed},null,2),'comparison.json','application/json');return;}
        if(format==='py'){const response=await fetch('/api/exploration/comparison-script',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});if(!response.ok)throw Error((await response.json()).detail);this.download(await response.blob(),'comparison.py','text/x-python');return;}
        const response=await fetch('/api/exploration/comparison-export',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...payload,format})});if(!response.ok)throw Error((await response.json()).detail);this.download(await response.blob(),'comparison.'+format,format==='svg'?'image/svg+xml':'image/png');
      }catch(error){this.element('caption',id).textContent=error.message;}
    }
    download(data,name,type){const url=URL.createObjectURL(data instanceof Blob?data:new Blob([data],{type})),link=document.createElement('a');link.href=url;link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
  }
  window.PyMicrogliaGroupComparison=GroupComparison;
})();
