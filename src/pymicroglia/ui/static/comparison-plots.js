/* Repeated linked plots: one builder, independent controls, shared selection. */
(() => {
  "use strict";
  const FAMILIES = [
    {key:"barcode", panel:"barcode-panel", title:"Barcodes", metric:"barcode-metric", draw:"drawBarcode"},
    {key:"tissue-map", panel:"map-panel", title:"Tissue maps", metric:"tectonics-metric", draw:"drawMap"},
    {key:"distribution", panel:"distribution-panel", title:"Cell and group comparisons", metric:"distribution-metric", draw:"drawDistribution"}
  ];
  class ComparisonPlots {
    constructor(explorer){
      this.explorer=explorer;this.views=new Map();this.groups=new Map();
      for(const spec of FAMILIES){
        const panel=explorer.$(spec.panel),template=panel.cloneNode(true),wrapper=document.createElement("div"),bar=document.createElement("div"),grid=document.createElement("div");
        wrapper.className="xp-comparison-group";wrapper.id=`xp-${spec.key}-comparisons`;
        bar.className="xp-comparison-toolbar";grid.className="xp-comparison-grid";
        const add=document.createElement("button");add.type="button";add.textContent="Add plot";add.title=`Add another ${spec.title.toLowerCase()} plot with independent settings`;
        const label=document.createElement("label"),columns=document.createElement("select");label.textContent="Columns ";columns.id=`xp-${spec.key}-columns`;columns.setAttribute("aria-label",`${spec.title} columns`);
        for(const n of [1,2,3,4])columns.add(new Option(String(n),String(n)));columns.value="2";columns.title="Number of plots in each row; narrow windows stack the plots";label.append(columns);bar.append(add,label);
        panel.before(wrapper);wrapper.append(bar,grid);grid.append(panel);
        const group={spec,template,wrapper,grid,columns,add};this.groups.set(spec.key,group);
        this.views.set(spec.key,{id:spec.key,base:spec.key,panel,mapping:new Map(),selectionKey:null});
        add.onclick=()=>{if(!explorer.data)return;this.add(spec.key);this.layout(group);explorer.draw();explorer.saveSession();if(spec.key==="distribution")explorer.groupComparison?.changed(false);};
        columns.onchange=()=>{this.layout(group);explorer.draw();explorer.saveSession();if(spec.key==="distribution")explorer.groupComparison?.changed(false);};
        this.layout(group);
      }
    }
    base(id){return this.views.get(id)?.base||id;}
    element(id,view){return this.explorer.$(this.views.get(view)?.mapping.get(id)||id);}
    value(id,view){return this.element(id,view).value;}
    copies(base){return [...this.views.values()].filter(view=>view.base===base&&view.id!==base);}
    save(){return {version:1,groups:Object.fromEntries([...this.groups].map(([base,group])=>[base,{columns:+group.columns.value,plots:this.copies(base).map(view=>view.id)}]))};}
    layout(group){
      const count=1+this.copies(group.spec.key).length;group.grid.style.setProperty("--xp-comparison-columns",Math.min(count,+group.columns.value));
      group.wrapper.classList.toggle("has-comparisons",count>1);group.add.disabled=!this.explorer.data||count>=8;
    }
    add(base,id){
      const e=this.explorer,group=this.groups.get(base);if(!group||this.copies(base).length>=7)return;
      if(!id){let number=2;while(this.views.has(`${base}-${number}`))number++;id=`${base}-${number}`;}
      if(!new RegExp(`^${base}-[2-8]$`).test(id)||this.views.has(id))return;
      const panel=group.template.cloneNode(true),mapping=new Map();
      for(const element of [panel,...panel.querySelectorAll("[id]")]){
        const old=element.id.slice(3),next=old===base?id:`${old}-${id}`;mapping.set(old,next);element.id=`xp-${next}`;
      }
      const view={id,base,panel,mapping,selectionKey:null};this.views.set(id,view);group.grid.append(panel);
      for(const el of panel.querySelectorAll("select,input")){
        const original=[...mapping].find(([,next])=>`xp-${next}`===el.id)?.[0],source=e.$(original);
        if(el.tagName==="SELECT")el.innerHTML=source.innerHTML;
        if(el.type==="checkbox")el.checked=source.checked;else el.value=source.value;
        el.title=source.title;
        el.addEventListener("change",()=>{this.drawView(view);e.saveSession();});
      }
      const canvas=this.element(base,id);canvas.dataset.comparisonBase=base;
      if(e.styles[base]&&!e.styles[id])e.styles[id]={...e.styles[base]};
      e.exports([panel]);e.plotEditors([panel]);
      const remove=document.createElement("button");remove.type="button";remove.textContent="Remove";remove.title="Remove this comparison plot";
      remove.onclick=()=>{this.remove(id);this.layout(group);e.draw();e.saveSession();if(base==="distribution")e.groupComparison?.changed(false);};e.plotHeader(panel).actions.append(remove);
      e.updateMetricPicker(mapping.get(group.spec.metric));
      canvas.onclick=event=>{
        if(!e.data)return;const point=e.point(id,event),hit=e.hits[id];if(!hit)return;
        if(base==="barcode"){
          const fraction=Math.max(0,Math.min(1,(point.x-hit.left)/(hit.right-hit.left))),row=Math.floor((point.y-hit.top)/hit.rowHeight);
          e.select({frame:e.nearestFrame(fraction),...(hit.cells[row]?{identity:hit.cells[row].identity}:{})});
        }else{
          const closest=hit.reduce((best,item)=>{const distance=Math.hypot(point.x-item.x,point.y-item.y);return distance<best.distance?{...item,distance}:best;},{distance:25});
          if(closest.identity!=null)e.select({identity:closest.identity});
        }
      };
      canvas.onkeydown=event=>{if(e.data&&["ArrowLeft","ArrowRight"].includes(event.key)){e.select({frame:e.state.frame+(event.key==="ArrowLeft"?-1:1)});event.preventDefault();}};
      e.groupComparison?.bindPanel(view);e.rhythmAnalyses?.refreshPlots();this.layout(group);return view;
    }
    remove(id){
      const view=this.views.get(id);if(!view||id===view.base)return;
      this.explorer.groupComparison?.removed(id);
      for(const entry of this.explorer.metricPickers?.values()||[])if(view.panel.contains(entry.select)){entry.details.open=false;entry.menu.hidePopover?.();this.explorer.metricPickers.delete(entry.select.id.slice(3));}
      view.panel.remove();delete this.explorer.hits[id];this.views.delete(id);
    }
    reset(saved){
      for(const view of [...this.views.values()])if(view.id!==view.base)this.remove(view.id);
      for(const [base,group] of this.groups){
        const state=saved?.version===1?saved.groups?.[base]:null;group.columns.value=String([1,2,3,4].includes(state?.columns)?state.columns:2);
        for(const id of state?.plots||[])this.add(base,id);
        this.layout(group);
      }
    }
    refresh(){
      const e=this.explorer;
      for(const view of this.views.values())if(view.id!==view.base){
        if(view.base==="distribution")continue;
        for(const [original,next] of view.mapping){
          const target=e.$(next),source=e.$(original);if(target?.tagName!=="SELECT"||!source)continue;
          const selected=target.value;target.innerHTML=source.innerHTML;
          if([...target.options].some(option=>option.value===selected&&!option.disabled))target.value=selected;
          if(e.metricPickers?.has(next))e.updateMetricPicker(next);
        }
      }
    }
    drawView(view){
      if(!this.explorer.data)return;
      if(["barcode","distribution"].includes(view.base))this.element(view.base+"-aggregation-label",view.id).hidden=this.value(view.base+"-metric",view.id)!=="light";
      this.explorer[this.groups.get(view.base).spec.draw](view.id);
    }
    draw(base){for(const view of this.copies(base))this.drawView(view);}
  }
  window.PyMicrogliaComparisonPlots=ComparisonPlots;
})();
