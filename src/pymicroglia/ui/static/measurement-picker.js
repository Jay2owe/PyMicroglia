"use strict";
(() => {
  // One component for checklists and single-metric menus. Browsing tabs never
  // dispatches a measurement change or starts an analysis.
  class MeasurementPicker {
    static popup(details,menu){
      if(menu.dataset.measurementPopup)return;menu.dataset.measurementPopup="true";menu.classList.add("xp-measurement-popup");menu.setAttribute("popover","auto");
      const place=()=>{const anchor=details.querySelector("summary").getBoundingClientRect(),bounds=menu.getBoundingClientRect();menu.style.left=`${Math.max(8,Math.min(anchor.left,innerWidth-bounds.width-8))}px`;menu.style.top=`${Math.max(8,Math.min(anchor.bottom+3,innerHeight-bounds.height-8))}px`;};
      details.addEventListener("toggle",()=>{if(details.open){menu.showPopover?.();place();}else if(menu.matches(":popover-open"))menu.hidePopover();});
      menu.addEventListener("toggle",event=>{if(event.newState==="closed")details.open=false;});
      menu.addEventListener("keydown",event=>{if(event.key==="Escape"){menu.hidePopover?.();details.open=false;details.querySelector("summary").focus();}});
      window.addEventListener("resize",()=>{if(details.open)place();});
    }
    constructor(host,{catalogue,selected=[],mode="multiple",choices=null,active=null,onChange=null,onTab=null}) {
      this.host=host;this.catalogue=catalogue;this.mode=mode;this.onChange=onChange;this.onTab=onTab;
      this.prefix=host.id;this.inputs=new Map();this.tabs=new Map();this.panels=new Map();this.selected=new Set(selected);
      const allowed=choices?new Map(choices.map(row=>[row.key,row])):null;
      this.rows=catalogue.metrics.map(row=>{
        const choice=allowed?.get(row.key);
        return {...row,...choice,available:row.available!==false&&(!allowed||!!choice),
          status:row.available===false?row.status:allowed&&!choice?"Not available in this view":row.status};
      });
      if(allowed)for(const row of choices)if(!this.rows.some(item=>item.key===row.key))this.rows.unshift({...row,available:true,group:row.group||"signal",status:"Available"});
      this.host.replaceChildren();this.host.classList.add("xp-measurement-browser");
      if(host.tagName==="FIELDSET"){const legend=document.createElement("legend");legend.textContent="Measurements to test";host.append(legend);}
      this.strip=document.createElement("div");this.strip.className="xp-measurement-selected";this.strip.setAttribute("aria-label","Selected measurements");host.append(this.strip);
      const tablist=document.createElement("div");tablist.className="xp-measurement-tabs";tablist.setAttribute("role","tablist");tablist.setAttribute("aria-label","Measurement groups");host.append(tablist);
      for(const group of catalogue.groups){
        const rows=this.rows.filter(row=>row.group===group.key),tab=document.createElement("button"),panel=document.createElement("div");
        tab.type="button";tab.id=`${this.prefix}-tab-${group.key}`;tab.dataset.measurementGroup=group.key;tab.setAttribute("role","tab");tab.title=`Browse ${group.label.toLowerCase()} measurements; selections in other groups are retained`;
        panel.id=`${this.prefix}-panel-${group.key}`;panel.className="xp-measurement-group";panel.setAttribute("role","tabpanel");panel.setAttribute("aria-labelledby",tab.id);panel.hidden=true;
        tab.setAttribute("aria-controls",panel.id);tab.onclick=()=>this.activate(group.key,true);
        tab.onkeydown=event=>{const keys=[...this.tabs.keys()],index=keys.indexOf(group.key);let target;
          if(event.key==="ArrowRight")target=keys[(index+1)%keys.length];if(event.key==="ArrowLeft")target=keys[(index+keys.length-1)%keys.length];
          if(event.key==="Home")target=keys[0];if(event.key==="End")target=keys.at(-1);
          if(target){event.preventDefault();this.activate(target,true);this.tabs.get(target).focus();}
        };
        this.tabs.set(group.key,tab);this.panels.set(group.key,panel);tablist.append(tab);host.append(panel);
        if(mode==="multiple"){
          const actions=document.createElement("div");actions.className="xp-measurement-actions";
          for(const [label,checked] of [["Select group",true],["Clear group",false]]){const button=document.createElement("button");button.type="button";button.textContent=label;button.title=`${checked?"Select all available":"Clear selected"} measurements in ${group.label.toLowerCase()}`;
            button.onclick=()=>{for(const row of rows){const input=this.inputs.get(row.key);if(!input.disabled)input.checked=checked;}this.refresh();this.host.dispatchEvent(new Event("change",{bubbles:true}));};actions.append(button);}
          panel.append(actions);
        }
        const list=document.createElement("div");list.className="xp-measurement-options";panel.append(list);
        for(const row of rows){
          const label=document.createElement("label"),input=document.createElement("input"),text=document.createElement("span");
          input.type=mode==="single"?"radio":"checkbox";input.name=this.prefix+"-measurements";input.value=row.key;input.dataset.available=String(row.available);input.disabled=!row.available;input.checked=row.available&&this.selected.has(row.key);
          label.className="xp-measurement-option";label.classList.toggle("is-unavailable",!row.available);label.title=row.help||"Choose this saved cell measurement";
          text.className="xp-measurement-name";text.textContent=row.label;label.append(input,text);
          if(!row.available){const status=document.createElement("small");status.className="xp-measurement-availability";status.textContent=row.status||"Not measured yet";label.append(status);}
          input.onchange=()=>{this.refresh();this.onChange?.(this.values());};this.inputs.set(row.key,input);list.append(label);
        }
        if(!rows.length){const empty=document.createElement("span");empty.className="xp-measurement-empty";empty.textContent="No saved measurements in this group";list.append(empty);}
      }
      this.refresh();this.activate(active&&this.tabs.has(active)?active:this.rows.find(row=>this.selected.has(row.key))?.group||catalogue.groups[0].key);
    }
    values(){return [...this.inputs.values()].filter(input=>input.checked&&input.dataset.available!=="false").map(input=>input.value);}
    setSelected(values){const selected=new Set(values);for(const input of this.inputs.values())input.checked=input.dataset.available!=="false"&&selected.has(input.value);this.refresh();}
    refresh(){
      this.strip.replaceChildren();const heading=document.createElement("span");heading.textContent=this.mode==="single"?"Selected:":"Selected measurements:";this.strip.append(heading);
      const chosen=this.values();if(!chosen.length){const empty=document.createElement("span");empty.textContent="None";this.strip.append(empty);}
      for(const key of chosen){const row=this.rows.find(row=>row.key===key),chip=document.createElement("span");chip.className="xp-measurement-chip";chip.textContent=row.label;chip.title=row.help||row.label;this.strip.append(chip);}
      for(const group of this.catalogue.groups){const available=this.rows.filter(row=>row.group===group.key&&row.available),count=available.filter(row=>chosen.includes(row.key)).length;this.tabs.get(group.key).textContent=group.label+(count?` (${count})`:"");}
    }
    activate(key,notify=false){this.active=key;for(const [group,tab] of this.tabs){const selected=group===key;tab.setAttribute("aria-selected",String(selected));tab.tabIndex=selected?0:-1;this.panels.get(group).hidden=!selected;}if(notify)this.onTab?.(key);}
  }
  window.PyMicrogliaMeasurementPicker=MeasurementPicker;
})();
