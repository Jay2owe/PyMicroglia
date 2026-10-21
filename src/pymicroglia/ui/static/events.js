"use strict";
(() => {
  const $=id=>document.getElementById(`ev-${id}`),html=v=>String(v??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"})[c]);
  class EventBrowser {
    constructor(options){
      this.options=options;this.data=null;this.selected=null;this.epoch=0;this.frameEpoch=0;this.cache=new Map();this.previews=new Map();this.galleryEpoch=0;this.view="grid";
      let saved={};try{saved=JSON.parse(localStorage.getItem("pm-events-v1")||"{}");}catch{}
      this.settings={event_hours:6,fps:6,crop_px:96,cell_lut:"red",...([1,2].includes(saved.version)?saved.settings:{})};
      this.saved=saved.version===1?{...saved,version:2,filter:"all",status:"all",search:"",view:"grid"}:saved;
      for(const key of Object.keys(this.settings))$(key).value=this.settings[key];
      for(const key of Object.keys(this.settings))$(key).onchange=()=>{this.pause();const value=$(key).value;this.settings[key]=key==="cell_lut"?value:Number(value);this.cache.clear();this.save();if(["crop_px","cell_lut"].includes(key)){this.previews.clear();this.renderList();}if(this.selected)this.choose(this.selected.id);};
      for(const id of ["filter","status","search"])$(id).oninput=()=>{this.showGrid();this.renderList();this.save();};
      $("back").onclick=()=>{this.showGrid();this.save();};
      $("open").onclick=()=>{if(this.selected)this.openEvent(this.selected.id);};
      $("refresh").onclick=()=>this.load($("run").value.trim(),$("stem").value);
      $("run").onchange=()=>this.load($("run").value.trim());
      $("stem").onchange=()=>this.load(this.data?.run||$("run").value,$("stem").value);
      $("slider").oninput=()=>{this.pause();this.showFrame(Number($("slider").value));};
      $("play").onclick=()=>this.timer?this.pause():this.play();
      $("previous").onclick=()=>{this.pause();this.showFrame(Math.max(this.first,this.frame-1));};
      $("next").onclick=()=>{this.pause();this.showFrame(Math.min(this.last,this.frame+1));};
      $("preview-mode").onclick=()=>this.media("preview");$("video-mode").onclick=()=>this.media("video");
      $("clip").onchange=()=>this.media("video");$("generate").onclick=()=>this.generate();
      $("inspect").onclick=()=>this.options.onInspect?.({run:this.data.run,stem:this.data.stem,identity:this.selected.identity,frame:this.frame});
      $("theme").onchange=()=>this.options.onTheme?.($("theme").value);
      $("video").onerror=()=>{$("message").textContent="This saved video could not be played. Use Preview or download the file.";};
      $("video").ontimeupdate=()=>{const clip=this.clips?.[Number($("clip").value)||0];if(!clip?.frames?.length)return;this.frame=clip.frames[Math.min(clip.frames.length-1,Math.floor($("video").currentTime*clip.options.fps))];const clock=this.data.frames[this.frame];$("frame-note").textContent=`Source frame ${clock.source_frame_index+1} · ${Number(clock.hours?.toFixed(2))} h · saved video`;};
      this.visible=false;
    }
    async open(run,stem){this.visible=true;$("theme").value=localStorage.getItem("pm-extras-theme")||"auto";if(run&&(this.data?.run!==run||stem&&this.data?.stem!==stem))await this.load(run,stem);else if(this.data)this.render();else $("empty").textContent="Choose a saved measurement run to see its recorded events.";}
    close(){this.visible=false;this.pause();$("video").pause();}
    save(){this.saved={version:2,run:this.data?.run,stem:this.data?.stem,event:this.selected?.id,frame:this.frame,settings:{...this.settings},filter:$("filter").value,status:$("status").value,search:$("search").value,media:this.mode,view:this.view};localStorage.setItem("pm-events-v1",JSON.stringify(this.saved));}
    async load(run,stem){
      this.pause();this.selected=null;this.cache.clear();this.previews.clear();this.previewAbort?.abort();++this.galleryEpoch;const epoch=++this.epoch;++this.frameEpoch;
      $("run").value=run;this.showGrid();$("tabs").replaceChildren();$("grid-progress").textContent="Loading recorded events…";$("empty").hidden=false;$("empty").textContent="Loading recorded events…";$("list").replaceChildren();$("image").hidden=true;$("video").hidden=true;$("generate").disabled=true;$("facts").hidden=true;
      if(!run){this.data=null;$("grid-progress").textContent="";$("empty").textContent="Choose a saved measurement run.";return;}
      try{const data=await this.options.api(`/api/events?${new URLSearchParams({run,...(stem?{stem}:{})})}`);if(epoch!==this.epoch)return;
        this.data=data;$("run").value=data.run;$("stem").innerHTML=data.recordings.map(s=>`<option value="${html(s)}">${html(s)}</option>`).join("");$("stem").value=data.stem;
        $("filter").innerHTML=data.tabs.map(tab=>`<option value="${tab.key}">${html(tab.title)}</option>`).join("");
        if(this.saved.run===data.run&&this.saved.stem===data.stem){for(const id of ["filter","status","search"])if(this.saved[id]!=null)$(id).value=this.saved[id];}
        if(!$("filter").value)$("filter").value="all";if(!$("status").value)$("status").value="all";
        $("source-note").textContent=data.lifecycle_review?"Lifecycle review uses original accepted Motion masks. Births and deaths remain candidates.":data.lifecycle_saved?"Saved classifications retain their recorded status.":"No saved birth/death classifications. Tracking assessments remain separate.";
        const candidates=this.filtered(),selected=data.rows.find(r=>r.id===this.saved.event&&candidates.includes(r))||candidates[0];
        const restored=this.saved.run===data.run&&this.saved.stem===data.stem&&this.saved.view==="detail";
        this.view=restored?"detail":"grid";$("gallery").hidden=restored;$("detail").hidden=!restored;
        this.selected=selected||null;this.renderList();if(selected)await this.choose(selected.id,this.saved.run===data.run&&this.saved.event===selected.id?this.saved.frame:undefined);else this.empty();
        this.options.onRecording?.(data);
      }catch(error){if(epoch!==this.epoch)return;this.data=null;this.empty(error.message);$("grid-progress").textContent="";$("empty").hidden=false;$("empty").textContent=error.message;}
    }
    matches(row,tab){return !tab||tab.key==="all"|| (tab.event?row.event===tab.event:row.group===tab.group);}
    filtered(type=$("filter").value){const tab=this.data?.tabs.find(t=>t.key===type),status=$("status").value,q=$("search").value.toLowerCase().trim();const cohort=this.options.cohort?.(this.data);return (this.data?.rows||[]).filter(r=>(!cohort||cohort.has(r.identity))&&this.matches(r,tab)&&(status==="all"||status==="not-censored"&&r.status!=="censored"||status===r.status)&&(!q||`${r.identity} ${r.label} ${r.status} ${r.partners.join(" ")}`.toLowerCase().includes(q)));}
    renderTabs(){
      $("tabs").replaceChildren();
      for(const tab of this.data?.tabs||[]){
        const button=document.createElement("button");button.type="button";button.id=`ev-tab-${tab.key}`;button.dataset.type=tab.key;
        button.setAttribute("role","tab");button.setAttribute("aria-controls","ev-list");const active=$("filter").value===tab.key;
        button.setAttribute("aria-selected",String(active));button.tabIndex=active?0:-1;button.textContent=`${tab.title} (${this.filtered(tab.key).length})`;
        button.onclick=()=>{$("filter").value=tab.key;this.showGrid();this.renderList();this.save();$("tabs").querySelector(`[data-type="${tab.key}"]`)?.focus();};
        button.onkeydown=event=>{if(!["ArrowLeft","ArrowRight","Home","End"].includes(event.key))return;event.preventDefault();const tabs=[...$("tabs").children],index=tabs.indexOf(button),next=event.key==="Home"?0:event.key==="End"?tabs.length-1:(index+(event.key==="ArrowRight"?1:-1)+tabs.length)%tabs.length;tabs[next].click();};
        $("tabs").append(button);
      }
      $("list").setAttribute("aria-labelledby",`ev-tab-${$("filter").value}`);
    }
    previewKey(row){return JSON.stringify([this.data.run,this.data.stem,row.id,this.settings.crop_px,this.settings.cell_lut]);}
    tile(row){
      const button=document.createElement("button");button.type="button";button.className="ev-tile";button.dataset.event=row.id;button.classList.toggle("active",row.id===this.selected?.id);
      button.title=`Cell ${row.identity} · ${row.label} · ${row.status} · source frames ${row.source_frame}–${row.source_end_frame}; open event frames and video`;
      button.innerHTML=`<span class="ev-tile-media"><img hidden alt="Cell ${row.identity}, original event frame"><span class="ev-tile-loading">Loading cell preview…</span><span class="ev-tile-mask" hidden>No mask</span></span><strong>Cell ${row.identity}<span>${row.hours==null?"?":Number(row.hours.toFixed(2))} h</span></strong><small>${html(row.label)}</small><small class="ev-tile-status">${html(row.status)}${this.data.clips.some(c=>c.event_id===row.id)?" · saved video":""}</small>`;
      button.onclick=()=>this.openEvent(row.id);return button;
    }
    showGrid(){this.pause();$("video").pause();this.view="grid";$("gallery").hidden=false;$("detail").hidden=true;}
    openEvent(id){this.view="detail";$("gallery").hidden=true;$("detail").hidden=false;this.choose(id);$("detail").scrollIntoView({block:"start"});this.save();}
    renderList(){
      const rows=this.filtered();$("count").textContent=`${rows.length} / ${this.data?.total||0} events${this.data?.truncated?" (limited)":""}`;
      this.renderTabs();$("list").replaceChildren(...rows.map(row=>this.tile(row)));const tab=this.data?.tabs.find(t=>t.key===$("filter").value);
      $("empty").hidden=!!rows.length;$("empty").textContent=rows.length?"":this.data?.rows.length?`No saved ${tab?.title.toLowerCase()||"events"} match these filters.`:this.data?.message||"No saved events in this recording.";
      this.fillGrid(rows).catch(error=>{$("grid-progress").textContent=error.message;});
      if(this.selected&&!rows.some(r=>r.id===this.selected.id)){if(rows[0])this.choose(rows[0].id);else{this.selected=null;this.empty("No events match these filters.");}}
      else if(!this.selected&&rows[0])this.choose(rows[0].id);
      else if(!rows.length&&!this.selected)this.empty(this.data?.message||"No events match these filters.");
    }
    empty(message="No event selected."){this.pause();++this.frameEpoch;$("heading").textContent="Events";$("message").textContent=message;$("image").hidden=true;$("video").hidden=true;$("transport").hidden=true;$("facts").hidden=true;$("open").disabled=true;$("generate").disabled=true;$("video-mode").disabled=true;$("clip").hidden=true;$("download").hidden=true;$("script").hidden=true;}
    async fillGrid(rows){
      const epoch=++this.galleryEpoch;this.previewAbort?.abort();const controller=this.previewAbort=new AbortController();let completed=0,failed=0;
      const note=()=>{$("grid-progress").textContent=`${completed} / ${rows.length} cell previews${failed?` · ${failed} unavailable`:""}`;$("grid-progress").dataset.completed=completed;$("grid-progress").dataset.total=rows.length;};note();
      const apply=async(row,result)=>{
        if(epoch!==this.galleryEpoch)return;const tile=$("list").querySelector(`[data-event="${row.id}"]`);if(!tile)return;
        const indicator=tile.querySelector(".ev-tile-loading"),picture=tile.querySelector("img");let error=result.error;
        if(!error){try{picture.src=result.image;await picture.decode();}catch{error="The saved preview could not be displayed";}}
        if(epoch!==this.galleryEpoch)return;
        if(error){indicator.textContent=`Preview unavailable: ${error}`;++failed;tile.dataset.preview="error";}
        else{picture.hidden=false;indicator.hidden=true;tile.querySelector(".ev-tile-mask").hidden=result.present;tile.dataset.preview="ready";tile.dataset.sourceFrame=result.source_frame;}
        ++completed;note();
      };
      const pending=[];for(const row of rows){const result=this.previews.get(this.previewKey(row));if(result)await apply(row,result);else pending.push(row);}
      let position=0;const worker=async()=>{
        while(position<pending.length&&epoch===this.galleryEpoch){
          const group=pending.slice(position,position+8);position+=8;
          try{const result=await this.options.api(`/api/events/previews?${new URLSearchParams({run:this.data.run,stem:this.data.stem,event_ids:group.map(row=>row.id).join(","),crop_px:this.settings.crop_px,cell_lut:this.settings.cell_lut})}`,{signal:controller.signal});
            if(epoch!==this.galleryEpoch)return;const found=new Map(result.previews.map(value=>[value.event_id,value]));
            for(const row of group){const preview=found.get(row.id)||{error:"No saved source preview returned"};this.previews.set(this.previewKey(row),preview);while(this.previews.size>512)this.previews.delete(this.previews.keys().next().value);await apply(row,preview);}
          }catch(error){if(epoch!==this.galleryEpoch||controller.signal.aborted)return;for(const row of group)await apply(row,{error:error.message});}
        }
      };await Promise.all([worker(),worker()]);
    }
    render(){if(this.selected){this.renderList();this.choose(this.selected.id,this.frame);}else this.renderList();}
    async choose(id,frame){
      this.pause();$("video").pause();const row=this.data.rows.find(r=>r.id===id);if(!row)return;this.selected=row;this.cache.clear();++this.frameEpoch;
      const frames=this.data.frames,a=frames[row.frame_index].hours,b=frames[row.end_frame].hours,context=this.settings.event_hours;
      const window=frames.filter(f=>f.hours!=null&&a-context<=f.hours&&f.hours<=b+context);this.first=window[0]?.frame_index??row.frame_index;this.last=window.at(-1)?.frame_index??row.end_frame;
      this.frame=Math.max(this.first,Math.min(this.last,frame??row.frame_index));$("slider").min=this.first;$("slider").max=this.last;
      $("heading").textContent=`Cell ${row.identity} · ${row.label}`;$("status-label").textContent=row.status;
      $("facts").hidden=false;$("generate").disabled=false;$("transport").hidden=false;
      $("open").disabled=false;
      $("event-time").textContent=`Source frames ${row.source_frame}–${row.source_end_frame} · ${Number(row.hours?.toFixed(2))} h`;
      $("event-source").textContent=row.source==="tracking"?"Recorded tracking assessment; not a confirmed cell fate.":`Saved ${row.source} event · ${row.status}${row.source==="lifecycle"&&this.data.lifecycle_review?" · classified using original Motion masks":""}`;
      $("event-cells").textContent=row.partners.length?`Related recorded identities: ${row.partners.map(v=>`Cell ${v}`).join(", ")}`:"No related identity recorded.";
      $("evidence").textContent=JSON.stringify(row.evidence,null,2);$("message").textContent="";
      for(const button of $("list").children)button.classList.toggle("active",button.dataset.event===id);
      const clips=this.data.clips.filter(c=>c.event_id===id&&c.token);$("clip").innerHTML=clips.map((c,i)=>`<option value="${i}">${c.options?`${c.options.event_hours} h context · ${c.options.fps} fps · ${html(c.options.cell_lut)}`:"Existing event film"}</option>`).join("");
      this.clips=clips;const preferred=clips.findIndex(c=>c.options&&Object.keys(this.settings).every(k=>c.options[k]===this.settings[k]));if(clips.length)$("clip").value=preferred>=0?preferred:clips.length-1;$("video-mode").disabled=!clips.length;$("clip").hidden=!clips.length;
      const restorePreview=this.saved.event===id&&this.saved.media==="preview";
      this.media(clips.length&&!restorePreview&&(preferred>=0||clips.at(-1).legacy)?"video":"preview");this.save();
    }
    media(mode){
      this.pause();$("video").pause();this.mode=mode;const clip=this.clips?.[Number($("clip").value)||0];
      if(mode==="video"&&!clip){this.mode="preview";mode="preview";}
      $("image").hidden=mode!=="preview";$("video").hidden=mode!=="video";$("transport").hidden=mode!=="preview";
      $("preview-mode").classList.toggle("active",mode==="preview");$("video-mode").classList.toggle("active",mode==="video");
      $("colour-note").textContent=mode==="preview"?"Teal: selected identity. Gold: recorded related identities.":"Teal: selected identity. Grey: neighbouring saved masks.";
      $("download").hidden=!clip;$("script").hidden=!clip?.script_token;
      if(clip){$("download").href=`/api/artifacts/${clip.token}`;$("download").download=clip.name;$("script").href=`/api/artifacts/${clip.script_token}`;}
      if(mode==="video") {if(this.view==="detail")$("video").src=`/api/artifacts/${clip.token}`;$("frame-note").textContent="Saved event video";}
      else if(this.selected&&this.view==="detail"){this.showFrame(this.frame);}
      this.save();
    }
    async fetchFrame(frame){
      const id=this.selected.id,key=JSON.stringify([this.data.run,this.data.stem,id,frame,this.settings]);
      if(this.cache.has(key))return this.cache.get(key);
      const promise=this.options.api(`/api/events/frame?${new URLSearchParams({run:this.data.run,stem:this.data.stem,event_id:id,frame,...{event_hours:this.settings.event_hours,crop_px:this.settings.crop_px,cell_lut:this.settings.cell_lut}})}`);
      this.cache.set(key,promise);while(this.cache.size>48)this.cache.delete(this.cache.keys().next().value);
      try{return await promise;}catch(error){this.cache.delete(key);throw error;}
    }
    async showFrame(frame){
      if(!this.selected)return;const epoch=++this.frameEpoch;this.frame=frame;$("slider").value=frame;$("image").setAttribute("aria-busy","true");
      const clock=this.data.frames[frame];$("frame-note").textContent=`Source frame ${clock.source_frame_index+1} · ${Number(clock.hours?.toFixed(2))} h · loading`;
      try{const result=await this.fetchFrame(frame);if(epoch!==this.frameEpoch||this.mode!=="preview")return;
        const image=new Image();image.src=result.image;await image.decode();if(epoch!==this.frameEpoch||this.mode!=="preview")return;
        $("image").src=result.image;$("image").hidden=false;$("frame-note").textContent=`Source frame ${result.source_frame} · ${Number(result.hours?.toFixed(2))} h · ${result.present?"mask present":"no mask for this identity"}`;
        $("image").dataset.frame=frame;$("image").dataset.eventId=result.event_id;$("message").textContent="";this.save();
        if(this.visible&&frame<this.last)this.fetchFrame(frame+1).catch(()=>{});
      }catch(error){if(epoch===this.frameEpoch){$("image").hidden=true;$("message").textContent=error.message;this.pause();}}
      finally{if(epoch===this.frameEpoch)$("image").removeAttribute("aria-busy");}
    }
    play(){if(!this.selected)return;if(this.frame>=this.last)this.frame=this.first-1;$("play").textContent="Pause";let busy=false;this.timer=setInterval(async()=>{if(busy)return;if(this.frame>=this.last){this.pause();return;}busy=true;await this.showFrame(this.frame+1);busy=false;},1000/this.settings.fps);}
    pause(){clearInterval(this.timer);this.timer=null;$("play").textContent="Play";}
    async generate(){
      if(!this.selected)return;const run=this.data.run,stem=this.data.stem,event_id=this.selected.id;$("generate").disabled=true;
      try{const job=await this.options.api("/api/events/clips",{method:"POST",body:JSON.stringify({run,stem,event_id,options:this.settings})});
        $("job-status").textContent=job.reused?"Saved clip reused":"Event video queued";this.options.onJob?.(job);
        const poll=async()=>{try{const current=await this.options.api(`/api/jobs/${job.id}`),same=this.data?.run===run&&this.data?.stem===stem;
          if(same)$("job-status").textContent=current.state==="failed"?current.error:current.progress?.total?`${current.progress.stage} · ${current.progress.completed} / ${current.progress.total} frames`:`Event video ${current.state}`;
          if(["completed","failed"].includes(current.state)){if(same){$("generate").disabled=!this.selected;if(current.state==="completed"){this.saved={...this.saved,version:2,run,stem,event:this.selected?.id,frame:this.frame,settings:this.settings,media:"video"};await this.load(run,stem);}}return;}
          setTimeout(poll,500);
        }catch(error){if(this.data?.run===run){$("job-status").textContent=error.message;$("generate").disabled=!this.selected;}}};poll();
      }catch(error){$("job-status").textContent=error.message;$("generate").disabled=!this.selected;}
    }
  }
  window.PyMicrogliaEvents={mount:options=>new EventBrowser(options)};
})();
