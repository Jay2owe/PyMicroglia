"use strict";
(() => {
  const data = JSON.parse(document.getElementById("review-data").textContent);
  const $ = id => document.getElementById(id);
  // Start on an existing observation, rather than the smallest ID, which
  // might not appear until much later. This selects a view; it assigns no IDs.
  const startingFrame=Math.min(data.frame_count-1,data.source_frame_offset+1);
  const atFrame=frame=>data.ids.filter(id=>data.cells[String(id)].observations[frame]);
  const pairAt=frame=>frame>0?atFrame(frame).filter(id=>data.cells[String(id)].observations[frame-1]):[];
  let initialFrame=startingFrame, initialIds=pairAt(startingFrame);
  if(!initialIds.length) {
    for(let frame=1;frame<data.frame_count;frame++) {
      const ids=pairAt(frame);if(ids.length) {initialFrame=frame;initialIds=ids;break;}
    }
  }
  if(!initialIds.length) {
    initialIds=atFrame(initialFrame);
    if(!initialIds.length) for(let frame=0;frame<data.frame_count;frame++) {
      const ids=atFrame(frame);if(ids.length) {initialFrame=frame;initialIds=ids;break;}
    }
  }
  const initialCell=initialIds[0] ?? data.ids[0] ?? null;
  const initialPins=[...atFrame(initialFrame),...data.ids.filter(id=>!atFrame(initialFrame).includes(id))].filter(id=>id!==initialCell).slice(0,2);
  const R = window.CellReview = {
    data, $, views: [], ready: false, imageCache: new Map(), renderToken: 0,
    colours: {cyan:"rgb(65,225,232)", amber:"rgb(255,185,71)", white:"rgb(241,247,252)", muted:"rgb(160,182,201)"},
    state: {cellId: initialCell, frame: initialFrame,
            playing: false, fps:4, changeMode:"centered", sculptureMode:"centered", pinned:initialPins},
    cell(id=this.state.cellId) {return data.cells[String(id)] ?? null;},
    observed(id=this.state.cellId, frame=this.state.frame) {return this.cell(id)?.observations[frame] ?? null;},
    location(cell, frame) {
      if (!cell) return null;
      for (let f=frame; f>=0; f--) if (cell.observations[f]) return cell.observations[f];
      return cell.observations.find(Boolean) ?? null;
    },
    number(value) {
      if (value == null) return "—";
      if (Math.abs(value)>=100000) return value.toExponential(2);
      return value.toLocaleString(undefined,{maximumFractionDigits:1});
    },
    hours(value) {return value.toLocaleString(undefined,{maximumFractionDigits:6});},
    getImage(frame) {
      if (this.imageCache.has(frame)) return this.imageCache.get(frame);
      const promise = new Promise((resolve,reject) => {
        const img = new Image(); img.onload=()=>resolve(img);
        img.onerror=()=>reject(new Error(`Could not decode original frame ${frame+1}`)); img.src=data.frames[frame];
      });
      this.imageCache.set(frame,promise);
      while(this.imageCache.size>6) this.imageCache.delete(this.imageCache.keys().next().value);
      return promise;
    },
    drawRaw(canvas,image,item,size) {
      const ctx=canvas.getContext("2d"); ctx.imageSmoothingEnabled=false;
      ctx.fillStyle="black"; ctx.fillRect(0,0,canvas.width,canvas.height);
      if (!item || !image) return {ctx,left:0,top:0,scale:1,size};
      const left=item.anchor[0]-Math.floor(size/2), top=item.anchor[1]-Math.floor(size/2);
      const scale=canvas.width/size;
      ctx.drawImage(image,left,top,size,size,0,0,canvas.width,canvas.height);
      return {ctx,left,top,scale,size};
    },
    contours(view,item,{colour=this.colours.white,width=1.4,offset=[0,0]}={}) {
      if(!item) return;
      const ctx=view.ctx; ctx.strokeStyle=colour; ctx.lineWidth=width;
      for(const contour of item.contours) {
        ctx.beginPath();
        contour.forEach(([x,y],i)=>{
          const cx=(x+offset[0]-view.left)*view.scale, cy=(y+offset[1]-view.top)*view.scale;
          if(i===0) ctx.moveTo(cx,cy); else ctx.lineTo(cx,cy);
        }); ctx.stroke();
      }
    },
    scaleBar(canvas,view) {
      const wanted=view.size*.23, power=10**Math.floor(Math.log10(wanted));
      const length=[1,2,5,10].filter(n=>n*power<=wanted).pop()*power;
      const ctx=view.ctx, y=canvas.height-25;
      ctx.fillStyle="rgba(0,0,0,.8)"; ctx.fillRect(12,y-22,length*view.scale+16,34);
      ctx.strokeStyle="white"; ctx.lineWidth=2; ctx.beginPath();ctx.moveTo(20,y);ctx.lineTo(20+length*view.scale,y);ctx.stroke();
      ctx.fillStyle="white";ctx.font="13px system-ui";ctx.fillText(`${length} px`,20,y-7);
    },
    async select(patch,{pause=true}={}) {
      if(pause) this.pause();
      if(patch.cellId!=null && !data.ids.includes(Number(patch.cellId))) return;
      Object.assign(this.state,patch);
      this.state.frame=Math.max(0,Math.min(data.frame_count-1,Math.round(this.state.frame)));
      return this.update();
    },
    pause() {this.state.playing=false;clearTimeout(this.timer);$("play").textContent="Play";},
    async play() {
      if(this.state.playing) {this.pause();return;}
      if(this.state.frame===data.frame_count-1) this.state.frame=0;
      this.state.playing=true;$("play").textContent="Pause";
      const tick=async()=>{
        if(!this.state.playing) return;
        await this.select({frame:this.state.frame+1},{pause:false});
        if(this.state.frame>=data.frame_count-1) {this.pause();return;}
        if(this.state.playing) this.timer=setTimeout(tick,1000/this.state.fps);
      };
      await this.update(); this.timer=setTimeout(tick,1000/this.state.fps);
    },
    async update() {
      const token=++this.renderToken, frame=this.state.frame;
      $("cell-select").value=String(this.state.cellId);$("frame-slider").value=frame;
      $("time-label").textContent=`Frame ${frame+1} / ${data.frame_count} · ${R.hours(data.times_h[frame])} h`;
      $("step-back").disabled=frame===0;$("step-forward").disabled=frame===data.frame_count-1;
      try {
        const image=await this.getImage(frame); if(token!==this.renderToken) return;
        for(const view of this.views) view.render(image);
        this.renderedFrame=frame;this.renderedCell=this.state.cellId;this.ready=true;
        $("review-root").dataset.frame=String(frame);$("review-root").dataset.cell=String(this.state.cellId);
      } catch(error) {$("review-error").hidden=false;$("review-error").textContent=error.message; throw error;}
    },
    async start() {
      $("recording-title").textContent=data.title;document.title=data.title+" · Cell review";
      $("recording-summary").textContent=`${data.ids.length} saved identities · ${data.frame_count} original frames · ${R.number(data.times_h.at(-1))} hours`;
      for(const id of data.ids) {const option=document.createElement("option");option.value=id;option.textContent=`Cell ${id}`;$("cell-select").append(option);}
      $("cell-select").onchange=event=>this.select({cellId:Number(event.target.value)});
      $("frame-slider").max=data.frame_count-1;$("frame-slider").oninput=event=>this.select({frame:Number(event.target.value)});
      $("step-back").onclick=()=>this.select({frame:this.state.frame-1});$("step-forward").onclick=()=>this.select({frame:this.state.frame+1});
      $("play").onclick=()=>this.play();$("speed").onchange=event=>{this.state.fps=Number(event.target.value);};
      document.addEventListener("keydown",event=>{
        if(["INPUT","SELECT","BUTTON"].includes(event.target.tagName)) return;
        if(event.key===" ") {event.preventDefault();this.play();}
        if(event.key==="ArrowLeft") {event.preventDefault();this.select({frame:this.state.frame-1});}
        if(event.key==="ArrowRight") {event.preventDefault();this.select({frame:this.state.frame+1});}
      });
      for(const view of this.views) if(view.init) view.init();
      await this.update();
    }
  };
})();
