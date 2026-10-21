"use strict";
(() => {
  const R=window.CellReview, initial={yaw:-.65,pitch:.58,zoom:1};
  const sculpture=R.sculpture={...initial, mode:"centered", geometry:null, geometryKey:null, observedFrames:[],
    init() {
      const canvas=R.$("sculpture");
      R.$("sculpture-mode").onchange=event=>{this.mode=event.target.value;this.geometryKey=null;this.draw();};
      R.$("sculpture-reset").onclick=()=>{Object.assign(this,initial);this.draw();};
      canvas.addEventListener("pointerdown",event=>{
        this.drag={x:event.clientX,y:event.clientY};canvas.setPointerCapture(event.pointerId);
      });
      canvas.addEventListener("pointermove",event=>{
        if(!this.drag) return;
        this.yaw+=(event.clientX-this.drag.x)*.008;
        this.pitch=Math.max(.12,Math.min(1.42,this.pitch+(event.clientY-this.drag.y)*.006));
        this.drag={x:event.clientX,y:event.clientY};this.draw();
      });
      canvas.addEventListener("pointerup",()=>{this.drag=null;});
      canvas.addEventListener("pointercancel",()=>{this.drag=null;});
      canvas.addEventListener("wheel",event=>{
        event.preventDefault();this.zoom=Math.max(.5,Math.min(5,this.zoom*Math.exp(-event.deltaY*.001)));this.draw();
      },{passive:false});
      canvas.addEventListener("keydown",event=>{
        if(!event.key.startsWith("Arrow")) return;
        event.stopPropagation();event.preventDefault();
        if(event.key==="ArrowLeft") this.yaw-=.12;if(event.key==="ArrowRight") this.yaw+=.12;
        if(event.key==="ArrowUp") this.pitch=Math.min(1.42,this.pitch+.1);
        if(event.key==="ArrowDown") this.pitch=Math.max(.12,this.pitch-.1);this.draw();
      });
    },
    prepare() {
      const key=`${R.state.cellId}:${this.mode}`;
      if(this.geometryKey===key) return;
      this.geometryKey=key;const cell=R.cell();this.observedFrames=[];
      if(!cell) {this.geometry=null;return;}
      const slices=[];let xmin=Infinity,ymin=Infinity,xmax=-Infinity,ymax=-Infinity;
      cell.observations.forEach((item,frame)=>{
        if(!item) return;
        this.observedFrames.push(frame);
        const anchor=this.mode==="centered"?[item.anchor[0]+.5,item.anchor[1]+.5]:[0,0];
        const contours=item.contours.map(contour=>contour.map(([x,y])=>{
          const px=x-anchor[0],py=y-anchor[1];xmin=Math.min(xmin,px);xmax=Math.max(xmax,px);ymin=Math.min(ymin,py);ymax=Math.max(ymax,py);
          return [px,py];
        }));
        slices.push({frame,time:R.data.times_h[frame],contours});
      });
      if(!slices.length) {this.geometry=null;return;}
      const span=Math.max(xmax-xmin,ymax-ymin,10),cx=(xmin+xmax)/2,cy=(ymin+ymax)/2;
      const duration=R.data.times_h.at(-1)-R.data.times_h[0],height=span*1.5;
      slices.forEach(slice=>{slice.z=duration>0?(slice.time-R.data.times_h[0])/duration*height:0;
        slice.contours=slice.contours.map(contour=>contour.map(([x,y])=>[x-cx,y-cy,slice.z]));});
      this.geometry={slices,span,height,xmin:xmin-cx,xmax:xmax-cx,ymin:ymin-cy,ymax:ymax-cy};
    },
    project(point) {
      const [x,y,z]=point,c=Math.cos(this.yaw),s=Math.sin(this.yaw);
      const xr=x*c-y*s,yr=x*s+y*c;
      return [xr,yr*Math.sin(this.pitch)-z*Math.cos(this.pitch),yr*Math.cos(this.pitch)+z*Math.sin(this.pitch)];
    },
    draw() {
      this.prepare();const canvas=R.$("sculpture"),ctx=canvas.getContext("2d");
      ctx.clearRect(0,0,canvas.width,canvas.height);
      if(!this.geometry) {R.$("sculpture-status").textContent="No saved outlines in this recording.";return;}
      const g=this.geometry,extent=g.span*.64,volume=[];
      for(const x of [-extent,extent]) for(const y of [-extent,extent]) for(const z of [0,g.height]) volume.push(this.project([x,y,z]));
      const xs=volume.map(p=>p[0]),ys=volume.map(p=>p[1]);
      const xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);
      const scale=Math.min((canvas.width-160)/(xmax-xmin),(canvas.height-100)/(ymax-ymin))*this.zoom;
      const centre=[(xmin+xmax)/2,(ymin+ymax)/2];
      const screen=point=>{const p=this.project(point);return [(p[0]-centre[0])*scale+canvas.width/2,(p[1]-centre[1])*scale+canvas.height/2,p[2]];};
      this.projection={scale,yaw:this.yaw,pitch:this.pitch,zoom:this.zoom};
      const line=(a,b,colour,width=1)=>{const pa=screen(a),pb=screen(b);ctx.strokeStyle=colour;ctx.lineWidth=width;ctx.beginPath();ctx.moveTo(...pa.slice(0,2));ctx.lineTo(...pb.slice(0,2));ctx.stroke();};
      // The floor grid is a spatial reference only; no masks are connected
      // between time slices, so missing observations stay missing.
      for(let k=0;k<=4;k++) {
        const q=-extent+2*extent*k/4;
        line([-extent,q,0],[extent,q,0],"rgb(33,47,64)");line([q,-extent,0],[q,extent,0],"rgb(33,47,64)");
      }
      const slices=g.slices.map(slice=>({...slice,depth:this.project([0,0,slice.z])[2]})).sort((a,b)=>a.depth-b.depth);
      for(const slice of slices) {
        const fraction=R.data.frame_count>1?slice.frame/(R.data.frame_count-1):0;
        const current=slice.frame===R.state.frame;
        ctx.strokeStyle=current?R.colours.white:`rgba(${Math.round(164-99*fraction)},${Math.round(126+99*fraction)},${Math.round(226+6*fraction)},${.35+.4*fraction})`;
        ctx.lineWidth=current?2.7:1.2;
        for(const contour of slice.contours) {
          ctx.beginPath();contour.forEach((point,i)=>{const p=screen(point);if(i===0) ctx.moveTo(p[0],p[1]);else ctx.lineTo(p[0],p[1]);});ctx.stroke();
        }
      }
      const base=[-extent,-extent,0];line(base,[-extent,-extent,g.height],"rgb(114,161,182)",1.5);
      ctx.font="14px system-ui";ctx.fillStyle=R.colours.muted;
      for(const f of [0,.5,1]) {
        const p=screen([-extent,-extent,f*g.height]);ctx.fillText(`${R.hours(R.data.times_h[0]+f*(R.data.times_h.at(-1)-R.data.times_h[0]))} h`,p[0]-48,p[1]+5);
      }
      const timeTop=screen([-extent,-extent,g.height]);ctx.fillStyle=R.colours.cyan;ctx.fillText("Time",timeTop[0]-44,timeTop[1]-15);
      line(base,[extent,-extent,0],"rgb(94,118,140)");line(base,[-extent,extent,0],"rgb(94,118,140)");
      const xEnd=screen([extent,-extent,0]),yEnd=screen([-extent,extent,0]);ctx.fillStyle=R.colours.muted;
      ctx.fillText(`x · ${R.number(2*extent)} px`,xEnd[0]-50,xEnd[1]+21);ctx.fillText(`y · ${R.number(2*extent)} px`,yEnd[0]-45,yEnd[1]+21);
      const atCurrent=R.observed()!=null;
      R.$("sculpture-status").textContent=`Cell ${R.state.cellId} · ${this.observedFrames.length} observed times / ${R.data.frame_count} frames${atCurrent?"":" · no outline at the selected time"}`;
      canvas.dataset.observedFrames=this.observedFrames.join(",");canvas.dataset.currentFrame=String(R.state.frame);
    },render() {this.draw();}
  };
  R.views.push(sculpture);
})();
