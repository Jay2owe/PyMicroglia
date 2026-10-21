"use strict";
(() => {
  const R=window.CellReview, metricNames={signal_raw:"Total intensity · raw",signal_mean3:"Total intensity · three-frame mean",area:"Mask area · pixels"};
  function extent(values) {
    let low=Infinity,high=-Infinity;
    for(const value of values) if(value!==null && Number.isFinite(value)) {low=Math.min(low,value);high=Math.max(high,value);}
    return low===Infinity?[0,1]:[low,high];
  }
  function colour(value,range) {
    const f=range[1]===range[0]?.5:Math.max(0,Math.min(1,(value-range[0])/(range[1]-range[0])));
    const stops=[[23,43,78],[43,139,178],[244,219,139]],k=f<=.5?0:1,t=f<=.5?f*2:(f-.5)*2;
    return `rgb(${stops[k].map((channel,i)=>Math.round(channel+(stops[k+1][i]-channel)*t)).join(",")})`;
  }
  const barcode=R.barcode={metric:"signal_mean3",scale:"cell",background:null,key:null,
    init() {
      R.$("barcode-metric").onchange=event=>{this.metric=event.target.value;this.key=null;R.update();};
      R.$("barcode-scale").onchange=event=>{this.scale=event.target.value;this.key=null;R.update();};
      const canvas=R.$("barcode");
      canvas.addEventListener("click",event=>{const hit=this.hit(event);if(hit) R.select({cellId:hit.id,frame:hit.frame});});
      canvas.addEventListener("pointermove",event=>{const hit=this.hit(event);if(hit) this.tooltip(hit);});
      canvas.addEventListener("keydown",event=>{
        if(event.key!=="ArrowUp" && event.key!=="ArrowDown") return;
        event.preventDefault();event.stopPropagation();const row=R.data.ids.indexOf(R.state.cellId)+(event.key==="ArrowDown"?1:-1);
        R.select({cellId:R.data.ids[Math.max(0,Math.min(R.data.ids.length-1,row))]});
      });
      window.addEventListener("resize",()=>{this.key=null;R.update();});
    },
    hit(event) {
      const g=this.geometry;if(!g) return null;const box=R.$("barcode").getBoundingClientRect();
      const x=(event.clientX-box.left)/box.width*g.width,y=(event.clientY-box.top)/box.height*g.height;
      const row=Math.floor((y-g.top)/g.rowHeight),frame=Math.floor((x-g.left)/g.columnWidth);
      if(row<0 || row>=R.data.ids.length || frame<0 || frame>=R.data.frame_count) return null;
      return {id:R.data.ids[row],frame,row,value:R.cell(R.data.ids[row])[this.metric][frame]};
    },
    tooltip(hit) {
      let value=R.number(hit.value);
      if(hit.value===null) value=R.observed(hit.id,hit.frame)?(this.metric==="signal_mean3"?"no complete consecutive three-frame signal window":"no finite measured value"):"no saved mask";
      R.$("barcode-tooltip").textContent=`Cell ${hit.id} · frame ${hit.frame+1} · ${R.hours(R.data.times_h[hit.frame])} h · ${metricNames[this.metric]}: ${value}`;
    },
    prepare() {
      const canvas=R.$("barcode"),container=canvas.parentElement,width=Math.max(640,Math.round(container.clientWidth)),dpr=window.devicePixelRatio||1;
      const key=`${this.metric}:${this.scale}:${width}:${dpr}`;if(this.key===key) return;
      this.key=key;
      const left=70,right=15,top=36,rowHeight=18,bottom=28,height=top+rowHeight*R.data.ids.length+bottom;
      this.geometry={width,height,left,top,rowHeight,columnWidth:(width-left-right)/R.data.frame_count};
      canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);canvas.style.width=`${width}px`;canvas.style.height=`${height}px`;
      const backing=document.createElement("canvas");backing.width=canvas.width;backing.height=canvas.height;
      const ctx=backing.getContext("2d");ctx.scale(dpr,dpr);ctx.fillStyle="rgb(12,19,29)";ctx.fillRect(0,0,width,height);
      const globalRange=extent(R.data.ids.flatMap(id=>R.cell(id)[this.metric]));this.ranges={};
      ctx.font="11px system-ui";ctx.textBaseline="middle";
      let finite=0;
      for(let row=0;row<R.data.ids.length;row++) {
        const id=R.data.ids[row],series=R.cell(id)[this.metric],range=this.scale==="cell"?extent(series):globalRange;
        this.ranges[id]=range;const y=top+row*rowHeight;
        ctx.fillStyle=row%2===0?"rgb(14,23,35)":"rgb(18,27,39)";ctx.fillRect(left,y,width-left-right,rowHeight-1);
        ctx.fillStyle=R.colours.muted;ctx.textAlign="right";ctx.fillText(String(id),left-12,y+rowHeight/2);
        for(let frame=0;frame<series.length;frame++) {
          const value=series[frame];if(value===null) continue;finite++;
          ctx.fillStyle=colour(value,range);ctx.fillRect(left+frame*this.geometry.columnWidth,y,this.geometry.columnWidth+.15,rowHeight-1);
        }
      }
      ctx.textAlign="right";ctx.fillStyle=R.colours.muted;ctx.fillText("Cell",left-12,18);
      for(let step=0;step<=5;step++) {
        const f=step/5,x=left+f*(width-left-right),time=R.data.times_h[0]+f*(R.data.times_h.at(-1)-R.data.times_h[0]);
        ctx.textAlign=step===0?"left":step===5?"right":"center";ctx.fillText(`${R.hours(time)} h`,x,18);
      }
      this.background=backing;this.finiteCount=finite;
      R.$("barcode-status").textContent=`${metricNames[this.metric]} · ${this.scale==="cell"?"each cell uses its own colour range":"one colour range for the entire recording"} · ${finite.toLocaleString()} observed values`;
    },
    render() {
      this.prepare();const canvas=R.$("barcode"),ctx=canvas.getContext("2d"),g=this.geometry,dpr=window.devicePixelRatio||1;
      ctx.setTransform(1,0,0,1,0,0);ctx.drawImage(this.background,0,0);ctx.scale(dpr,dpr);
      const row=R.data.ids.indexOf(R.state.cellId),x=g.left+(R.state.frame+.5)*g.columnWidth;
      ctx.strokeStyle="rgba(255,255,255,.7)";ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x,g.top);ctx.lineTo(x,g.height-28);ctx.stroke();
      if(row>=0) {
        const y=g.top+row*g.rowHeight;ctx.strokeStyle=R.colours.cyan;ctx.lineWidth=1.5;ctx.strokeRect(g.left,y,g.width-g.left-15,g.rowHeight-1);
        ctx.strokeStyle="white";ctx.lineWidth=2;ctx.strokeRect(g.left+R.state.frame*g.columnWidth,y,g.columnWidth,g.rowHeight-1);
        this.tooltip({id:R.state.cellId,frame:R.state.frame,value:R.cell()[this.metric][R.state.frame]});
        if(this.previousId!==R.state.cellId) {
          const container=canvas.parentElement;if(y<container.scrollTop || y+g.rowHeight>container.scrollTop+container.clientHeight) container.scrollTop=Math.max(0,y-container.clientHeight/2);
          this.previousId=R.state.cellId;
        }
      }
      canvas.dataset.currentCell=String(R.state.cellId);canvas.dataset.currentFrame=String(R.state.frame);
    }
  };
  R.views.push(barcode);
})();
