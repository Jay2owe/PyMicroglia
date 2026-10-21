"use strict";
(() => {
  const R=window.CellReview, ns="http://www.w3.org/2000/svg";
  const el=(name,attrs={},text)=>{
    const node=document.createElementNS(ns,name);for(const [key,value] of Object.entries(attrs)) node.setAttribute(key,value);
    if(text!==undefined) node.textContent=text;return node;
  };
  function trace(cell,metric) {
    const isSignal=metric==="signal_raw",values=cell[metric],mean=isSignal?cell.signal_mean3:null;
    const svg=el("svg",{viewBox:"0 0 350 130",class:"trace",role:"img","aria-label":isSignal?`Cell ${cell.id} total intensity trace`:`Cell ${cell.id} mask area trace`});
    const width=350,height=130,left=42,right=8,top=12,bottom=24;
    const finite=values.filter(v=>v!==null),min=finite.length?Math.min(...finite):0,max=finite.length?Math.max(...finite):1;
    const span=Math.max(max-min,1),lo=min-span*.07,hi=max+span*.07;
    const x=f=>left+f/Math.max(1,R.data.frame_count-1)*(width-left-right);
    const y=value=>top+(hi-value)/(hi-lo)*(height-top-bottom);
    for(const f of [0,.5,1]) {
      const gy=top+f*(height-top-bottom);
      svg.append(el("line",{x1:left,y1:gy,x2:width-right,y2:gy,stroke:"rgb(34,49,65)","stroke-width":.7}));
    }
    for(const [value,yy] of [[max,y(max)],[min,y(min)]]) svg.append(el("text",{x:left-5,y:yy+3,fill:R.colours.muted,"text-anchor":"end","font-size":9},R.number(value)));
    for(const frame of [0,R.data.frame_count-1]) svg.append(el("text",{x:x(frame),y:height-7,fill:R.colours.muted,"text-anchor":frame===0?"start":"end","font-size":9},`${R.hours(R.data.times_h[frame])} h`));
    function path(series,colour,lineWidth,opacity=1) {
      let d="",open=false;
      series.forEach((value,frame)=>{
        if(value===null) {open=false;return;}
        d+=`${open?"L":"M"}${x(frame).toFixed(3)},${y(value).toFixed(3)} `;open=true;
        // Isolated observations are visible without connecting across gaps.
        svg.append(el("circle",{cx:x(frame),cy:y(value),r:.8,fill:colour,opacity}));
      });
      const node=el("path",{d,fill:"none",stroke:colour,"stroke-width":lineWidth,opacity,"data-series":series===mean?"mean3":metric});svg.append(node);
    }
    path(values,isSignal?R.colours.muted:R.colours.amber,isSignal?1:1.6,isSignal?.5:1);
    if(mean) path(mean,R.colours.cyan,1.8);
    const cursor=el("line",{y1:top,y2:height-bottom,class:"cursor"});svg.append(cursor);
    const dot=el("circle",{r:3.3,fill:isSignal?R.colours.cyan:R.colours.amber});svg.append(dot);
    svg.addEventListener("click",event=>{
      const box=svg.getBoundingClientRect(),px=(event.clientX-box.left)/box.width*width;
      const frame=Math.round((px-left)/(width-left-right)*(R.data.frame_count-1));R.select({cellId:cell.id,frame});
    });
    return {svg,update(frame) {
      cursor.setAttribute("x1",x(frame));cursor.setAttribute("x2",x(frame));cursor.dataset.frame=String(frame);
      const value=(mean??values)[frame];dot.style.display=value===null?"none":"";
      if(value!==null) {dot.setAttribute("cx",x(frame));dot.setAttribute("cy",y(value));}
    }};
  }
  const cards=R.cards={key:null,items:[],
    init() {
      R.$("pin-cell").onclick=()=>{
        const id=R.state.cellId;if(id===null) return;
        const found=R.state.pinned.indexOf(id);
        if(found>=0) R.state.pinned.splice(found,1);else if(R.state.pinned.length<5) R.state.pinned.push(id);
        R.update();
      };
    },
    sync() {
      const ids=[R.state.cellId,...R.state.pinned.filter(id=>id!==R.state.cellId)].filter(id=>id!==null);
      const key=ids.join(",");if(key===this.key) return;
      this.key=key;this.items=[];R.$("cards").replaceChildren();
      for(const id of ids) {
        const cell=R.cell(id),root=document.createElement("article");root.className="cell-card";root.dataset.cell=String(id);
        const heading=document.createElement("div");heading.className="card-heading";
        const title=document.createElement("h3");title.textContent=`Cell ${id}`;heading.append(title);
        const button=document.createElement("button");button.textContent=id===R.state.cellId?"Selected":"Select";
        button.onclick=()=>R.select({cellId:id});heading.append(button);root.append(heading);
        const body=document.createElement("div");body.className="card-body";
        const movie=document.createElement("canvas");movie.width=movie.height=360;movie.className="movie";movie.setAttribute("aria-label",`Cell ${id} movie`);body.append(movie);
        const plots=document.createElement("div");plots.className="card-traces";
        const signal=trace(cell,"signal_raw"),area=trace(cell,"area");
        const signalLabel=document.createElement("div");signalLabel.className="trace-label";
        const signalName=document.createElement("span");signalName.textContent="Total intensity · sum";
        const signalValue=document.createElement("span");signalLabel.append(signalName,signalValue);
        const areaLabel=document.createElement("div");areaLabel.className="trace-label";
        const areaName=document.createElement("span");areaName.textContent="Mask area · pixels";
        const areaValue=document.createElement("span");areaLabel.append(areaName,areaValue);
        plots.append(signalLabel,signal.svg,areaLabel,area.svg);body.append(plots);root.append(body);
        const missing=document.createElement("p");missing.className="card-missing";root.append(missing);R.$("cards").append(root);
        this.items.push({id,cell,root,movie,signal,area,signalValue,areaValue,missing});
      }
    },
    render(image) {
      this.sync();const frame=R.state.frame;
      for(const item of this.items) {
        const obs=R.observed(item.id),location=obs??R.location(item.cell,frame);
        const view=R.drawRaw(item.movie,image,location,item.cell.crop_size_px);
        R.contours(view,obs);R.scaleBar(item.movie,view);
        item.root.classList.toggle("selected",item.id===R.state.cellId);
        item.signal.update(frame);item.area.update(frame);
        item.signalValue.textContent=R.number(item.cell.signal_mean3[frame]);
        item.signalValue.title="Total intensity, centred three-frame mean";
        item.areaValue.textContent=R.number(item.cell.area[frame]);
        item.missing.textContent=obs?`Frame ${frame+1} · ${R.hours(R.data.times_h[frame])} h`:"No saved mask at this time";
        item.missing.style.color=obs?R.colours.muted:R.colours.amber;
      }
      const pinned=R.state.pinned.includes(R.state.cellId);
      R.$("pin-cell").textContent=pinned?"Unpin selected cell":R.state.pinned.length>=5?"Five comparison cells pinned":"Pin selected cell";
      R.$("pin-cell").disabled=R.state.cellId===null || (!pinned && R.state.pinned.length>=5);
    }
  };
  R.views.push(cards);
})();
