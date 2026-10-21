"use strict";
(() => {
  const $ = id => document.getElementById(id);
  const N = 240, COUNT = 24, DT = 0.5, TAU = Math.PI * 2;
  const C = {ink: "#dceef2", muted: "#8eacba", grid: "#213c4b", cyan: "#79e3d2", gold: "#e8b66e", blue: "#91adf0"};
  const state = {cell: 19, frame: 72, workspace: "extras", yaw: -0.62, pitch: 0.42, zoom: 1, playing: false};
  const metrics = {light: {label: "Light", unit: "counts"}, area: {label: "Mask area", unit: "px²"}, turnover: {label: "Footprint turnover", unit: "%"}, speed: {label: "Speed", unit: "px/h"}};
  const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
  const avg = a => a.reduce((s, v) => s + v, 0) / a.length;
  const fmt = (v, digits = 0) => v == null ? "Missing" : v.toLocaleString("en-GB", {maximumFractionDigits: digits});
  const hash = (a, b, c = 0) => { const n = Math.sin(a * 127.1 + b * 311.7 + c * 74.7) * 43758.5453; return n - Math.floor(n); };
  const missing = (id, f) => (id === 7 && f >= 81 && f <= 92) || (id === 19 && f >= 168 && f <= 173) || (id === 12 && f < 12);
  const area = p => Math.abs(p.reduce((s, v, i) => {const q = p[(i + 1) % p.length]; return s + v[0] * q[1] - q[0] * v[1];}, 0)) / 2;
  const xy = (id, f) => {
    const i = id - 1, h = f * DT;
    return [64 + (i % 6) * 91 + 12 * Math.sin(h / 11 + i) + 6 * Math.sin(h / 4 + i / 3),
      55 + Math.floor(i / 6) * 87 + 13 * Math.cos(h / 13 + i * 1.7) + 5 * Math.cos(h / 5 + i)];
  };
  const makeOutline = (id, f) => {
    const h = f * DT, p = xy(id, f), phase = id * 0.41;
    return Array.from({length: 64}, (_, j) => {
      const a = TAU * j / 64, rot = phase + 0.1 * Math.sin(h / 8 + id);
      let r = 7 + id % 4 + 0.8 * Math.sin(a * 3 + h / 9);
      for (let arm = 0; arm < 5; arm++) {
        const centre = arm * TAU / 5 + rot + 0.18 * Math.sin(h / 6 + arm);
        const distance = Math.atan2(Math.sin(a - centre), Math.cos(a - centre));
        r += (9 + 5 * Math.sin(h / 7 + arm + phase)) * Math.exp(-distance * distance / 0.035);
      }
      return Object.freeze([p[0] + r * Math.cos(a), p[1] + r * Math.sin(a)]);
    });
  };
  const cells = Array.from({length: COUNT}, (_, i) => {
    const id = i + 1, phase = (i * 1.13 + (i >= 12 ? 3 : 0)) % 24, group = i < 12 ? "A" : "B";
    const geometry = Array.from({length: N}, (_, f) => makeOutline(id, f));
    const observations = geometry.map((outline, f) => {
      if (missing(id, f)) return null;
      const p = xy(id, f), prior = f ? xy(id, f - 1) : p;
      const maskArea = area(outline), mean = 22 + id % 7 * 2 + (group === "B" ? 5 : 0) + 11 * Math.cos(TAU * (f * DT - phase) / 24) + 3 * (hash(id, f) - 0.5);
      const base = outline.map((_, j) => Object.freeze([0, 1].map(axis => avg(geometry.slice(Math.max(0, f - 3), Math.min(N, f + 4)).map(shape => shape[j][axis])))));
      return Object.freeze({x: p[0], y: p[1], outline: Object.freeze(outline), base: Object.freeze(base), area: maskArea, mean, total: maskArea * mean,
        speed: f ? Math.hypot(p[0] - prior[0], p[1] - prior[1]) / DT : 0,
        turnover: 4 + 5 * (1 + Math.sin(f / 10 + id * 0.31)) + 2 * hash(f, id)});
    });
    return Object.freeze({id, phase, group, observations: Object.freeze(observations)});
  });
  Object.freeze(cells);
  const selected = () => cells[state.cell - 1];
  const observation = () => selected().observations[state.frame];
  const rawValue = (cell, f, metric, aggregation = "total") => {
    const o = cell.observations[f]; return o ? (metric === "light" ? o[aggregation] : o[metric]) : null;
  };
  const value = (cell, f, metric, aggregation, smoothing = 1) => {
    const v = rawValue(cell, f, metric, aggregation);
    if (v == null || smoothing === 1) return v;
    const values = [v], half = Math.floor(smoothing / 2);
    for (const step of [-1, 1]) for (let k = 1; k <= half; k++) {
      const t = f + k * step;
      if (t < 0 || t >= N || cell.observations[t] == null) break;
      values.push(rawValue(cell, t, metric, aggregation));
    }
    return avg(values);
  };
  const settings = prefix => ({metric: $(prefix + "-metric").value, aggregation: $(prefix + "-aggregation").value, smoothing: Number($(prefix + "-smoothing")?.value || 1)});
  const unit = s => s.metric === "light" && s.aggregation === "mean" ? "counts/px" : metrics[s.metric].unit;
  const metricLabel = s => s.metric === "light" ? (s.aggregation === "mean" ? "Mean light per pixel" : "Total light in mask") : metrics[s.metric].label;
  const dataSeries = (cell, s) => Array.from({length: N}, (_, f) => value(cell, f, s.metric, s.aggregation, s.smoothing));
  const bounds = a => {const v = a.filter(x => x != null && Number.isFinite(x)); return v.length ? [Math.min(...v), Math.max(...v) + 0.001] : [0, 1];};
  const ramp = t => {t = clamp(t, 0, 1); const a = t < 0.65 ? [23, 56, 76] : [66, 159, 168], b = t < 0.65 ? [66, 159, 168] : [238, 210, 148], k = t < 0.65 ? t / 0.65 : (t - 0.65) / 0.35; return `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * k)).join(",")})`;};
  const phaseColour = phase => `hsl(${(phase / 24 * 300 + 140) % 360},50%,65%)`;
  const text = (ctx, str, x, y, colour = C.muted, size = 12, align = "left") => {ctx.fillStyle = colour; ctx.font = `${size}px Segoe UI, sans-serif`; ctx.textAlign = align; ctx.fillText(str, x, y);};
  const line = (ctx, a, b, colour = C.grid, width = 1) => {ctx.strokeStyle = colour;ctx.lineWidth = width;ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();};
  const circle = (ctx, x, y, r, fill, stroke) => {ctx.beginPath();ctx.arc(x,y,r,0,TAU);if(fill){ctx.fillStyle=fill;ctx.fill();}if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=2;ctx.stroke();}};
  const polygon = (ctx, points, stroke, fill, width = 1) => {ctx.beginPath();points.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.closePath();if(fill){ctx.fillStyle=fill;ctx.fill();}if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=width;ctx.stroke();}};
  const start = id => {const canvas=$(id),ctx=canvas.getContext("2d");ctx.clearRect(0,0,canvas.width,canvas.height);canvas.dataset.cell=state.cell;canvas.dataset.frame=state.frame;return [ctx,canvas.width,canvas.height];};
  const timeAxis = (ctx, w, h, left = 66, right = 22, bottom = 30) => {
    for (let hr = 0; hr <= 120; hr += 24) {const x=left+hr/(N*DT)*(w-left-right);line(ctx,[x,18],[x,h-bottom]);text(ctx,`${hr} h`,x,h-9,C.muted,11,"center");}
    const x=left+state.frame/(N-1)*(w-left-right);line(ctx,[x,15],[x,h-bottom],"#edf5f4aa",1.2);return x;
  };
  const hits = {};
  function image() {
    const [ctx,w,h]=start("cell-image"),field=$("image-mode").value==="field",o=observation(),p=o?[o.x,o.y]:xy(state.cell,state.frame);
    const centre=field?[300,190]:p,scale=field?Math.min(w/600,h/380):4.2;
    const toScreen=q=>[(q[0]-centre[0])*scale+w/2,(q[1]-centre[1])*scale+h/2];
    const raster=document.createElement("canvas");raster.width=field?300:132;raster.height=field?190:76;
    const rc=raster.getContext("2d"),pixels=rc.createImageData(raster.width,raster.height),contrast=Number($("contrast").value);
    const nearby=cells.filter(c=>{const a=c.observations[state.frame];return a&&(field||Math.hypot(a.x-p[0],a.y-p[1])<115);});
    for(let y=0;y<raster.height;y++)for(let x=0;x<raster.width;x++){
      const q=[centre[0]+(x/raster.width*w-w/2)/scale,centre[1]+(y/raster.height*h-h/2)/scale];
      let signal=6+hash(x,y,state.frame)*18;
      for(const c of nearby){const a=c.observations[state.frame],dx=q[0]-a.x,dy=q[1]-a.y,d=Math.hypot(dx,dy),ang=Math.atan2(dy,dx);
        const j=Math.round((ang+TAU)%TAU/TAU*64)%64,r=Math.hypot(a.outline[j][0]-a.x,a.outline[j][1]-a.y);
        if(d<r)signal+=a.mean*(1.2+1.7*Math.exp(-d*d/100))*(.75+hash(x+state.frame,y,c.id)*.5);
        else if(d<r+2)signal+=a.mean*.3;
      }
      const v=clamp(Math.round(signal*contrast),0,255),i=(y*raster.width+x)*4;pixels.data[i]=v;pixels.data[i+1]=v;pixels.data[i+2]=v;pixels.data[i+3]=255;
    }
    rc.putImageData(pixels,0,0);ctx.imageSmoothingEnabled=false;ctx.drawImage(raster,0,0,w,h);
    hits.image=[];
    const layer=$("outline-layer").value;
    for(const c of nearby){const a=c.observations[state.frame],sel=c.id===state.cell;
      if(layer==="base"||layer==="both")polygon(ctx,a.base.map(toScreen),sel?C.gold:"#b9976355",null,sel?1.8:1);
      if(layer==="refined"||layer==="both")polygon(ctx,a.outline.map(toScreen),sel?C.cyan:"#89cbb985",null,sel?2:1);
      const q=toScreen([a.x,a.y]);hits.image.push({id:c.id,x:q[0],y:q[1],r:field?22:80});
      if(field)text(ctx,c.id,q[0],q[1]-24,sel?"white":C.muted,11,"center");
    }
    if(o&&$("show-change").checked){const before=selected().observations[state.frame-1];if(before){ctx.setLineDash([4,4]);polygon(ctx,before.outline.map(toScreen),C.gold,null,1.4);ctx.setLineDash([]);}}
    if($("show-trails").checked&&o){const count=clamp(Number($("trail-length").value)||12,1,48);for(let f=Math.max(1,state.frame-count);f<=state.frame;f++){const a=selected().observations[f-1],b=selected().observations[f];if(!a||!b)continue;ctx.globalAlpha=.12+.75*(f-(state.frame-count))/count;line(ctx,toScreen([a.x,a.y]),toScreen([b.x,b.y]),C.cyan,2);}ctx.globalAlpha=1;}
    if(!o){ctx.fillStyle="#071520c9";ctx.fillRect(0,0,w,h);text(ctx,"No observation for this cell at this frame",w/2,h/2,C.gold,16,"center");}
    text(ctx,`Cell ${state.cell} · ${fmt(state.frame*DT,1)} h`,15,25,"white",12);
    const scalePixels=field?50:10,screenLength=scalePixels*scale;line(ctx,[w-screenLength-20,h-27],[w-20,h-27],"white",2);text(ctx,`${scalePixels} px`,w-20,h-36,"white",10,"right");
    $("image-caption").textContent=o?"Simulated signal · fixed contrast · no display smoothing":"Missing observations remain blank";
  }
  function history(){
    const [ctx,w,h]=start("outline-history"),cell=selected(),moving=$("history-position").value==="moving",local=$("history-window").value!=="all";
    const from=local?Math.max(0,state.frame-24):0,to=local?Math.min(N-1,state.frame+24):N-1;
    const project=(point,f)=>{
      const o=cell.observations[f],ref=moving?xy(state.cell,state.frame):[o.x,o.y];
      const x=(point[0]-ref[0])*2.8,z=(point[1]-ref[1])*2.8,y=(f-(from+to)/2)/(to-from||1)*260;
      const rx=x*Math.cos(state.yaw)-z*Math.sin(state.yaw),rz=x*Math.sin(state.yaw)+z*Math.cos(state.yaw);
      return [w/2+rx*state.zoom,h/2+(y*Math.cos(state.pitch)+rz*Math.sin(state.pitch))*state.zoom];
    };
    ctx.globalAlpha=.55;for(let f=from;f<=to;f+=local?1:3){const o=cell.observations[f];if(!o)continue;polygon(ctx,o.outline.map(p=>project(p,f)),phaseColour(f/N*24),null,.75);}ctx.globalAlpha=1;
    const o=observation();if(o)polygon(ctx,o.outline.map(p=>project(p,state.frame)),"#ffffff", "#79e3d217",2.8);
    text(ctx,`${fmt(from*DT,1)} h`,20,27,C.muted,12);text(ctx,`${fmt(to*DT,1)} h`,20,h-18,C.muted,12);text(ctx,"TIME",w-20,h/2,C.muted,10,"right");
    $("history-caption").textContent=`Cell ${cell.id} · ${local?"48-frame window":"240-frame history"}${o?"":" · selected observation missing"}`;
  }
  function trace(){
    const [ctx,w,h]=start("trace"),s=settings("trace"),extra=$("trace-secondary").value;
    const lanes=[{s,colour:C.cyan}];if(extra!=="none"&&extra!==s.metric)lanes.push({s:{metric:extra,aggregation:"total",smoothing:1},colour:C.gold});
    const left=66,right=22,laneHeight=(h-36)/lanes.length;
    lanes.forEach((lane,l)=>{
      const series=dataSeries(selected(),lane.s),[lo,hi]=bounds(series),top=l*laneHeight+20,bottom=(l+1)*laneHeight-12;
      line(ctx,[left,bottom],[w-right,bottom]);text(ctx,fmt(hi,1),left-9,top+5,C.muted,10,"right");text(ctx,fmt(lo,1),left-9,bottom,C.muted,10,"right");
      ctx.beginPath();let connected=false;series.forEach((v,f)=>{if(v==null){connected=false;return;}const x=left+f/(N-1)*(w-left-right),y=bottom-(v-lo)/(hi-lo)*(bottom-top-12);if(connected)ctx.lineTo(x,y);else ctx.moveTo(x,y);connected=true;});ctx.strokeStyle=lane.colour;ctx.lineWidth=1.8;ctx.stroke();
      text(ctx,`${metricLabel(lane.s)} · ${unit(lane.s)}`,left+8,top,lane.colour,11);
      const v=series[state.frame];if(v!=null){const x=left+state.frame/(N-1)*(w-left-right),y=bottom-(v-lo)/(hi-lo)*(bottom-top-12);circle(ctx,x,y,4,lane.colour,"#0b1c28");text(ctx,fmt(v,1),w-right-4,top,lane.colour,12,"right");}
    });
    timeAxis(ctx,w,h);$("trace-caption").textContent=`Cell ${state.cell} · ${metricLabel(s)} · ${s.smoothing===1?"raw":`centred ${s.smoothing}-frame mean`} · gaps stay blank`;
  }
  let barcodeRows=cells;
  function barcode(){
    const [ctx,w,h]=start("barcode"),s=settings("barcode"),left=47,right=20,top=10,bottom=29,rowH=(h-top-bottom)/COUNT;
    barcodeRows=[...cells];const order=$("barcode-order").value;if(order==="phase")barcodeRows.sort((a,b)=>a.phase-b.phase);if(order==="group")barcodeRows.sort((a,b)=>a.group.localeCompare(b.group)||a.id-b.id);
    const all=barcodeRows.map(c=>dataSeries(c,s)),globalBounds=bounds(all.flat()),cw=(w-left-right)/N;
    barcodeRows.forEach((cell,row)=>{const [lo,hi]=$("barcode-scale").value==="cell"?bounds(all[row]):globalBounds;all[row].forEach((v,f)=>{if(v==null)return;ctx.fillStyle=ramp((v-lo)/(hi-lo));ctx.fillRect(left+f*cw,top+row*rowH,Math.ceil(cw),rowH-.5);});text(ctx,cell.id,left-12,top+(row+.68)*rowH,cell.id===state.cell?"white":C.muted,10,"right");if(cell.id===state.cell){ctx.strokeStyle="white";ctx.lineWidth=1.4;ctx.strokeRect(left,top+row*rowH,w-left-right,rowH);}});
    for(let hr=0;hr<=120;hr+=24)text(ctx,`${hr} h`,left+hr/(N*DT)*(w-left-right),h-9,C.muted,11,"center");
    const x=left+(state.frame+.5)*cw;line(ctx,[x,top],[x,h-bottom],"#ffffffbb",1.2);
    $("barcode-caption").textContent=`${metricLabel(s)} · ${s.smoothing===1?"raw":`${s.smoothing}-frame mean`} · ${$("barcode-scale").value==="cell"?"colour scaled separately per cell":"one colour scale for all cells"}`;
  }
  function events(){
    const [ctx,w,h]=start("events"),left=105,right=22,mode=$("event-mode").value,cell=selected(),cw=(w-left-right)/N;
    const rows=[{label:"Activity",y:50},{label:"Observation",y:108},{label:"Intervention",y:166}];
    rows.forEach(r=>{text(ctx,r.label,12,r.y+5,C.muted,12);line(ctx,[left,r.y],[w-right,r.y]);});
    for(let f=0;f<N;f++){const o=cell.observations[f];if(mode!=="observations"&&o){ctx.fillStyle=o.speed>3?C.gold:"#386e76";ctx.fillRect(left+f*cw,38,cw+0.2,24);}if(mode!=="states"){ctx.fillStyle=o?"#347c78":"#d0a466";ctx.fillRect(left+f*cw,96,cw+0.2,24);}}
    if(mode==="all"){const x=left+120/(N-1)*(w-left-right);line(ctx,[x,145],[x,182],C.gold,3);text(ctx,"60 h",x+6,161,C.gold,11);}
    timeAxis(ctx,w,h,left,right,25);text(ctx,"Activity: gold = faster movement",left,23,C.muted,10);
  }
  function clock(){
    const [ctx,w,h]=start("phase-clock"),x=w/2,y=h/2+2,r=77,cell=selected();hits.clock=[];
    for(let tick=0;tick<24;tick+=3){const a=tick/24*TAU-Math.PI/2;line(ctx,[x+Math.cos(a)*(r-5),y+Math.sin(a)*(r-5)],[x+Math.cos(a)*r,y+Math.sin(a)*r],C.grid,2);text(ctx,`${tick} h`,x+Math.cos(a)*(r+20),y+Math.sin(a)*(r+20)+4,C.muted,10,"center");}
    circle(ctx,x,y,r,null,C.grid);
    cells.filter(c=>$("clock-scope").value!=="group"||c.group===cell.group).forEach(c=>{const a=c.phase/24*TAU-Math.PI/2,rr=r-(c.id%3)*9,px=x+Math.cos(a)*rr,py=y+Math.sin(a)*rr;circle(ctx,px,py,c.id===cell.id?5.5:3,phaseColour(c.phase),c.id===cell.id?"white":null);hits.clock.push({id:c.id,x:px,y:py,r:14});});
    const a=(state.frame*DT%24)/24*TAU-Math.PI/2;line(ctx,[x,y],[x+Math.cos(a)*(r-19),y+Math.sin(a)*(r-19)],"#ffffff88",1.4);circle(ctx,x,y,2,"white");text(ctx,`Cell ${cell.id}`,x,y-9,"white",12,"center");text(ctx,`${fmt(cell.phase,1)} h phase`,x,y+11,C.cyan,11,"center");
  }
  function map(){
    const [ctx,w,h]=start("tissue-map"),layer=$("map-layer").value,scale=Math.min((w-65)/600,(h-42)/380),to=q=>[35+q[0]*scale,16+q[1]*scale];hits.map=[];
    for(let x=0;x<=600;x+=100)line(ctx,to([x,0]),to([x,380]));for(let y=0;y<=380;y+=100)line(ctx,to([0,y]),to([600,y]));
    const valid=cells.filter(c=>c.observations[state.frame]),s={metric:layer==="light"?"light":layer,aggregation:"total"},v=valid.map(c=>rawValue(c,state.frame,s.metric,s.aggregation)),[lo,hi]=bounds(v);
    valid.forEach((c,i)=>{const o=c.observations[state.frame],colour=layer==="phase"?phaseColour(c.phase):layer==="coverage"?"#63bdb2":ramp((v[i]-lo)/(hi-lo));const p=to([o.x,o.y]);
      if(layer==="coverage")polygon(ctx,o.outline.map(to),colour,colour+"44",1);else circle(ctx,p[0],p[1],Math.sqrt(o.area)/5,colour);
      if(c.id===state.cell)circle(ctx,p[0],p[1],17,null,"white");text(ctx,c.id,p[0]+15,p[1]+4,c.id===state.cell?"white":C.muted,10);hits.map.push({id:c.id,x:p[0],y:p[1],r:22});
    });
    text(ctx,"0 px",35,h-8,C.muted,10);text(ctx,"600 px",35+600*scale,h-8,C.muted,10,"right");
    $("map-caption").textContent=layer==="phase"?"Colour = simulated phase, 0–24 h":layer==="coverage"?"Current refined boundaries in recording coordinates":`${layer==="light"?"Total light":metrics[layer].label} · ${fmt(lo,1)}–${fmt(hi,1)} ${unit(s)}`;
  }
  function network(){
    const [ctx,w,h]=start("relationships"),threshold=Number($("neighbour-distance").value),o=observation(),valid=cells.filter(c=>c.observations[state.frame]);
    const neighbours=o?valid.filter(c=>{const a=c.observations[state.frame];return c.id!==state.cell&&Math.hypot(a.x-o.x,a.y-o.y)<=threshold;}):[];
    const visible=$("network-scope").value==="all"?valid:valid.filter(c=>c.id===state.cell||neighbours.includes(c));
    const positions=new Map();visible.forEach((c,i)=>{const a=i/visible.length*TAU-Math.PI/2;positions.set(c.id,c.id===state.cell?[w/2,h/2]:[w/2+Math.cos(a)*Math.min(w*.38,180),h/2+Math.sin(a)*h*.34]);});hits.network=[];
    let edges=0;
    visible.forEach((a,i)=>visible.slice(i+1).forEach(b=>{const p=a.observations[state.frame],q=b.observations[state.frame];if(Math.hypot(p.x-q.x,p.y-q.y)<=threshold){edges++;line(ctx,positions.get(a.id),positions.get(b.id),a.id===state.cell||b.id===state.cell?"#7fe4d39a":"#365260",1.4);}}));
    visible.forEach(c=>{const p=positions.get(c.id),sel=c.id===state.cell;circle(ctx,...p,sel?19:14,sel?"#235b57":"#183645",sel?C.cyan:"#547e8c");text(ctx,c.id,p[0],p[1]+4,sel?"white":C.ink,12,"center");hits.network.push({id:c.id,x:p[0],y:p[1],r:23});});
    if(!o)text(ctx,"Selected cell has no observation",w/2,h/2,C.gold,14,"center");
    $("distance-label").textContent=`${threshold} px`;$("network-caption").textContent=`Cell ${state.cell} · ${neighbours.length} neighbours · ${edges} displayed connections`;
  }
  const quantile=(a,q)=>{const s=[...a].sort((x,y)=>x-y),p=(s.length-1)*q;return s[Math.floor(p)]+(s[Math.ceil(p)]-s[Math.floor(p)])*(p%1);};
  function distribution(){
    const [ctx,w,h]=start("distribution"),s=settings("distribution"),current=$("distribution-time").value==="frame",grouped=$("distribution-group").value==="group",left=90,right=25,top=30,bottom=h-42;
    const samples=cells.map(c=>{const all=(current?[rawValue(c,state.frame,s.metric,s.aggregation)]:dataSeries(c,s)).filter(v=>v!=null);return {cell:c,v:all.length?avg(all):null};}).filter(a=>a.v!=null);
    const [lo,hi]=bounds(samples.map(a=>a.v)),pad=(hi-lo)*.14,y=v=>bottom-(v-lo+pad)/(hi-lo+2*pad)*(bottom-top),groups=grouped?["A","B"]:["All"],xs=groups.map((g,i)=>left+(i+.5)/groups.length*(w-left-right));hits.distribution=[];
    for(let i=0;i<4;i++){const v=lo+(hi-lo)*i/3,yy=y(v);line(ctx,[left,yy],[w-right,yy]);text(ctx,fmt(v,1),left-12,yy+4,C.muted,11,"right");}
    text(ctx,unit(s),12,20,C.muted,11);
    groups.forEach((group,i)=>{const a=samples.filter(p=>!grouped||p.cell.group===group),values=a.map(p=>p.v),x=xs[i],col=i?C.gold:C.cyan;
      if(!values.length)return;
      const q1=quantile(values,.25),q3=quantile(values,.75),median=quantile(values,.5);
      ctx.fillStyle=i?"#e8b66e19":"#79e3d219";ctx.fillRect(x-70,y(q3),140,y(q1)-y(q3));line(ctx,[x-70,y(median)],[x+70,y(median)],col,2);
      a.forEach(p=>{const xx=x+(hash(p.cell.id,13)-.5)*100,yy=y(p.v),sel=p.cell.id===state.cell;circle(ctx,xx,yy,sel?6:4,col,sel?"white":null);if(sel)text(ctx,`Cell ${p.cell.id}`,xx+11,yy-6,"white",11);hits.distribution.push({id:p.cell.id,x:xx,y:yy,r:12});});
      text(ctx,grouped?`Group ${group} · ${values.length} cells`:`${values.length} cells`,x,h-15,C.muted,12,"center");
    });
    $("distribution-caption").textContent=`${metricLabel(s)} · ${current?`selected frame at ${fmt(state.frame*DT,1)} h`:"one recording mean per cell"} · median + interquartile range`;
  }
  function motionTable(){
    $("motion-frame").textContent=`Shared selection: frame ${state.frame+1} / ${N} · ${fmt(state.frame*DT,1)} h · cell ${state.cell}`;
    $("track-rows").replaceChildren(...cells.map(c=>{const o=c.observations[state.frame],row=document.createElement("tr");row.className=c.id===state.cell?"selected":"";const button=document.createElement("button");button.textContent=`Cell ${c.id}`;button.addEventListener("click",()=>select(c.id,state.frame));const td=document.createElement("td");td.append(button);row.append(td);[c.group,o?"Observed":"Missing",o?fmt(o.x,1):"—",o?fmt(o.y,1):"—",o?fmt(o.area):"—"].forEach(v=>{const cell=document.createElement("td");cell.textContent=v;row.append(cell);});return row;}));
  }
  function render(){
    const c=selected(),o=observation();$("cell-select").value=state.cell;$("frame-slider").value=state.frame;
    $("time-label").textContent=`${fmt(state.frame*DT,1)} h · frame ${state.frame+1} / ${N}`;
    $("sidebar-cell").textContent=`Cell ${state.cell}`;$("sidebar-context").textContent=`Simulated group ${c.group} · phase ${fmt(c.phase,1)} h`;
    $("area-readout").textContent=o?`${fmt(o.area)} px²`:"Missing";$("light-readout").textContent=o?`${fmt(o.total)} counts`:"Missing";$("speed-readout").textContent=o?`${fmt(o.speed,1)} px/h`:"Missing";
    $("selection-status").textContent=o?`Cell ${state.cell} · one shared time cursor · changing a plot metric changes only that plot.`:`Cell ${state.cell} is missing at this frame. Its identity remains selected; no observation is invented.`;
    for(const prefix of ["trace","barcode","distribution"])$(prefix+"-aggregation-label").hidden=$(prefix+"-metric").value!=="light";
    if(state.workspace==="extras"){image();history();trace();barcode();events();clock();map();network();distribution();}
    if(state.workspace==="motion")motionTable();
  }
  let scheduled=false;
  function queueRender(){if(scheduled)return;scheduled=true;requestAnimationFrame(()=>{scheduled=false;render();});}
  function select(id,frame){state.cell=clamp(Math.round(id),1,COUNT);state.frame=clamp(Math.round(frame),0,N-1);queueRender();}
  function workspace(name){stop();state.workspace=name;document.querySelectorAll("[data-workspace]").forEach(b=>{const active=b.dataset.workspace===name;b.setAttribute("aria-selected",String(active));b.tabIndex=active?0:-1;});for(const key of ["auto","motion","extras"]){$("workspace-"+key).hidden=key!==name;$("sidebar-"+key).hidden=key!==name;}$("main").scrollTop=0;render();}
  let timer;
  function stop(){clearInterval(timer);state.playing=false;$("play").textContent="Play";$("play").setAttribute("aria-label","Play recording");}
  function play(){if(state.playing){stop();return;}state.playing=true;$("play").textContent="Pause";$("play").setAttribute("aria-label","Pause recording");timer=setInterval(()=>{if(state.frame===N-1){stop();return;}select(state.cell,state.frame+1);},1000/Number($("play-rate").value));}
  function coords(e,canvas){const r=canvas.getBoundingClientRect();return [(e.clientX-r.left)/r.width*canvas.width,(e.clientY-r.top)/r.height*canvas.height];}
  function selectHit(key,id){$(id).addEventListener("click",e=>{const p=coords(e,$(id)),close=(hits[key]||[]).filter(a=>Math.hypot(p[0]-a.x,p[1]-a.y)<=a.r).sort((a,b)=>Math.hypot(p[0]-a.x,p[1]-a.y)-Math.hypot(p[0]-b.x,p[1]-b.y))[0];if(close)select(close.id,state.frame);});}
  for(const c of cells){const option=document.createElement("option");option.value=c.id;option.textContent=c.id;$("cell-select").append(option);}
  document.querySelectorAll(".metric-select").forEach(s=>{for(const [key,m]of Object.entries(metrics)){const option=document.createElement("option");option.value=key;option.textContent=m.label;s.append(option);}});
  $("barcode-metric").value="light";$("distribution-metric").value="area";
  $("cell-select").addEventListener("change",e=>select(Number(e.target.value),state.frame));$("frame-slider").addEventListener("input",e=>select(state.cell,Number(e.target.value)));
  $("previous").addEventListener("click",()=>select(state.cell,state.frame-1));$("next").addEventListener("click",()=>select(state.cell,state.frame+1));$("play").addEventListener("click",play);
  $("play-rate").addEventListener("change",()=>{if(state.playing){stop();play();}});
  document.querySelectorAll("select,input").forEach(el=>{if(!["cell-select","frame-slider","play-rate"].includes(el.id)){el.addEventListener("change",queueRender);if(el.type==="range")el.addEventListener("input",queueRender);}});
  document.querySelectorAll("[data-workspace]").forEach(b=>b.addEventListener("click",()=>workspace(b.dataset.workspace)));
  document.querySelectorAll("[data-open-extras]").forEach(b=>b.addEventListener("click",()=>workspace("extras")));
  document.querySelector(".top-tabs").addEventListener("keydown",e=>{if(!["ArrowLeft","ArrowRight","Home","End"].includes(e.key))return;e.preventDefault();const tabs=[...document.querySelectorAll("[data-workspace]")],i=tabs.indexOf(document.activeElement),n=e.key==="Home"?0:e.key==="End"?2:(i+(e.key==="ArrowRight"?1:2))%3;workspace(tabs[n].dataset.workspace);tabs[n].focus();});
  $("trace").addEventListener("click",e=>{const [x]=coords(e,$("trace"));select(state.cell,(x-66)/(1280-66-22)*(N-1));});
  $("events").addEventListener("click",e=>{const [x]=coords(e,$("events"));select(state.cell,(x-105)/(900-105-22)*(N-1));});
  $("barcode").addEventListener("click",e=>{const [x,y]=coords(e,$("barcode"));const row=clamp(Math.floor((y-10)/(370-10-29)*COUNT),0,COUNT-1),frame=clamp(Math.floor((x-47)/(1280-47-20)*N),0,N-1);select(barcodeRows[row].id,frame);});
  for(const id of ["trace","events","barcode","cell-image","tissue-map","relationships","distribution","phase-clock"]){$(id).addEventListener("keydown",e=>{if(e.key==="ArrowLeft"||e.key==="ArrowRight"){e.preventDefault();select(state.cell,state.frame+(e.key==="ArrowRight"?1:-1));}if(e.key==="ArrowUp"||e.key==="ArrowDown"){e.preventDefault();select(state.cell+(e.key==="ArrowDown"?1:-1),state.frame);}});}
  selectHit("image","cell-image");selectHit("map","tissue-map");selectHit("network","relationships");selectHit("distribution","distribution");selectHit("clock","phase-clock");
  let drag=null;const sculpture=$("outline-history");sculpture.addEventListener("pointerdown",e=>{drag=[e.clientX,e.clientY];sculpture.setPointerCapture(e.pointerId);});sculpture.addEventListener("pointermove",e=>{if(!drag)return;state.yaw+=(e.clientX-drag[0])*.008;state.pitch=clamp(state.pitch+(e.clientY-drag[1])*.005,-1.3,1.3);drag=[e.clientX,e.clientY];queueRender();});sculpture.addEventListener("pointerup",()=>{drag=null;});sculpture.addEventListener("pointercancel",()=>{drag=null;});sculpture.addEventListener("wheel",e=>{e.preventDefault();state.zoom=clamp(state.zoom*(e.deltaY<0?1.08:.92),.4,2);queueRender();},{passive:false});
  sculpture.addEventListener("keydown",e=>{if(["ArrowLeft","ArrowRight","ArrowUp","ArrowDown"].includes(e.key)){e.preventDefault();state.yaw+=e.key==="ArrowRight"?.08:e.key==="ArrowLeft"?-.08:0;state.pitch=clamp(state.pitch+(e.key==="ArrowDown"?.05:e.key==="ArrowUp"?-.05:0),-1.3,1.3);queueRender();}});
  $("reset-rotation").addEventListener("click",()=>{state.yaw=-.62;state.pitch=.42;state.zoom=1;queueRender();});
  $("open-pipelines").addEventListener("click",()=>$("pipeline-dialog").showModal());$("close-pipelines").addEventListener("click",()=>$("pipeline-dialog").close());
  document.querySelectorAll("#sidebar-extras .jump-links a").forEach(link=>link.addEventListener("click",e=>{e.preventDefault();const target=document.querySelector(link.getAttribute("href"));$("main").scrollTo({top:$("main").scrollTop+target.getBoundingClientRect().top-$("main").getBoundingClientRect().top-70,behavior:matchMedia("(prefers-reduced-motion: reduce)").matches?"instant":"smooth"});}));
  const observer=new IntersectionObserver(entries=>{for(const entry of entries)if(entry.isIntersecting){document.querySelectorAll("#sidebar-extras .jump-links a").forEach(a=>a.classList.toggle("active",a.getAttribute("href")==="#"+entry.target.id));}},{root:$("main"),rootMargin:"-10% 0px -65% 0px"});document.querySelectorAll(".plot-group").forEach(el=>observer.observe(el));
  document.addEventListener("visibilitychange",()=>{if(document.hidden)stop();});
  window.demo=Object.freeze({getState:()=>({...state}),getDiagnostics:()=>({cells:COUNT,frames:N,intervalHours:DT,plots:[...document.querySelectorAll("[data-plot-type]")].map(p=>p.dataset.plotType),immutableObservations:Object.isFrozen(cells)&&cells.every(c=>Object.isFrozen(c.observations)),selectionMissing:observation()==null,selection:{cell:state.cell,frame:state.frame},settings:{trace:settings("trace"),barcode:settings("barcode"),distribution:settings("distribution")},hitTargets:JSON.parse(JSON.stringify(hits))}),getObservation:(id,f)=>cells[id-1]?.observations[f]||null,getValue:(id,f,metric,aggregation="total",smoothing=1)=>value(cells[id-1],f,metric,aggregation,smoothing)});
  render();document.documentElement.dataset.demoReady="true";
})();
