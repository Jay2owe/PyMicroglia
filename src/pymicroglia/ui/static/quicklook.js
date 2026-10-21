"use strict";
(() => {
  const ns="http://www.w3.org/2000/svg";
  const colours=["#75d9ca","#e9af78","#a99bed","#78b5eb","#e089ac","#bbd67b"];
  const titles={
    rhythms:["Cell light output over time","Measured signal per cell"],
    tissue:["Territory across the recording","Measured coverage per cell"],
    cell_explorer:["Cells across time","Measured value per cell"],
    shape:["Shape and surveillance over time","Measured morphology per cell"],
    movement:["Movement across time","Measured movement per cell"],
    states:["Cell traits across time","Measured trait per cell"],
    relationships:["Neighbourhood across time","Measured relationship per cell"],
    interventions:["Response across time","Measured value per cell"],
    methods:["Measurement evidence over time","Observed and inferred pixels per cell"]
  };
  function label(value){return String(value).replaceAll("_"," ").replace(/\b\w/g,c=>c.toUpperCase());}
  function fmt(value){if(!Number.isFinite(value))return "–";const abs=Math.abs(value);return abs>=10000?value.toExponential(1):abs>=100?value.toFixed(0):abs>=10?value.toFixed(1):value.toFixed(2);}
  function svgElement(tag,attrs={}){const el=document.createElementNS(ns,tag);for(const [key,value] of Object.entries(attrs))el.setAttribute(key,String(value));return el;}
  function text(x,y,value,attrs={}){const node=svgElement("text",{x,y,...attrs});node.textContent=value;return node;}
  function select(labelText,values,chosen){
    const wrap=document.createElement("label");wrap.className="quick-setting";wrap.append(document.createTextNode(labelText));
    const input=document.createElement("select");for(const [value,name] of values){const option=document.createElement("option");option.value=String(value);option.textContent=name;input.append(option);}input.value=String(chosen);wrap.append(input);return [wrap,input];
  }
  function smooth(values,span){
    if(span===1)return values;
    const radius=Math.floor(span/2);
    return values.map((value,index)=>{
      if(value===null||!Number.isFinite(value))return null;
      let sum=0,count=0;for(let i=Math.max(0,index-radius);i<=Math.min(values.length-1,index+radius);i++){
        if(Number.isFinite(values[i])){sum+=values[i];count++;}
      }
      return count?sum/count:null;
    });
  }
  function frame(titleText,subtitle){
    const article=document.createElement("article");article.className="quick-chart";
    const head=document.createElement("div");head.className="quick-chart-head";
    const words=document.createElement("div");const title=document.createElement("h3");title.textContent=titleText;
    const note=document.createElement("p");note.textContent=subtitle;words.append(title,note);head.append(words);article.append(head);
    return {article,head,note};
  }
  function path(points,x,y){
    let d="",active=false;for(const [time,value] of points){
      if(!Number.isFinite(time)||!Number.isFinite(value)){active=false;continue;}
      d+=`${active?"L":"M"}${x(time).toFixed(2)},${y(value).toFixed(2)}`;active=true;
    }return d;
  }
  function axes(svg,bounds,xmin,xmax,ymin,ymax,yTitle){
    const {left,top,width,height}=bounds;
    const x=value=>left+(value-xmin)/(xmax-xmin||1)*width;
    const y=value=>top+height-(value-ymin)/(ymax-ymin||1)*height;
    svg.append(svgElement("rect",{x:left,y:top,width,height,fill:"#0a1c2a",rx:7}));
    for(let i=0;i<=4;i++){
      const xx=left+i*width/4,yy=top+i*height/4;
      svg.append(svgElement("path",{d:`M${xx} ${top}V${top+height} M${left} ${yy}H${left+width}`,stroke:"#244558","stroke-width":1}));
      svg.append(text(xx,top+height+23,fmt(xmin+i*(xmax-xmin)/4),{"text-anchor":"middle",class:"quick-axis"}));
      svg.append(text(left-10,yy+4,fmt(ymax-i*(ymax-ymin)/4),{"text-anchor":"end",class:"quick-axis"}));
    }
    svg.append(text(left+width/2,top+height+48,"Hours from recording start",{"text-anchor":"middle",class:"quick-axis-label"}));
    svg.append(text(20,top+height/2,yTitle,{"text-anchor":"middle",transform:`rotate(-90 20 ${top+height/2})`,class:"quick-axis-label"}));
    return {x,y};
  }
  function lineChart(host,traces,metric,config){
    host.replaceChildren();const svg=svgElement("svg",{viewBox:"0 0 940 340",role:"img","aria-label":`${label(metric)} over time`});
    svg.classList.add("quick-svg");host.append(svg);
    const selected=traces.slice(0,Number(config.cells));
    let series=selected.map(item=>({name:`${item.stem} · ${item.identity}`,hours:item.hours,
      values:smooth(item.values[metric]||[],Number(config.smoothing))}));
    if(config.mode==="mean"){
      const buckets=new Map();for(const item of series)item.hours.forEach((hour,index)=>{
        const value=item.values[index];if(!Number.isFinite(hour)||!Number.isFinite(value))return;
        const key=Math.round(hour*1000),entry=buckets.get(key)||{hour,sum:0,n:0};entry.sum+=value;entry.n++;buckets.set(key,entry);
      });
      const points=[...buckets.values()].sort((a,b)=>a.hour-b.hour);
      series=[{name:"Mean of displayed cells",hours:points.map(x=>x.hour),values:points.map(x=>x.sum/x.n)}];
    }
    const all=series.flatMap(item=>item.hours.map((hour,index)=>[hour,item.values[index]]).filter(point=>Number.isFinite(point[0])&&Number.isFinite(point[1])));
    if(!all.length){host.append(Object.assign(document.createElement("p"),{className:"quick-empty",textContent:"No finite measured values are available for this view."}));svg.remove();return;}
    const xmin=Math.min(...all.map(point=>point[0])),xmax=Math.max(...all.map(point=>point[0]));
    let ymin=Math.min(...all.map(point=>point[1])),ymax=Math.max(...all.map(point=>point[1]));
    if(ymin===ymax){ymin-=1;ymax+=1;}const pad=(ymax-ymin)*.07;ymin-=pad;ymax+=pad;
    const scales=axes(svg,{left:83,top:16,width:821,height:252},xmin,xmax,ymin,ymax,label(metric));
    series.forEach((item,index)=>{
      const points=item.hours.map((hour,k)=>[hour,item.values[k]]);
      const d=path(points,scales.x,scales.y);if(!d)return;
      const line=svgElement("path",{d,fill:"none",stroke:colours[index%colours.length],
        "stroke-width":config.mode==="mean"?3.2:1.8,"stroke-opacity":config.mode==="mean"?1:.7});
      line.append(svgElement("title"));line.querySelector("title").textContent=item.name;svg.append(line);
    });
  }
  function histogram(host,traces,metric,bins){
    host.replaceChildren();const means=traces.map(item=>(item.values[metric]||[]).filter(Number.isFinite))
      .filter(values=>values.length).map(values=>values.reduce((a,b)=>a+b,0)/values.length);
    if(!means.length){host.textContent="No measured values available.";return;}
    const svg=svgElement("svg",{viewBox:"0 0 620 310",role:"img","aria-label":`Distribution of cell means for ${label(metric)}`});svg.classList.add("quick-svg");host.append(svg);
    const min=Math.min(...means),max=Math.max(...means),count=Number(bins),range=max-min||1;
    const bars=Array(count).fill(0);for(const value of means)bars[Math.min(count-1,Math.floor((value-min)/range*count))]++;
    const high=Math.max(...bars,1),left=50,top=18,width=545,height=225;
    svg.append(svgElement("rect",{x:left,y:top,width,height,fill:"#0a1c2a",rx:7}));
    for(let i=0;i<=4;i++){const yy=top+height*(1-i/4);svg.append(svgElement("path",{d:`M${left} ${yy}H${left+width}`,stroke:"#244558"}));svg.append(text(left-7,yy+4,fmt(high*i/4),{"text-anchor":"end",class:"quick-axis"}));}
    bars.forEach((value,index)=>{const x=left+(index+.1)*width/count,w=.8*width/count,h=value/high*height;
      const bar=svgElement("rect",{x,y:top+height-h,width:w,height:h,rx:3,fill:"#75d9ca"});bar.append(svgElement("title"));bar.querySelector("title").textContent=`${value} cells`;svg.append(bar);});
    svg.append(text(left,top+height+24,fmt(min),{class:"quick-axis"}),text(left+width,top+height+24,fmt(max),{"text-anchor":"end",class:"quick-axis"}));
    svg.append(text(left+width/2,top+height+48,`Mean ${label(metric)} per cell`,{"text-anchor":"middle",class:"quick-axis-label"}));
  }
  function rhythmPlot(host,rows,metric){
    host.replaceChildren();const metrics=[...new Set(rows.map(item=>item.metric).filter(Boolean))];
    const chosen=metrics.includes(metric)?metric:metrics[0];
    const selected=rows.filter(item=>(!chosen||item.metric===chosen)&&Number.isFinite(item.period_hours)).slice(0,18);
    if(!selected.length){host.textContent="No saved period estimates are available for these cells.";return chosen;}
    const svg=svgElement("svg",{viewBox:`0 0 620 ${Math.max(150,selected.length*25+65)}`,role:"img","aria-label":"Saved rhythm periods by cell"});svg.classList.add("quick-svg");host.append(svg);
    const min=Math.min(...selected.map(item=>item.period_hours)),max=Math.max(...selected.map(item=>item.period_hours));
    const low=Math.max(0,min-(max-min||2)*.15),high=max+(max-min||2)*.15;
    selected.forEach((item,index)=>{
      const y=26+index*25,x=125+(item.period_hours-low)/(high-low||1)*455;
      svg.append(text(10,y+4,`Cell ${item.identity}`,{class:"quick-axis"}));
      svg.append(svgElement("path",{d:`M125 ${y}H580`,stroke:"#244558"}));
      const circle=svgElement("circle",{cx:x,cy:y,r:6,fill:item.status==="rhythmic"?"#aa97ec":"#9cb4be"});
      circle.append(svgElement("title"));circle.querySelector("title").textContent=`${item.status}: ${fmt(item.period_hours)} h`;svg.append(circle);
    });
    const bottom=selected.length*25+40;
    svg.append(text(125,bottom,fmt(low),{class:"quick-axis"}),text(580,bottom,fmt(high),{"text-anchor":"end",class:"quick-axis"}));
    svg.append(text(350,bottom+20,"Saved period (hours)",{"text-anchor":"middle",class:"quick-axis-label"}));
    return chosen;
  }
  function render(container,page,data,message=""){
    const section=document.createElement("section");section.className="quick-section";container.append(section);
    const heading=document.createElement("div");heading.className="section-heading";
    const h=document.createElement("h2");h.textContent="At a glance";heading.append(h);
    const note=document.createElement("small");note.textContent=data?`${data.displayed_cells} of ${data.total_cells} cells · from saved measurements`:"Measured plots appear here when a run is selected";heading.append(note);section.append(heading);
    if(!data||data.message||!data.traces?.length){
      const empty=document.createElement("div");empty.className="quick-empty-panel";
      const title=document.createElement("strong");title.textContent=data?.message||message||"Choose a saved measurement run to see its plots.";
      const sub=document.createElement("p");sub.textContent="These charts use saved measurements and do not rerun tracking or rhythm fitting.";empty.append(title,sub);section.append(empty);return;
    }
    const metrics=(data.pages?.[page.key]||[]).filter(item=>data.metrics.includes(item));
    if(!metrics.length){const empty=document.createElement("div");empty.className="quick-empty-panel";empty.textContent="This run has no measured columns for this page.";section.append(empty);return;}
    let saved={};try{saved=JSON.parse(localStorage.getItem(`pm-quick-${page.key}`)||"{}");}catch{/* Ignore old settings. */}
    const config={metric:metrics.includes(saved.metric)?saved.metric:metrics[0],smoothing:[1,3,7].includes(Number(saved.smoothing))?Number(saved.smoothing):(page.key==="rhythms"?3:1),
      cells:[6,12,24,32].includes(Number(saved.cells))?Number(saved.cells):12,mode:saved.mode==="mean"?"mean":"cells",bins:[8,12,20].includes(Number(saved.bins))?Number(saved.bins):12};
    const grid=document.createElement("div");grid.className="quick-grid";section.append(grid);
    const main=frame(titles[page.key]?.[0]||"Measurements over time",titles[page.key]?.[1]||"Saved per-cell measurements");main.article.classList.add("quick-wide");grid.append(main.article);
    const settingsButton=document.createElement("button");settingsButton.className="button secondary quick-settings-button";settingsButton.textContent="Settings";settingsButton.setAttribute("aria-expanded","false");main.head.append(settingsButton);
    const controls=document.createElement("div");controls.className="quick-controls";controls.hidden=true;main.article.append(controls);
    const metric=select("Measurement",metrics.map(item=>[item,label(item)]),config.metric);
    const smoothing=select("Display mean",[[1,"No smoothing"],[3,"3 frames"],[7,"7 frames"]],config.smoothing);
    const count=select("Cells shown",[[6,"6"],[12,"12"],[24,"24"],[32,"32"]],config.cells);
    const mode=select("Lines",[["cells","Individual cells"],["mean","Mean of displayed cells"]],config.mode);
    controls.append(metric[0],smoothing[0],count[0],mode[0]);settingsButton.onclick=()=>{controls.hidden=!controls.hidden;settingsButton.setAttribute("aria-expanded",String(!controls.hidden));};
    const chart=document.createElement("div");chart.className="quick-plot";main.article.append(chart);
    const legend=document.createElement("p");legend.className="quick-caption";main.article.append(legend);
    const distribution=frame("Between-cell distribution","Mean measured value for each displayed cell");grid.append(distribution.article);
    const binButton=document.createElement("button");binButton.className="button ghost";binButton.textContent="Settings";distribution.head.append(binButton);
    const binWrap=select("Histogram bins",[[8,"8 bins"],[12,"12 bins"],[20,"20 bins"]],config.bins);binWrap[0].classList.add("quick-bin-settings");binWrap[0].hidden=true;distribution.article.append(binWrap[0]);binButton.onclick=()=>binWrap[0].hidden=!binWrap[0].hidden;
    const hist=document.createElement("div");hist.className="quick-plot";distribution.article.append(hist);
    let rhythm=null;if(page.key==="rhythms"&&data.rhythms?.length){rhythm=frame("Saved rhythm estimates","These are stored period results; changing chart settings does not refit them.");grid.append(rhythm.article);rhythm.plot=document.createElement("div");rhythm.plot.className="quick-plot";rhythm.article.append(rhythm.plot);}
    function paint(){config.metric=metric[1].value;config.smoothing=Number(smoothing[1].value);config.cells=Number(count[1].value);config.mode=mode[1].value;config.bins=Number(binWrap[1].value);
      localStorage.setItem(`pm-quick-${page.key}`,JSON.stringify(config));
      const visible=data.traces.slice(0,config.cells);lineChart(chart,visible,config.metric,config);histogram(hist,visible,config.metric,config.bins);
      legend.textContent=`${visible.length} cells · ${label(config.metric)} · ${config.smoothing===1?"raw measured points":`${config.smoothing}-frame display mean`} · no new fit`;
      if(rhythm){const fitted=rhythmPlot(rhythm.plot,data.rhythms,config.metric);rhythm.note.textContent=`Saved ${label(fitted||"rhythm")} period results; no new fit.`;}
    }
    for(const input of [metric[1],smoothing[1],count[1],mode[1],binWrap[1]])input.onchange=paint;
    paint();
  }
  window.PyMicrogliaQuicklook={render};
})();
