"use strict";
(() => {
  const C=window.PyMicrogliaExplorationCore;
  const cyan="#72ded3",gold="#efc77b",muted="#a2b3c2",white="#eef4fa";
  const escape=s=>String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const heat=(v,accent=cyan)=>{const rgb=[1,3,5].map(i=>parseInt(accent.slice(i,i+2),16));return `rgb(${[30,53,74].map((lo,i)=>Math.round(lo+(rgb[i]-lo)*Math.max(0,Math.min(1,v)))).join(",")})`;};
  const fmt=x=>C.finite(x)?Number(x.toPrecision(4)).toString():"—";
  const SESSION_KEY="pm-exploration-session-v1";
  const RHYTHM_FIELDS=[["rhythm-method","fit_method"],["rhythm-min","period_min_hours",true],["rhythm-max","period_max_hours",true],["rhythm-detrend","detrend"],["rhythm-correction","multiple_testing"],["rhythm-observations","min_observations",true],["rhythm-cycles","min_cycles",true],["rhythm-alpha","rhythmic_alpha",true]];
  function readSession(){try{const value=JSON.parse(localStorage.getItem(SESSION_KEY)||"null");return value?.version===1?value:null;}catch{return null;}}
  const plotHelp={height:"Drawing height for this view.",fontSize:"Plot label size for this view.",lineWidth:"Line thickness for this view.",background:"Plot background; source pixels stay unchanged.",accent:"Main data colour for this view.",undo:"Undo the last appearance change in this plot; up to 20 changes are remembered."};
  const HELP={
    "xp-trace-fit":["Dashed rhythm fit","Overlay the saved fitted model, with its baseline restored to the trace units. Applies to selected, nearby and all-cell traces; no rhythm analysis runs."],
    "xp-trace-grid-fit":["Dashed rhythm fit","Overlay the same saved fitted model across the cell grid and selected-cell traces. Missing or untested models stay blank."],
    "xp-rhythm-measurements-toggle":["Rhythm measurements","Tick any combination of saved measurements to test for all cells in the current recording. Total light uses the accepted three-frame mean."],
    "xp-rhythm-select-all":["Select every measurement","Tick every saved measurement available in this recording; analysis starts only when you press Run rhythm analysis."],
    "xp-rhythm-clear":["Clear measurement selection","Untick every measurement. Choose at least one before running rhythm analysis."],
    "xp-run-rhythms":["Run rhythm analysis","Run the accepted cell trace analysis as a background job, save its evidence and fill the linked views. Masks, identities and measured values stay unchanged."],
    "xp-rhythm-settings-toggle":["Rhythm settings","Choose period estimation, significance testing, search limits and baseline removal before running."],
    "xp-rhythm-visible":["Show rhythm results","Show or hide each cell's tested rhythm result beneath its linked plots."],
    "xp-rhythm-method":["Period estimator","Choose the existing Circadian Workbench method that estimates the period."],
    "xp-rhythm-recipe":["Rhythm calculation method","Swap the period estimator and its rhythmicity test together. Fourier components includes its required robust linear baseline. Other methods keep your detrending choice. Custom combinations remain available in Advanced."],
    "xp-rhythm-use-shown":["Use shown settings","Restore the calculation settings and tested measurements that produced the displayed results. This starts no analysis."],
    "xp-rhythm-test":["Rhythm significance test","Test whether the signal is rhythmic separately from estimating its period. The default is the accepted Fourier component test."],
    "xp-rhythm-min":["Minimum candidate period","Shortest period searched, in hours."],
    "xp-rhythm-max":["Maximum candidate period","Longest period searched, in hours."],
    "xp-rhythm-observations":["Minimum measured samples","A cell needs this many usable measured samples before it can be tested."],
    "xp-rhythm-cycles":["Minimum observed cycles","Minimum duration expressed in the fitted cell's cycles; shorter recordings retain limited period support."],
    "xp-rhythm-alpha":["Significance threshold","The probability threshold used to mark a cell's tested measurement rhythmic."],
    "xp-rhythm-detrend":["Baseline removal","Remove the slowly changing baseline using the selected existing method before fitting."],
    "xp-rhythm-correction":["Multiple test correction","Use the package's correction within each measurement. Fourier component testing uses the accepted uncorrected component probabilities."],
    "xp-recording-select":["Recording","Choose any loaded recording; combined comparisons stay loaded."],
    "xp-cell-select":["Cell","Choose a saved Motion cell identity; changing this never recomputes tracks."],
    "xp-previous":["Previous frame","Move one frame back. Arrow Left also moves the selected time."],
    "xp-play":["Playback","Play or pause the saved frames. Playback does not run analysis."],
    "xp-next":["Next frame","Move one frame forward. Arrow Right also moves the selected time."],
    "xp-frame-slider":["Time","Move the shared selected time across all nine views."],
    "xp-play-rate":["Playback rate","Set how many saved frames playback shows each second."],
    "xp-image-mode":["Image view","Switch between the selected cell crop and the whole saved field."],
    "xp-image-lut":["Image colour lookup table","Apply the selected colours to the main image and comparison cell images. The choice is remembered with this recording."],
    "xp-outline-layer":["Outline","Show saved masks, base masks, both or neither without changing measurements."],
    "xp-show-trails":["Motion trail","Overlay previous saved positions; off by default."],
    "xp-trail-length":["Trail frames","Choose how many earlier frames the motion trail includes."],
    "xp-contrast":["Display contrast","Change only the displayed brightness of the source image."],
    "xp-reset-rotation":["Reset view","Return the rotating outline history to its starting orientation."],
    "xp-history-position":["Outline position","Centre each shape or retain its saved movement across the field."],
    "xp-history-window":["Outline window","Show the full saved history or a short window around the selected time."],
    "xp-trace-metric":["Trace metric","Choose the quantity on the selected cell's one measurement trace."],
    "xp-trace-aggregation":["Trace light aggregation","Use total signal within the mask or mean signal per pixel."],
    "xp-trace-smoothing":["Trace time mean","Average consecutive observed frames for display; gaps remain missing."],
    "xp-trace-grid-metric":["All cell trace metric","Use the same measurement as the linked cell traces above."],
    "xp-trace-grid-aggregation":["All cell light aggregation","Choose total signal or mean signal per pixel for all linked traces."],
    "xp-trace-grid-smoothing":["All cell time mean","Average consecutive observations for display; missing frames remain gaps."],
    "xp-trace-grid-columns":["Trace grid columns","Choose the number of traces across the grid; Auto follows the window width."],
    "xp-trace-grid-scale":["Trace grid scale","Compare each cell's own range or use the same vertical scale for every cell."],
    "xp-trace-layout":["Measurement layout","Stack measurements in aligned lanes with their own units, or overlay them with a colour for each measurement."],
    "xp-trace-grid-measurement-layout":["Measurement layout","Overlay or stack all ticked measurements within each cell's trace. This is shared with the selected and nearby cell plots."],
    "xp-trace-grid-measurement-scale":["Measurement overlay scale","Display overlaid measurements in original units or scale each measured trace to 0–1; the same conversion applies to its saved fit."],
    "xp-trace-overlay-scale":["Measurement overlay scale","Each trace 0–1 scales its observed minimum and maximum for display and applies exactly the same scale to its saved fit. Original units uses one shared axis."],
    "xp-trace-grid-overlay-toggle":["Cell trace layout","Keep separate cells in the grid or overlay the cells selected in Compare cells. Choose Columns 1 for a vertical list with an aligned time axis."],
    "xp-trace-grid-overlay-scale":["Cell overlay scale","Use original measurement units or display each trace on a 0–1 scale. Saved measurements and fits stay unchanged."],
    "xp-trace-cells-nearby":["Compare selected and nearby cells","Select the current cell and the two comparison cells shown above."],
    "xp-trace-cells-all":["Compare all cells","Select every identity in this recording for comparison."],
    "xp-trace-cells-clear":["Clear compared cells","Clear the comparison selection; choose cells again using the checklist."],
    "xp-trace-rhythm-colours":["Colour by significance","Stacked traces use red for a significant saved rhythm, grey otherwise. Each measurement uses its own test. Overlays use distinct series colours and asterisks for significance. Untick to use your chosen data colour in stacked traces."],
    "xp-barcode-metric":["Barcode metric","Choose the quantity in the one cell-by-time matrix."],
    "xp-barcode-aggregation":["Barcode light aggregation","Use total signal within each mask or mean signal per pixel."],
    "xp-barcode-smoothing":["Barcode time mean","Average consecutive observed frames for display when exact frames are available."],
    "xp-barcode-scale":["Barcode scale","Use each cell's range or one shared range across the recording."],
    "xp-barcode-rhythm-colours":["Barcode rhythm colours","Keep measurement brightness, using red for cells with significant saved rhythms and grey for other cells. Each barcode measurement uses its own test. Untick to use the chosen plot colour."],
    "xp-barcode-order":["Barcode ordering engine","Use the existing matrix engine to place similar displayed traces together: principal component gradient or spectral continuum. Off restores cell identity order. Its internal smoothing affects row order only; measurements, gaps, cell identities and time stay unchanged."],
    "xp-event-mode":["Timeline view","Switch the one timeline between all saved cell tracks (Gantt), selected-cell events and observation gaps. Missing observations remain dashed gaps within the same identity."],
    "xp-timeline-order":["Track order","Order Gantt rows by first observation, identity or saved lifecycle events. This changes display order only."],
    "xp-tectonics-metric":["Tissue period measurement","Use this measurement's saved significance and period results for the tissue period map. No rhythm test runs when switching maps."],
    "xp-clock-scope":["Phase clock cells","Show every saved fitted phase or only the selected group."],
    "xp-clock-metric":["Phase clock metric","Choose the measurement for saved phases or the automatic fixed-cycle preview."],
    "xp-map-layer":["Tissue tectonics layer","Choose first coverage, cumulative occupancy, distinct cells, speed, significant period, recorded contact separations or cell positions. The full-field maps use the native tissue analysis and the saved masks."],
    "xp-network-scope":["Relationship scope","Show all saved links or only links to the selected cell."],
    "xp-link-source":["Saved links","Choose saved contacts or contacts computed from the selected frame's masks."],
    "xp-distribution-metric":["Comparison metric","Choose the measurement used to compare cells or groups."],
    "xp-distribution-aggregation":["Comparison light aggregation","Use total signal within masks or mean signal per pixel."],
    "xp-distribution-time":["Comparison time","Use each cell's displayed recording mean or the exact selected frame."],
    "xp-distribution-group":["Comparison grouping","Compare conditions, subjects or one population."],
    "xp-cell-image":["Cell image","Click a cell in the whole field to select it. Display pixels stay sharp."],
    "xp-cell-change":["Mask change","Cyan pixels were gained and amber pixels were lost since the preceding saved frame. White is the current outline."],
    "xp-peer-one":["First comparison cell","Choose a saved cell to compare at the same frame and time cursor."],
    "xp-peer-two":["Second comparison cell","Choose another saved cell to compare at the same frame and time cursor."],
    "xp-outline-history":["Outline history","Drag to rotate, scroll to zoom or use arrow keys. Height represents time."],
    "xp-trace":["Measurement trace","Click or use arrow keys to choose a time. Blank segments are missing observations."],
    "xp-barcode":["Recording barcode","Click a row and time to select a saved cell and frame. Blank cells mean missing observations."],
    "xp-events":["Event timeline","Click a time to inspect it. Only saved events, windows and observation gaps appear."],
    "xp-phase-clock":["Phase clock","Click a phase marker to select its cell. The caption identifies saved fits or a fixed-cycle preview."],
    "xp-tissue-map":["Tissue map","Click a saved cell position to select that cell."],
    "xp-relationships":["Cell relationships","Click a node to select its cell. Links come from the named saved result or the current-frame contact preview."],
    "xp-distribution":["Cell and group comparison","Click a point to select its cell. Bars show median and quartiles; no significance test is implied."]
  };
  class Explorer {
    constructor(host,options) {this.host=host;this.options=options;this.data=null;this.state={identity:null,frame:0};this.token=0;this.cellToken=0;this.hits={};this.rotation={yaw:-.55,pitch:.6,zoom:1};this.imageCache=new Map();this.cellImageCache=new Map();this.changeCache=new Map();this.peerRows=new Map();this.peerExact=new Map();this.peerStripUnavailable=new Set();this.peerEpoch=0;this.cellImageEpoch=0;this.imageCacheBytes=0;this.imageQueue=[];this.imagePending=new Set();this.imageWorkers=0;this.imageEpoch=0;this.frameCache=new Map();this.timer=null;this.pending=null;this.session=readSession();this.lutTables=new Map();this.lutCache=new Map();this.lutCacheBytes=0;this.styles=this.session?.styles||{};this.styleHistory={};this.ready=this.setup();}
    $(id){return this.host.querySelector(`#xp-${id}`);}
    async setup(){this.host.innerHTML=await (await fetch("/assets/exploration.html",{cache:"no-store"})).text();this.comparisons=new window.PyMicrogliaComparisonPlots(this);this.bind();this.filters=new window.PyMicrogliaCellFilters(this);this.outlineFilters=window.PyMicrogliaCellFilters.mountGrey(this);this.rhythmAnalyses=new window.PyMicrogliaRhythmRows(this);this.groupComparison=new window.PyMicrogliaGroupComparison(this);if(this.options.onAudit){const button=document.createElement("button");button.type="button";button.textContent="Audit this trace";button.title="Compare detrending and rhythm methods for the selected cell and measurement";button.onclick=()=>{if(this.data)this.options.onAudit({identity:this.state.identity,metric:this.rhythmAnalyses.traceMetric(this.rhythmAnalyses.metricFor("trace"))});};this.plotHeader(this.$("trace-panel")).actions.prepend(button);}this.resetTraceGrid();await this.loadLuts();this.lutsReady=true;if(this.pending)await this.setData(this.pending);else this.empty("Choose a saved measurement run to fill the linked views.");}
    empty(message){this.data=null;this.sourceData=null;this.filters?.reset(null);this.outlineFilters?.reset(null);this.greyOutlineIds=null;this.pause();clearTimeout(this.rhythmJobTimer);this.rhythmSchema=null;this.$("run-rhythms").disabled=true;this.$("rhythm-progress").hidden=true;this.$("rhythm-status").textContent="";this.host.querySelectorAll(".xp-rhythm-note").forEach(note=>note.hidden=true);this.resetTraceGrid();this.host.querySelectorAll("canvas").forEach(c=>{this.canvas(c.id.slice(3));this.blank(c,message);});this.$("selection-status").textContent=message;}
    barcodeHeight(id="barcode"){const width=this.$(id)?.getBoundingClientRect().width||600,rows=this.data?.cells.length,columns=this.data?.sample_indices.length;return rows&&columns?Math.round(43+Math.max(1,width-60)/columns/1.3*rows):125;}
    style(id){
      const baseId=this.comparisons?.base(id)||id,grid=id==="trace-grid",trace=grid||id==="trace"||id.startsWith("peer-")&&id.endsWith("-trace"),paper=trace||["barcode","events","tissue-map","distribution"].includes(baseId);
      const inherited={...(trace?this.styles.trace:null)};if(grid)delete inherited.height;
      const s={...inherited,...this.styles[id]},base=grid?120:baseId==="barcode"?this.barcodeHeight(id):Number(this.$(id)?.dataset.baseHeight||this.$(id)?.height||260),minimum=grid?92:baseId==="barcode"?80:id==="outline-history"?140:id==="trace"?120:id==="selected-mini"||id.startsWith("peer-")&&id.endsWith("-image")?60:160;
      return {height:Math.max(minimum,Math.min(baseId==="barcode"?2048:720,Number(s.height)||base)),fontSize:Math.max(10,Math.min(20,Number(s.fontSize)||13)),lineWidth:Math.max(.5,Math.min(6,Number(s.lineWidth)||(trace?1.25:2))),background:/^#[0-9a-fA-F]{6}$/.test(s.background||"")?s.background:paper?"#ffffff":"#101923",accent:/^#[0-9a-fA-F]{6}$/.test(s.accent||"")?s.accent:trace?this.data?.trace_style?.other||muted:paper?"#167b78":cyan,yMin:s.yMin??"",yMax:s.yMax??""};
    }
    saveSession(){if(!this.data||this.restoring)return;const controls={};this.host.querySelectorAll("select[id^='xp-'],input[id^='xp-']").forEach(el=>{if(!["xp-recording-select","xp-cell-select","xp-frame-slider"].includes(el.id)&&!el.classList.contains("xp-plot-control"))controls[el.id]=el.type==="checkbox"?el.checked:el.value;});const folds={};this.host.querySelectorAll("details[id]").forEach(el=>folds[el.id]=el.open);this.session={version:1,barcodeOrderingVersion:1,contextViewsVersion:1,traceGridControlsVersion:1,run:this.data.run,stem:this.data.stem,identity:this.state.identity,frame:this.state.frame,controls,rhythmMetrics:this.selectedRhythmMetrics(),traceMeasurements:this.tracePicks("measurements"),traceCells:this.tracePicks("cells"),measurementTabs:{...this.measurementTabs},comparisonPlots:this.comparisons.save(),comparisonRun:this.groupComparison?.run||this.data.run,groupComparison:this.groupComparison?.save(),cellFilters:this.filters?.save(),outlineFilters:this.outlineFilters?.save(),folds,styles:this.styles};try{localStorage.setItem(SESSION_KEY,JSON.stringify(this.session));}catch{/* Optional browser storage. */}}
    restoreControls(){const session=this.session;if(session?.run!==this.data.run||session?.stem!==this.data.stem)return;this.restoring=true;for(const [id,value] of Object.entries(session.controls||{})){const el=document.getElementById(id);if(!el||!this.host.contains(el))continue;if(id==="xp-barcode-order"&&value==="identity"&&!session.barcodeOrderingVersion)continue;if(!session.contextViewsVersion&&(id==="xp-map-layer"&&value==="identity"||id==="xp-event-mode"&&value==="all"))continue;if(el.type==="checkbox")el.checked=Boolean(value);else if(el.tagName!=="SELECT"||[...el.options].some(o=>o.value===value&&!o.disabled))el.value=value;}for(const [id,open] of Object.entries(session.folds||{})){const el=document.getElementById(id);if(el&&this.host.contains(el)&&el.tagName==="DETAILS")el.open=Boolean(open);}if(session.controls?.["xp-trace-grid-layout"]==="stack"){this.$("trace-grid-layout").value="grid";this.$("trace-grid-columns").value="1";}this.restoring=false;}
    canvas(id){
      const c=this.$(id),baseId=this.comparisons?.base(id)||id;if(!c.dataset.baseHeight)c.dataset.baseHeight=c.height;
      const s={...this.style(id),...(c.dataset.traceHeight?{height:+c.dataset.traceHeight}:{})},crisp=["barcode","phase-clock","relationships","events","tissue-map"].includes(baseId);
      if(crisp){
        const width=Math.max(1,Math.round(c.getBoundingClientRect().width));
        const aspect=this.comparisons?.views.get(id)?.aspect;
        const height=id==="events"&&this.val("event-mode")==="gantt"?44+(this.timeline?.cells?.length||0)*Math.ceil(Math.min(12,s.fontSize)+1):baseId==="tissue-map"&&this.comparisons.value("map-layer",id).startsWith("tectonics:")&&aspect&&!this.styles[id]?.height?Math.round(width/aspect+34):s.height;
        if(baseId==="barcode")c.parentElement.style.maxHeight=`${s.height}px`;
        if(id==="events")c.parentElement.style.maxHeight=`${s.height}px`;
        c.style.height=`${height}px`;
        const ratio=window.devicePixelRatio||1;
        c.dataset.logicalWidth=width;c.dataset.logicalHeight=height;
        const bw=Math.round(width*ratio),bh=Math.round(height*ratio);
        if(c.width!==bw)c.width=bw;if(c.height!==bh)c.height=bh;
        c.getContext("2d").setTransform(ratio,0,0,ratio,0,0);
      }else if(id==="trace"||id.startsWith("peer-")&&id.endsWith("-trace")){
        if(c.dataset.traceHeight||this.styles.trace?.height)c.style.height=`${s.height}px`;
        const rect=c.getBoundingClientRect(),width=Math.max(1,Math.round(rect.width)),height=Math.max(1,Math.round(rect.height)),ratio=window.devicePixelRatio||1;
        c.dataset.logicalWidth=width;c.dataset.logicalHeight=height;
        if(c.width!==Math.round(width*ratio))c.width=Math.round(width*ratio);if(c.height!==Math.round(height*ratio))c.height=Math.round(height*ratio);
        c.getContext("2d").setTransform(ratio,0,0,ratio,0,0);
      }else if(id==="selected-mini"||id.startsWith("peer-")||["cell-image","cell-change","outline-history"].includes(id)){
        if(this.styles[id]?.height){if(id==="trace"||id==="outline-history")c.style.height=`${s.height}px`;else if(id==="cell-image"||id==="cell-change")c.style.maxWidth=`${s.height}px`;}
        const rect=c.getBoundingClientRect(),width=Math.max(1,Math.round(rect.width)),height=Math.max(1,Math.round(rect.height));
        if(c.width!==width)c.width=width;if(c.height!==height)c.height=height;
      }else if(c.height!==s.height)c.height=s.height;
      const w=Number(c.dataset.logicalWidth)||c.width,h=Number(c.dataset.logicalHeight)||c.height,ctx=c.getContext("2d");
      ctx.clearRect(0,0,w,h);ctx.fillStyle=s.background;ctx.fillRect(0,0,w,h);ctx.font=`${s.fontSize}px system-ui`;ctx.fillStyle=muted;
      return [ctx,w,h];
    }
    blank(c,message){const s=this.style(c.id.slice(3)),ctx=c.getContext("2d"),w=Number(c.dataset.logicalWidth)||c.width,h=Number(c.dataset.logicalHeight)||c.height;ctx.clearRect(0,0,w,h);ctx.fillStyle=s.background;ctx.fillRect(0,0,w,h);ctx.fillStyle=muted;ctx.font=`${s.fontSize}px system-ui`;ctx.textAlign="center";ctx.fillText(message.slice(0,110),w/2,h/2,Math.max(10,w-24));ctx.textAlign="left";}
    stroke(ctx,width){ctx.lineWidth=width*this.style(ctx.canvas.id.slice(3)).lineWidth/2;}
    accent(ctx){return this.style(ctx.canvas.id.slice(3)).accent;}
    rangeFor(id,values){const [baseLo,baseHi]=C.range(values),s=this.styles[id]||{};const lo=s.yMin!==""&&s.yMin!=null&&Number.isFinite(Number(s.yMin))?Number(s.yMin):baseLo,hi=s.yMax!==""&&s.yMax!=null&&Number.isFinite(Number(s.yMax))?Number(s.yMax):baseHi;return hi>lo?[lo,hi]:[baseLo,baseHi];}
    plotTarget(panel){const canvas=panel.querySelector("[data-primary-canvas]")||panel.querySelector("canvas");return {canvas,id:panel.dataset.styleKey||canvas.id.slice(3)};}
    plotHeader(panel){
      const heading=panel.querySelector(".panel-heading");let main=heading.querySelector(":scope > .xp-panel-main"),actions=heading.querySelector(":scope > .xp-panel-actions");
      if(!main){main=document.createElement("div");main.className="xp-panel-main";actions=document.createElement("div");actions.className="xp-panel-actions";main.append(...heading.childNodes);heading.append(main,actions);}
      for(const element of [...heading.children])if(element!==main&&element!==actions)main.append(element);
      return {main,actions};
    }
    plotEditors(panels=null){
      for(const panel of panels||this.host.querySelectorAll("[data-plot-type]")){
        const {canvas,id}=this.plotTarget(panel),header=this.plotHeader(panel);
        const button=document.createElement("button"),editor=document.createElement("details");
        button.type="button";button.className="xp-edit-plot";button.textContent="Edit plot";
        button.title=`Edit ${panel.querySelector("h3").textContent} appearance`;
        button.setAttribute("aria-controls",`xp-edit-${id}`);button.id=`xp-style-button-${id}`;
        editor.id=`xp-edit-${id}`;editor.className="settings xp-plot-settings";
        editor.innerHTML=`<summary>Plot appearance</summary><div class="controls">
          <label title="${plotHelp.height}">Height <input class="xp-plot-control" data-style="height" type="number" min="160" max="720" step="20"></label>
          <label title="${plotHelp.fontSize}">Label size <input class="xp-plot-control" data-style="fontSize" type="number" min="10" max="20"></label>
          <label title="${plotHelp.lineWidth}">Line width <input class="xp-plot-control" data-style="lineWidth" type="number" min="0.5" max="6" step="0.5"></label>
          <label title="${plotHelp.accent}">Data colour <input class="xp-plot-control" data-style="accent" type="color"></label>
          <label title="${plotHelp.background}">Background <input class="xp-plot-control" data-style="background" type="color"></label>
          <button type="button" class="xp-undo-plot" title="${plotHelp.undo}" disabled>Undo appearance</button>
        </div>`;
        if(["trace","trace-grid"].includes(id)){editor.querySelector('input[data-style="height"]').min=id==="trace-grid"?"92":"120";editor.querySelector('input[data-style="lineWidth"]').step="0.25";}
        if(this.comparisons?.base(id)==="barcode"){const heightInput=editor.querySelector('input[data-style="height"]');heightInput.min="80";heightInput.max="2048";heightInput.closest("label").title="Matrix height; the default makes each time-by-cell tile slightly wider than tall.";}
        if(["trace","trace-grid","distribution"].includes(this.comparisons?.base(id)||id))editor.querySelector(".controls").insertAdjacentHTML("beforeend",`<label title="Lower visible measurement value; blank uses the data range.">Y minimum <input class="xp-plot-control" data-style="yMin" type="number" step="any"></label><label title="Upper visible measurement value; blank uses the data range.">Y maximum <input class="xp-plot-control" data-style="yMax" type="number" step="any"></label>`);
        header.actions.append(button);if(id==="trace")panel.querySelector(".xp-cell-cards").before(editor);else if(id==="trace-grid")panel.querySelector(".settings").before(editor);else if(["barcode","events"].includes(this.comparisons?.base(id)||id))canvas.parentElement.before(editor);else canvas.before(editor);
        button.onclick=()=>{editor.open=!editor.open;this.saveSession();};
        editor.addEventListener("toggle",()=>this.saveSession());
        const undo=editor.querySelector(".xp-undo-plot");undo.id=`xp-undo-${id}`;
        undo.onclick=()=>{const previous=this.styleHistory[id]?.pop();if(!previous)return;this.styles[id]=previous;for(const input of editor.querySelectorAll("input"))input.value=this.style(id)[input.dataset.style];undo.disabled=!this.styleHistory[id].length;this.draw();this.saveSession();};
        for(const input of editor.querySelectorAll("input")){
          input.id=`xp-style-${id}-${input.dataset.style}`;input.title=plotHelp[input.dataset.style]||"Set a visible axis bound; blank uses the data range.";input.value=this.style(id)[input.dataset.style];
          input.onchange=()=>{const history=this.styleHistory[id]||(this.styleHistory[id]=[]);history.push({...this.styles[id]});if(history.length>20)history.shift();this.styles[id]={...this.styles[id],[input.dataset.style]:input.value};const actual=this.style(id)[input.dataset.style];input.value=actual;this.styles[id][input.dataset.style]=actual;undo.disabled=false;this.draw();this.saveSession();};
        }
      }
      (panels?panels.flatMap(panel=>[...panel.querySelectorAll("details:not([id])")]):[...this.host.querySelectorAll("details:not([id])")]).forEach((el,i)=>{el.id=panels?`${panels[0].id}-fold-${i}`:`xp-fold-${i}`;el.addEventListener("toggle",()=>this.saveSession());});
      for(const panel of panels||this.host.querySelectorAll("[data-plot-type]")){
        for(const [details,label] of [[panel.querySelector(".settings:not(.xp-plot-settings)"),"Settings"],[panel.querySelector(".xp-export"),"Export"]]){
          if(!details)continue;
          const button=document.createElement("button");button.type="button";button.className="xp-panel-action";button.textContent=label;
          button.setAttribute("aria-controls",details.id);button.setAttribute("aria-expanded",String(details.open));
          button.onclick=()=>{details.open=!details.open;this.saveSession();};
          details.addEventListener("toggle",()=>button.setAttribute("aria-expanded",String(details.open)));
          this.plotHeader(panel).actions.append(button);
        }
      }
      if(!panels)this.host.querySelector("#xp-follow .group-heading").append(this.$("selection-status"));
    }
    attachHelp(){
      const rows=[];
      for(const [id,[label,description]] of Object.entries(HELP)){
        const el=document.getElementById(id);if(!el||!this.host.contains(el))continue;
        el.title=description;rows.push([label,description]);
      }
      for(const panel of this.host.querySelectorAll("[data-plot-type]")){
        const {id}=this.plotTarget(panel),label=panel.querySelector("h3").textContent;
        const button=panel.querySelector(".xp-edit-plot"),undo=panel.querySelector(".xp-undo-plot");
        button.id=`xp-style-button-${id}`;undo.id=`xp-undo-${id}`;
        rows.push([`Edit ${label}`,button.title],[`Undo ${label} appearance`,plotHelp.undo]);
        for(const input of panel.querySelectorAll(".xp-plot-control")){
          input.id=`xp-style-${id}-${input.dataset.style}`;
          input.title=["trace","trace-grid"].includes(id)&&input.dataset.style==="accent"?"Trace colour when Colour by significance is switched off in Settings.":plotHelp[input.dataset.style]||"Set a visible axis bound; blank uses the data range.";
          rows.push([`${label}: ${input.closest("label").textContent.trim()}`,input.title]);
        }
      }
      const details=document.createElement("details");details.id="xp-control-help";details.className="xp-control-help";
      details.innerHTML=`<summary>Control help</summary><dl>${rows.map(([label,description])=>`<dt>${escape(label)}</dt><dd>${escape(description)}</dd>`).join("")}</dl>`;
      this.host.querySelector(".xp-provenance-details").before(details);
      details.addEventListener("toggle",()=>this.saveSession());
      this.$("provenance-details")?.addEventListener("toggle",()=>this.saveSession());
      this.host.querySelectorAll(".hint").forEach(el=>el.remove());
    }
    measurementCatalogue(){return this.data.measurement_catalogue||{groups:[{key:"signal",label:"Saved measurements"}],metrics:this.data.metrics.map(m=>({...m,group:"signal",available:true}))};}
    measurementPicker(id,config){return new window.PyMicrogliaMeasurementPicker(this.$(id),{catalogue:this.measurementCatalogue(),active:this.measurementTabs?.[id],onTab:key=>{this.measurementTabs[id]=key;this.saveSession();},...config});}
    updateMetricPickers(){this.metricPickers??=new Map();for(const id of ["trace-metric","trace-grid-metric","barcode-metric","distribution-metric","clock-metric","tectonics-metric"])this.updateMetricPicker(id);}
    updateMetricPicker(id){
      const select=this.$(id);if(!select)return;this.metricPickers??=new Map();let entry=this.metricPickers.get(id);
      if(!entry){
        const label=select.closest("label"),control=document.createElement("div"),details=document.createElement("details"),summary=document.createElement("summary"),menu=document.createElement("div");
        control.className="xp-metric-control";const caption=document.createElement("span");caption.textContent=label.firstChild.textContent.trim();control.append(caption);
        details.className="xp-measurement-select";details.id=`xp-${id}-picker`;summary.title=select.title||select.getAttribute("aria-label")||"Choose a measurement";
        menu.id=`xp-${id}-browser`;menu.className="xp-measurement-menu";details.append(summary,menu);label.replaceWith(control);control.append(select,details);select.hidden=true;
        entry={select,details,summary,menu};this.metricPickers.set(id,entry);
        window.PyMicrogliaMeasurementPicker.popup(details,menu);
        select.addEventListener("change",()=>this.syncMetricPickers());details.addEventListener("toggle",()=>this.saveSession());
        details.addEventListener("keydown",event=>{if(event.key==="Escape"){details.open=false;summary.focus();}});
      }
      const choices=[...select.options].filter(option=>!option.disabled).map(option=>({key:option.value,label:option.textContent}));
      entry.picker=this.measurementPicker(id+"-browser",{mode:"single",selected:[select.value],choices,onChange:values=>{if(!values.length)return;select.value=values[0];entry.details.open=false;select.dispatchEvent(new Event("change",{bubbles:true}));}});
      entry.summary.textContent=select.selectedOptions[0]?.textContent||"No saved measurements";entry.value=select.value;
    }
    syncMetricPickers(){for(const entry of this.metricPickers?.values()||[]){if(entry.value===entry.select.value)continue;entry.value=entry.select.value;entry.picker.setSelected([entry.select.value]);entry.summary.textContent=entry.select.selectedOptions[0]?.textContent||"No saved measurements";}}
    val(id){return this.$(id).value;}
    setOptions(id,rows,preferred){const el=this.$(id),old=preferred??el.value;el.innerHTML=rows.map(([key,title])=>`<option value="${escape(key)}">${escape(title)}</option>`).join("");if(rows.some(r=>r[0]===old))el.value=old;if(this.metricPickers?.has(id))this.updateMetricPicker(id);}
    syncRecordingPicker(){
      if(!this.data)return;const rows=new Map(),key=row=>JSON.stringify([row.run,row.stem]),comparison=this.groupComparison;
      for(const row of this.data.recordings)rows.set(key({run:this.data.run,stem:row.stem}),{run:this.data.run,stem:row.stem});
      if(comparison?.design?.recordings.some(row=>row.run===this.data.run&&row.stem===this.data.stem))for(const row of comparison.design.recordings)rows.set(key(row),row);
      const sources=[...rows.values()],duplicates=stem=>sources.filter(row=>row.stem===stem).length>1;
      this.setOptions('recording-select',sources.map(row=>[key(row),duplicates(row.stem)?`${row.stem} · ${row.id?.slice(0,6)||row.run.split(/[\\/]/).at(-1)}`:row.stem]),key(this.data));
      for(const option of this.$('recording-select').options)option.title=JSON.parse(option.value)[0];
    }
    async switchRecording(source,{identity}={}){
      if(!source||!this.data)return false;
      if(source.run===this.data.run&&source.stem===this.data.stem){if(identity!=null)this.select({identity});return true;}
      const token=this.recordingToken=(this.recordingToken||0)+1,origin=this.data,comparison=this.groupComparison;this.token++;this.cancelMedia();this.pause();this.$('selection-status').textContent='Loading recording…';
      try{
        if(comparison?.design){comparison.design=comparison.save();const snapshot=JSON.stringify(comparison.design),pending=!!comparison.saveTimer;comparison.checkpoint();if(pending||comparison.saveTimer){clearTimeout(comparison.saveTimer);comparison.saveTimer=null;comparison.persist(comparison.epoch,comparison.run,comparison.design,snapshot);}}
        if(token!==this.recordingToken||this.data!==origin)return false;
        const data=await this.options.api('/api/exploration?'+new URLSearchParams({run:source.run,stem:source.stem,overview:false}));
        if(token!==this.recordingToken||this.data!==origin)return false;
        await this.setData(data);if(identity!=null)this.select({identity});this.options.onRecording?.(data);this.saveSession();return true;
      }catch(error){if(token===this.recordingToken){this.syncRecordingPicker();this.$('selection-status').textContent=error.message;}return false;}
    }
    async setData(data){if(!this.$("cell-select")||!this.lutsReady){this.pending=data;return this.ready;}this.pending=null;this.token++;this.cellToken++;this.cellRequestKey=null;this.overviewController?.abort();this.overviewSource=null;this.cancelMedia();this.cellImageBytes=0;this.pause();this.sourceData=data;this.data=data?{...data}:null;this.mediaBusy=!!data;this.cohortIds=null;this.greyOutlineIds=null;this.phaseCache=new Map();this.contactCache=new Map();this.barcodeSelectionKey=null;this.phaseRequest=0;this.phaseAutomatic=!data?.results?.rhythms?.rows?.length;this.previewContacts=!data?.results?.relations?.sources?.length;clearTimeout(this.contactDebounce);this.exact=null;this.frameData=null;this.frameValues=null;this.history=null;this.image=null;this.imageMeta=null;this.imageError="";this.historyError="";this.imageEpoch++;this.cellImageEpoch++;this.peerEpoch++;this.imageCache.clear();this.clearLutCache();this.cellImageCache.clear();this.changeCache.clear();this.peerRows.clear();this.peerExact.clear();this.peerStripUnavailable.clear();this.imageCacheBytes=0;this.imageSampleBytes=0;this.imagePrefetchExpanded=false;this.imageQueue=[];this.imagePending.clear();this.imageWorkers=0;this.frameCache.clear();this.fieldSize=null;
      if(!data){this.empty(this.options.message||"Choose a saved measurement run.");return;}
      if(data.trace_style){this.host.style.setProperty("--xp-rhythm-significant",data.trace_style.rhythmic);this.host.style.setProperty("--xp-rhythm-other",data.trace_style.other);}
      if(this.$("style-trace-accent"))this.$("style-trace-accent").value=this.style("trace").accent;
      const saved=this.session?.run===data.run&&this.session?.stem===data.stem?this.session:null;
      this.measurementTabs={...(saved?.measurementTabs||{})};
      this.tissueCache=new Map();this.tissueAspect=null;this.timeline=null;this.timelineKey=null;this.timelineEpoch=(this.timelineEpoch||0)+1;this.timelineError="";
      this.state=C.selection(data,this.state,{identity:saved?.identity??this.state.identity,frame:saved?.frame??0});
      this.syncRecordingPicker();
      this.setOptions("cell-select",data.identities.map(i=>[String(i),`Cell ${i}`]),String(this.state.identity));
      for(const slot of ["peer-one","peer-two"])this.setOptions(slot,data.identities.map(i=>[String(i),`Cell ${i}`]));
      const hasLight=data.metrics.some(m=>m.key==="integrated_density"||m.key==="signal_mean");
      const options=[...(hasLight?[["light","Reporter light"]]:[]),...data.metrics.filter(m=>!hasLight||!["integrated_density","signal_mean","integrated_density_mean3"].includes(m.key)).map(m=>[m.key,m.label])];
      for(const prefix of ["trace","trace-grid","barcode","distribution"]){this.setOptions(prefix+"-metric",options);const agg=this.$(prefix+"-aggregation");for(const o of agg.options)o.disabled=!data.metrics.some(m=>m.key===(o.value==="total"?"integrated_density":"signal_mean"));if(agg.selectedOptions[0].disabled)agg.value=Array.from(agg.options).find(o=>!o.disabled)?.value||"total";}
      const measuredDisplay=data.measurement_display;
      if(!saved&&measuredDisplay?.comparison_metrics){
        for(const prefix of ["trace","trace-grid","barcode","distribution"]){const select=this.$(prefix+"-metric");if([...select.options].some(o=>o.value===measuredDisplay.preferred_metric))select.value=measuredDisplay.preferred_metric;}
        this.$("outline-layer").value="measurement";this.$("trace-smoothing").value="1";this.$("trace-grid-smoothing").value="1";
        this.$("trace-layout").value="overlay";
        this.$("trace-overlay-scale").value="native";
      }
      this.setTracePicks(saved||{traceMeasurements:measuredDisplay?.comparison_metrics||[]});
      let measurementLegend=this.$("measurement-legend");
      if(measuredDisplay?.comparison_metrics&&!measurementLegend){
        measurementLegend=document.createElement("p");measurementLegend.id="xp-measurement-legend";measurementLegend.className="hint";
        measurementLegend.innerHTML='<strong style="color:#efc77b">Gold: original outline</strong> · White: expanded measurement outline · <strong style="color:#75a9ed">Blue: sampled local background</strong>';
        this.$("cell-image").closest(".xp-image-pair").after(measurementLegend);
        this.$("outline-layer").addEventListener("change",()=>{measurementLegend.hidden=this.val("outline-layer")!=="measurement";});
      }
      if(measurementLegend)measurementLegend.hidden=!measuredDisplay?.comparison_metrics||this.val("outline-layer")!=="measurement";
      const phaseOptions=(data.results?.rhythms?.metrics||[]).map(k=>["phase:"+k,"Saved phase · "+(data.metrics.find(m=>m.key===k)?.label||k)]);
      const tectonics=[["first_coverage","First coverage"],["cumulative_occupancy","Cumulative occupancy"],["unique_cells","Distinct cells"],["speed","Cell speed"],["significant_period","Significant cell periods"],["splitting_events","Contact separations"]];
      this.setOptions("map-layer",[...tectonics.map(([key,label])=>["tectonics:"+key,"Tectonics · "+label]),["identity","Cell positions · identities"],...phaseOptions,...options],"tectonics:first_coverage");
      this.setOptions("tectonics-metric",(data.results?.rhythms?.metrics||[]).map(k=>[k,data.metrics.find(m=>m.key===k)?.label||k]));
      this.setOptions("event-mode",[["gantt","All cell tracks (Gantt)"],["all","Selected cell events"],["observations","Observation gaps"],["states","Saved state events"]],"gantt");
      this.setOptions("distribution-group",[["condition","Condition"],["recording","Recording"],["subject","Subject"],["all","One population"],["rhythmicity","Completed rhythm analysis"]]);
      this.barcodeOrderCache=new Map();this.barcodeOrderStatus="";
      this.setOptions("barcode-order",[...(data.barcode_ordering?.engines||[]).map(e=>[e.key,e.label]),["identity","Off — cell identity"],...phaseOptions,["group","Condition"]],data.barcode_ordering?.default||"identity");
      const savedMetrics=data.results?.rhythms?.metrics||[],phaseMetrics=savedMetrics.length?savedMetrics:data.metrics.map(m=>m.key);
      this.setOptions("clock-metric",phaseMetrics.map(k=>[k,data.metrics.find(m=>m.key===k)?.label||k]),savedMetrics[0]||["integrated_density_mean3","integrated_density","signal_mean"].find(k=>phaseMetrics.includes(k))||phaseMetrics[0]);
      this.setOptions("link-source",(data.results?.relations?.sources||[]).map(s=>[s.key,s.title]));
      this.$("frame-slider").max=data.frames.length-1;
      for(const o of this.$("barcode-smoothing").options)o.disabled=+o.value!==1&&data.preview.stride!==1;
      if(data.preview.stride!==1)this.$("barcode-smoothing").value="1";
      this.$("provenance").textContent=`${data.preview.message} Read from ${data.source_table}. Display settings do not change saved masks, identities or measurements.`;
      this.rhythmSchema=null;this.rhythmRunning=false;this.$("run-rhythms").disabled=true;this.$("rhythm-status").textContent="";this.$("rhythm-progress").hidden=true;
      this.setRhythmMetrics(data.metrics,saved?.rhythmMetrics??data.results?.rhythms?.metrics??[]);
      this.resetTraceGrid();this.updateMetricPickers();this.comparisons.reset(saved?.comparisonPlots);this.restoreControls();if(saved?.traceGridControlsVersion!==1)this.$("trace-grid-settings").open=true;this.syncTraceGridControls();this.syncMetricPickers();this.select({identity:this.state.identity,frame:this.state.frame});this.loadRhythmOptions(saved);await this.groupComparison.load(data);await this.outlineFilters?.reset(saved?.outlineFilters);
    }
    applyOutlineFilter(result){
      this.greyOutlineIds=result?new Set(result.rows.filter(row=>!row.included).map(row=>row.identity)):null;
      this.greyOutlineColour=result?.grey_colour||"#aaaaaa";
      if(this.data?.identities.length){this.drawImage();this.drawChange();this.drawCellCards();this.drawHistory();}
    }
    outlineColour(identity,original){return this.greyOutlineIds?.has(identity)?this.greyOutlineColour:original;}
    inCohort(identity){return !this.cohortIds||this.cohortIds.has(identity);}
    async loadOverview(){
      const source=this.sourceData;if(!source?.preview.pending||this.overviewSource===source)return;
      this.overviewController?.abort();const controller=this.overviewController=new AbortController();this.overviewSource=source;
      try{const overview=await this.options.api(this.url('overview'),{signal:controller.signal});
        if(source!==this.sourceData||controller.signal.aborted)return;
        source.cells=overview.cells;source.preview={...source.preview};delete source.preview.pending;
        if(!this.cohortIds){this.data.cells=source.cells;this.data.preview=source.preview;}else delete this.data.preview.pending;
        this.drawSelection();this.scheduleSecondary();
      }catch(error){if(source===this.sourceData&&error.name!=='AbortError'){this.overviewSource=null;this.blank(this.$('barcode'),error.message);}}
    }
    applyCohort(result){
      if(!this.sourceData||!this.data)return;
      const source=this.sourceData,selected=this.state.identity;
      this.cohortIds=result?new Set(result.identities):null;
      this.data.identities=result?result.identities.slice():source.identities.slice();
      this.data.cells=result?result.overview_cells:source.cells;
      this.data.preview=result?{...source.preview,shown_cells:this.data.cells.length,total_cells:result.included,message:`Filtered overview: ${this.data.cells.length} / ${result.included} retained cells; every ${source.preview.stride} frame(s). Original identities preserved.`}:source.preview;
      const rows=this.data.identities.map(i=>[String(i),`Cell ${i}`]);
      this.setOptions("cell-select",rows,String(selected));
      for(const slot of ["peer-one","peer-two"])this.setOptions(slot,rows,this.val(slot));
      this.$("trace-cells-list").querySelectorAll("input").forEach(input=>{input.closest("label").hidden=!this.inCohort(+input.value);});
      this.resetTraceGrid();this.timeline=null;this.timelineKey=null;++this.timelineEpoch;this.barcodeSelectionKey=null;
      this.hits={};this.$("cell-select").disabled=!rows.length;
      if(!rows.length){this.pause();++this.cellToken;++this.imageEpoch;++this.cellImageEpoch;++this.peerEpoch;this.imageQueue=[];this.cellRequestKey=null;this.draw();}
      else{this.choosePeers();this.requestPeers();this.select({identity:this.inCohort(selected)?selected:this.data.identities[0]});}
      this.options.onCohort?.({run:source.run,stem:source.stem,identities:result?result.identities:null});
    }
    drawEmptyCohort(){
      this.host.querySelectorAll("canvas").forEach(c=>{this.canvas(c.id.slice(3));this.blank(c,"No cells match this filter. Edit the rules or turn filtering off.");});
      this.$("trace-grid").replaceChildren();this.$("selection-status").textContent="No cells in the filtered cohort";
      for(const id of ["timeline-axis","timeline-legend","timeline-detail"])this.$(id).replaceChildren();
    }
    bind(){
      this.bindRhythms();
      const clockLabel=document.createElement("label");clockLabel.textContent="Metric ";const clock=document.createElement("select");clock.id="xp-clock-metric";clockLabel.append(clock);this.$("clock-panel").querySelector(".panel-heading").append(clockLabel);
      this.$("recording-select").onchange=()=>{const [run,stem]=JSON.parse(this.val("recording-select"));this.switchRecording({run,stem});};
      this.$("cell-select").onchange=()=>this.select({identity:+this.val("cell-select")});
      this.$("clock-metric").onchange=()=>{if(this.phaseAutomatic)this.loadPhases();else this.drawClock();this.saveSession();};
      this.$("timeline-order").onchange=()=>{this.timelineKey=null;this.drawEvents();this.saveSession();};
      this.$("map-layer").onchange=this.$("tectonics-metric").onchange=()=>{this.drawMap();this.saveSession();};
      for(const slot of ["peer-one","peer-two"])this.$(slot).onchange=()=>{this.requestPeers(true);this.drawCellCards();this.updateRhythmAnnotations();this.saveSession();};
      this.$("image-mode").onchange=()=>this.select({frame:this.state.frame});
      const repaintImages=()=>{this.clearLutCache();if(this.data){this.drawImage();this.drawChange();this.drawCellCards();this.saveSession();}};
      this.$("image-lut").onchange=repaintImages;
      this.$("contrast").oninput=repaintImages;
      const gridJump=document.createElement("button");gridJump.type="button";gridJump.textContent="All cell traces";gridJump.title="Jump to the trace grid for this recording";gridJump.onclick=()=>this.$("trace-grid-group").scrollIntoView({block:"start"});this.$("trace-panel").querySelector(".panel-heading").append(gridJump);
      this.traceLegend=document.createElement("span");this.traceLegend.className="xp-trace-legend";this.traceLegend.innerHTML='<span class="xp-significant-key">* Significant rhythm</span><span class="xp-other-key">Other / not tested</span>';this.traceLegend.hidden=true;this.$("trace-panel").querySelector(".panel-foot").append(this.traceLegend);
      for(const prefix of ["trace","trace-grid"])for(const setting of ["metric","aggregation","smoothing","fit"]){
        this.$(prefix+"-"+setting).onchange=()=>{this.syncTraceGridControls(prefix);this.draw();this.saveSession();};
      }
      const repaintTraces=()=>{if(!this.data)return;this.syncTracePicks();this.drawTrace();this.drawCellCards();this.drawTraceGrid();this.updateRhythmAnnotations();this.saveSession();};
      this.$("trace-grid-overlay-toggle").onchange=()=>{this.$("trace-grid-layout").value=this.$("trace-grid-overlay-toggle").checked?"overlay":"grid";repaintTraces();};
      this.bindGridMeasurementControls(repaintTraces);
      for(const id of ["trace-layout","trace-overlay-scale","trace-grid-layout","trace-grid-overlay-scale","trace-measurements-list","trace-cells-list"])this.$(id).onchange=repaintTraces;
      for(const [id,choose] of [["trace-cells-all",()=>this.data.identities],["trace-cells-clear",()=>[]],["trace-cells-nearby",()=>[this.state.identity,+this.val("peer-one"),+this.val("peer-two")]]])this.$(id).onclick=()=>{const selected=new Set(choose());this.$("trace-cells-list").querySelectorAll("input").forEach(el=>el.checked=selected.has(+el.value));repaintTraces();};
      this.$("trace-comparison-canvas").onclick=event=>this.selectTraceComparison(event);
      this.$("trace-comparison-canvas").onkeydown=event=>{if(["ArrowLeft","ArrowRight"].includes(event.key)){this.select({frame:this.state.frame+(event.key==="ArrowLeft"?-1:1)});event.preventDefault();}};
      this.$("frame-slider").oninput=()=>this.select({frame:+this.val("frame-slider")});
      this.$("previous").onclick=()=>this.select({frame:this.state.frame-1});this.$("next").onclick=()=>this.select({frame:this.state.frame+1});
      this.$("play").onclick=()=>{if(this.timer){this.pause();return;}if(!this.data)return;this.$("play").textContent="Pause";this.timer=setInterval(()=>{if(this.state.frame===this.data.frames.length-1){this.pause();return;}this.select({frame:this.state.frame+1});},1000/+this.val("play-rate"));};
      this.$("play-rate").onchange=()=>{if(this.timer){this.pause();this.$("play").click();}};
      this.host.querySelectorAll("select,input").forEach(el=>{if(!el.onchange&&!el.oninput)el.addEventListener("change",()=>{this.draw();this.saveSession();});});
      this.$("reset-rotation").onclick=()=>{this.rotation={yaw:-.55,pitch:.6,zoom:1};this.drawHistory();};
      for(const id of ["trace","peer-one-trace","peer-two-trace","events","barcode"]){this.$(id).onclick=e=>{if(!this.data)return;const p=this.point(id,e),plot=this.hits[id];if(!plot)return;const fraction=Math.max(0,Math.min(1,(p.x-plot.left)/(plot.right-plot.left)));const f=this.nearestFrame(fraction);const patch={frame:f};if(plot.cells){const row=Math.floor((p.y-plot.top)/plot.rowHeight);if(row>=0&&row<plot.cells.length)patch.identity=plot.cells[row].identity;}this.select(patch);};}
      this.$("events").onmousemove=event=>{if(this.val("event-mode")!=="gantt")return;const point=this.point("events",event),plot=this.hits.events;if(!plot?.cells)return;const row=Math.floor((point.y-plot.top)/plot.rowHeight),cell=plot.cells[row];if(cell)this.timelineDetail(cell,this.nearestFrame(Math.max(0,Math.min(1,(point.x-plot.left)/(plot.right-plot.left)))));};
      this.$("events").onmouseleave=()=>this.timelineDetail();
      for(const id of ["cell-image","tissue-map","relationships","phase-clock","distribution"]){this.$(id).onclick=e=>{const p=this.point(id,e);const hit=(this.hits[id]||[]).reduce((best,h)=>{const d=Math.hypot(p.x-h.x,p.y-h.y);return d<best.distance?{...h,distance:d}:best;},{distance:25});if(hit.identity!=null)this.select({identity:hit.identity});};}
      this.host.querySelectorAll("canvas").forEach(el=>el.addEventListener("keydown",e=>{if(!this.data)return;if(el.id==="xp-outline-history"){if(e.key.startsWith("Arrow")){this.rotation.yaw+=(e.key==="ArrowLeft"?-.12:e.key==="ArrowRight"?.12:0);this.rotation.pitch+=(e.key==="ArrowUp"?-.08:e.key==="ArrowDown"?.08:0);this.drawHistory();e.preventDefault();}}else if(["ArrowLeft","ArrowRight"].includes(e.key)){this.select({frame:this.state.frame+(e.key==="ArrowLeft"?-1:1)});e.preventDefault();}}));
      const history=this.$("outline-history");let drag=null;
      history.onpointerdown=e=>{drag={x:e.clientX,y:e.clientY};history.setPointerCapture(e.pointerId);};
      history.onpointermove=e=>{if(!drag)return;this.rotation.yaw+=(e.clientX-drag.x)*.007;this.rotation.pitch+=(e.clientY-drag.y)*.007;drag={x:e.clientX,y:e.clientY};this.drawHistory();};
      history.onpointerup=history.onpointercancel=()=>drag=null;
      history.addEventListener("wheel",e=>{this.rotation.zoom=Math.max(.4,Math.min(3,this.rotation.zoom*(e.deltaY>0?.9:1.1)));this.drawHistory();e.preventDefault();},{passive:false});
      const historyOption=this.$("history-window").querySelector('option[value="48"]');historyOption.textContent="48 frames around selection";
      this.exports();
      this.plotEditors();
      this.attachHelp();
      window.addEventListener("resize",()=>{if(this.data&&!this.resizePending){this.resizePending=requestAnimationFrame(()=>{this.resizePending=null;this.draw();});}});
      for(const [panel,title,action,pipeline] of [["clock-panel","Configure rhythm analysis","rhythm_discovery",true],["events-panel","Configure state analysis","behaviour_states",true],["network-panel","Configure contact measurements","measure",false]]){
        const button=document.createElement("button");button.type="button";button.textContent=title;button.style.marginTop="10px";button.onclick=()=>this.options.onAction(action,{run:this.data?.run||""},pipeline);this.$(panel).append(button);
      }
    }
    exports(panels=null){const cat=this.options.catalogue;for(const panel of panels||this.host.querySelectorAll("[data-plot-type]")){if(panel.dataset.nativeComparison)continue;const entries=(cat?.capabilities||[]).filter(r=>r.kind==="figure view"&&r.route===panel.dataset.plotType&&(!panel.dataset.figure||r.name===panel.dataset.figure));if(!entries.length)continue;const controls=document.createElement("details");controls.className="xp-export";controls.innerHTML=`<summary>Export figure</summary><div class="xp-export-actions"><label>Saved figure <select aria-label="Configure saved ${escape(panel.dataset.plotType)} figure">${entries.map((r,i)=>`<option value="${i}">${escape(r.title)} · ${escape(r.view)}</option>`).join("")}</select></label><button type="button">Configure export</button></div>`;controls.querySelector("button").onclick=()=>{const entry=entries[+controls.querySelector("select").value];this.options.onAction(entry.name,{run:this.data?.run||"",stem:this.data?.stem||"",view:entry.view,...(entry.name==="all_cell_trace_grid"?{time_guides_hours:+this.val("trace-time-guides")||null,time_guides_offset_hours:+this.val("trace-time-guides-offset")||0}:{})});};panel.append(controls);}}
    pause(){if(this.timer)clearInterval(this.timer);this.timer=null;if(this.$("play"))this.$("play").textContent="Play";}
    point(id,e){const c=this.$(id),r=c.getBoundingClientRect();return {x:(e.clientX-r.left)*(Number(c.dataset.logicalWidth)||c.width)/r.width,y:(e.clientY-r.top)*(Number(c.dataset.logicalHeight)||c.height)/r.height};}
    nearestFrame(fraction){const frames=this.data.frames;const hours=frames.map(f=>f.hours);if(hours.every(C.finite)){const target=hours[0]+fraction*(hours.at(-1)-hours[0]);return hours.reduce((best,h,i)=>Math.abs(h-target)<Math.abs(hours[best]-target)?i:best,0);}return Math.round(fraction*(frames.length-1));}
    fraction(frame){const f=this.data.frames;return C.finite(f[frame]?.hours)&&C.finite(f[0]?.hours)&&C.finite(f.at(-1)?.hours)&&f.at(-1).hours!==f[0].hours?(f[frame].hours-f[0].hours)/(f.at(-1).hours-f[0].hours):frame/Math.max(1,f.length-1);}
    url(endpoint,extra={}){return `/api/exploration/${endpoint}?`+new URLSearchParams({run:this.data.run,stem:this.data.stem,...extra});}
    bindRhythms(){
      this.rhythmNotes=new Map();
      for(const panel of this.host.querySelectorAll("[data-plot-type]")){
        if(["xp-trace-panel","xp-trace-grid-panel","xp-events-panel","xp-outline-panel"].includes(panel.id))continue;
        const note=document.createElement("span");note.className="xp-rhythm-note";note.hidden=true;note.dataset.rhythmFor=panel.id;
        panel.append(note);this.rhythmNotes.set(panel.id,note);
      }
      for(const card of this.host.querySelectorAll(".xp-cell-card")){const note=document.createElement("span");note.className="xp-rhythm-note";note.hidden=true;card.append(note);}
      if(window.PyMicrogliaRhythmRows)return;
      this.$("run-rhythms").onclick=()=>this.runRhythms();
      this.$("rhythm-metric-list").onchange=()=>{this.syncRhythmMetrics();this.saveSession();};
      for(const [id,checked] of [["rhythm-select-all",true],["rhythm-clear",false]])this.$(id).onclick=()=>{this.$("rhythm-metric-list").querySelectorAll("input").forEach(el=>{if(!el.disabled)el.checked=checked;});this.syncRhythmMetrics();this.saveSession();};
      this.$("rhythm-settings-toggle").onclick=()=>{this.$("rhythm-settings").open=!this.$("rhythm-settings").open;this.$("rhythm-settings-toggle").setAttribute("aria-expanded",String(this.$("rhythm-settings").open));this.saveSession();};
      this.$("rhythm-recipe").onchange=()=>{const recipe=this.rhythmSchema?.recipes.find(r=>r.key===this.val("rhythm-recipe"));if(recipe){this.$("rhythm-method").value=recipe.fit_method;this.$("rhythm-test").value=recipe.test;for(const [id,key] of RHYTHM_FIELDS)if(key in recipe.required)this.$(id).value=recipe.required[key];}else{this.$("rhythm-settings").open=true;this.$("rhythm-settings-toggle").setAttribute("aria-expanded","true");}this.rhythmControlsChanged();};
      for(const id of ["rhythm-test",...RHYTHM_FIELDS.map(([id])=>id)])this.$(id).onchange=()=>this.rhythmControlsChanged(id);
      this.$("rhythm-use-shown").onclick=()=>{this.setRhythmForm(this.appliedRhythmOptions());this.rhythmControlsChanged();};
    }
    selectedRhythmMetrics(){return [...this.host.querySelectorAll("#xp-rhythm-metric-list input:checked")].filter(el=>el.dataset.available!=="false").map(el=>el.value);}
    setRhythmMetrics(metrics,chosen){
      this.rhythmPicker=this.measurementPicker("rhythm-metric-list",{selected:chosen,choices:metrics});
      window.PyMicrogliaMeasurementPicker.popup(this.$("rhythm-measurements"),this.host.querySelector(".xp-rhythm-choices"));
      this.syncRhythmMetrics();
    }
    syncRhythmMetrics(){this.rhythmPicker?.refresh();const count=this.selectedRhythmMetrics().length;this.$("rhythm-measurements-toggle").textContent=`Measurements (${count})`;this.$("run-rhythms").disabled=!this.rhythmSchema||!count||!!this.rhythmRunning;if(this.rhythmSchema)this.updateRhythmSettings();}
    rhythmFormOptions(){const options={metrics:this.selectedRhythmMetrics()};for(const [id,key,numeric] of RHYTHM_FIELDS)options[key]=numeric?Number(this.val(id)):this.val(id);options.fft_component_test=this.val("rhythm-test")==="fft_component";options.significance_method=options.fft_component_test?null:this.val("rhythm-test");return options;}
    appliedRhythmOptions(){const result=this.data?.results?.rhythms;if(!result?.analysis_id||result.preview)return null;const config={...this.rhythmSchema.defaults,...result.resolved_options};for(const [,key] of RHYTHM_FIELDS)if(result.settings?.[key]!=null)config[key]=result.settings[key];config.fft_component_test=result.settings?.fft_component_test??config.fft_component_test;config.significance_method=config.fft_component_test?null:result.settings?.significance_method||config.significance_method;config.metrics=result.metrics;return config;}
    setRhythmForm(config){if(!config)return;const plan=C.rhythmPlan(config,this.rhythmSchema);for(const [id,key] of RHYTHM_FIELDS)this.$(id).value=plan[key];this.$("rhythm-test").value=plan.test;this.setRhythmMetrics(this.rhythmSchema.metrics,config.metrics||this.rhythmSchema.selected_metrics);}
    rhythmControlsChanged(){
      if(!this.rhythmSchema)return;let notice="";
      const componentRecipe=this.rhythmSchema.recipes.find(r=>r.test==="fft_component");
      const supportedDetrends=componentRecipe?.compatible_detrends||[componentRecipe?.required?.detrend||"robust_linear"];
      const compatible=this.val("rhythm-method")==="fft_nlls"&&supportedDetrends.includes(this.val("rhythm-detrend"))&&this.val("rhythm-correction")==="none";
      if(this.val("rhythm-test")==="fft_component"&&!compatible){this.$("rhythm-test").value=this.rhythmSchema.tests.find(t=>t.key===this.val("rhythm-method"))?.key||"lomb";notice=`Fourier component testing requires a supported baseline and no correction; using ${this.rhythmSchema.tests.find(t=>t.key===this.val("rhythm-test"))?.label||this.val("rhythm-test")} instead.`;}
      const component=Array.from(this.$("rhythm-test").options).find(o=>o.value==="fft_component");if(component)component.disabled=!compatible;
      this.rhythmCompatibilityNotice=notice;this.updateRhythmSettings();this.saveSession();
    }
    rhythmPlanText(plan){const recipe=this.rhythmSchema.recipes.find(r=>r.fit_method===plan.fit_method&&r.test===plan.test),method=recipe?.label||this.rhythmSchema.methods.find(m=>m.key===plan.fit_method)?.label||plan.fit_method,test=this.rhythmSchema.tests.find(t=>t.key===plan.test)?.label||plan.test,detrend=this.rhythmSchema.detrends.find(d=>d.key===plan.detrend)?.label||plan.detrend;return `${method}${recipe?"":" + "+test} · ${detrend} · ${plan.period_min_hours}–${plan.period_max_hours} h`;}
    updateRhythmSettings(){
      if(!this.rhythmSchema||!this.data)return;
      const selected=C.rhythmPlan(this.rhythmFormOptions(),this.rhythmSchema),applied=this.appliedRhythmOptions(),shown=applied?C.rhythmPlan(applied,this.rhythmSchema):null,dirty=!shown||!C.sameRhythmPlan(selected,shown),recipe=this.rhythmSchema.recipes.find(r=>r.fit_method===selected.fit_method&&r.test===selected.test);
      this.$("rhythm-recipe").value=recipe?.key||"custom";this.$("rhythm-recipe").title=recipe?.description||"Custom period estimator and significance test; see Advanced.";
      this.$("rhythm-applied").textContent=shown?`Shown results: ${this.rhythmPlanText(shown)}`:this.data.results?.rhythms?.preview?"Shown: phase preview; rhythms have not been tested.":"No tested rhythm results.";
      const pending=this.$("rhythm-pending");pending.hidden=!dirty&&!this.rhythmCompatibilityNotice;pending.textContent=(dirty?`Pending: ${this.rhythmPlanText(selected)} · Run to apply.`:"")+(this.rhythmCompatibilityNotice?" "+this.rhythmCompatibilityNotice:"");
      this.$("rhythm-use-shown").hidden=!shown||!dirty;
      this.$("run-rhythms").textContent=shown?(dirty?"Apply rhythm settings":"Re-run rhythm analysis"):"Run rhythm analysis";
      this.$("rhythm-applied").title=shown?`Test: ${shown.test}; detrending: ${shown.detrend}; correction: ${shown.multiple_testing}; threshold: ${shown.rhythmic_alpha}; minimum samples: ${shown.min_observations}; minimum cycles: ${shown.min_cycles}; measurements: ${shown.metrics.join(", ")}.`:"";
    }
    async loadRhythmOptions(saved){
      const data=this.data;
      try{
        const schema=await this.options.api(this.url("rhythm-options"));if(data!==this.data)return;
        this.rhythmSchema=schema;const defaults=schema.defaults;
        this.setOptions("rhythm-method",schema.methods.map(m=>[m.key,m.label]),defaults.fit_method);
        this.setOptions("rhythm-test",schema.tests.map(m=>[m.key,m.label]),defaults.fft_component_test?"fft_component":defaults.significance_method||"lomb");
        this.setOptions("rhythm-detrend",schema.detrends.map(m=>[m.key,m.label]),defaults.detrend);
        this.setOptions("rhythm-recipe",[...schema.recipes.map(r=>[r.key,r.label]),["custom","Custom method / test"]]);
        const applied=this.appliedRhythmOptions(),initial=applied||defaults;for(const [id,key] of RHYTHM_FIELDS)this.$(id).value=initial[key];this.$("rhythm-test").value=C.rhythmPlan(initial,schema).test;
        const legacy=saved?.controls?.["xp-rhythm-metric"];
        const selected=saved?.rhythmMetrics??(legacy==="all"?schema.metrics.map(m=>m.key):legacy?[legacy]:data.results?.rhythms?.analysis_id?data.results.rhythms.metrics:schema.selected_metrics);
        this.setRhythmMetrics(schema.metrics,selected);
        for(const [id,value] of Object.entries(saved?.controls||{})){if(!id.startsWith("xp-rhythm-"))continue;const el=this.host.querySelector("#"+id);if(!el||el.type==="number"&&(value===""||value==null))continue;if(el.type==="checkbox")el.checked=Boolean(value);else if(el.tagName!=="SELECT"||[...el.options].some(o=>o.value===value))el.value=value;}
        this.rhythmControlsChanged();this.syncRhythmMetrics();this.updateRhythmAnnotations();this.saveSession();
        const jobs=await this.options.api("/api/jobs");if(data!==this.data)return;
        const job=jobs.find(j=>j.kind==="exploration_rhythms"&&j.params.run===data.run&&j.params.stem===data.stem&&["queued","running"].includes(j.state));
        if(job)this.watchRhythms(job,data);else this.rhythmStatus();
      }catch(error){if(data===this.data)this.$("rhythm-status").textContent=error.message;}
    }
    rhythmStatus(){
      this.$("rhythm-progress").hidden=true;
      this.updateRhythmSettings();
      const result=this.data?.results?.rhythms;if(!result||result.preview){this.$("rhythm-status").textContent="Rhythm analysis not run";return;}
      const rows=result.rows||[],cells=new Set(rows.map(r=>r.identity));
      this.$("rhythm-status").textContent=`${cells.size} cells · ${rows.filter(r=>r.status==="rhythmic").length} rhythmic traces`;
    }
    async runRhythms(){
      if(!this.data||!this.rhythmSchema)return;const data=this.data,d=this.rhythmSchema.defaults,options={};
      options.metrics=this.selectedRhythmMetrics();if(!options.metrics.length){this.syncRhythmMetrics();return;}
      this.$("rhythm-measurements").open=false;this.rhythmRunning=true;
      for(const [id,key,numeric] of RHYTHM_FIELDS){const value=numeric?Number(this.val(id)):this.val(id);if(value!==d[key])options[key]=value;}
      const component=this.val("rhythm-test")==="fft_component";
      if(component!==d.fft_component_test)options.fft_component_test=component;
      if(!component&&this.val("rhythm-test")!==d.significance_method)options.significance_method=this.val("rhythm-test");
      this.$("run-rhythms").disabled=true;this.$("rhythm-progress").hidden=false;this.$("rhythm-progress").removeAttribute("value");this.$("rhythm-status").textContent="Queuing rhythm analysis...";
      try{const job=await this.options.api("/api/exploration/rhythm-jobs",{method:"POST",body:JSON.stringify({run:data.run,stem:data.stem,options})});if(data===this.data){this.saveSession();this.watchRhythms(job,data);}}
      catch(error){if(data===this.data){this.rhythmRunning=false;this.syncRhythmMetrics();this.$("rhythm-progress").hidden=true;this.$("rhythm-status").textContent=error.message;}}
    }
    showRhythmProgress(job){
      const bar=this.$("rhythm-progress"),p=job.progress||{},duration=seconds=>{const s=Math.max(0,Math.round(seconds));return s<60?`${s}s`:`${Math.floor(s/60)}m ${s%60}s`;};
      bar.hidden=false;
      if(job.state==="queued"){bar.removeAttribute("value");this.$("rhythm-status").textContent="Rhythm analysis queued";return;}
      const elapsed=job.started_at?Math.max(0,(Date.now()-Date.parse(job.started_at))/1000):p.elapsed_seconds||0;
      if(!p.total){bar.removeAttribute("value");this.$("rhythm-status").textContent=`Preparing rhythm analysis · ${duration(elapsed)} elapsed`;return;}
      bar.max=p.total;bar.value=p.completed||0;
      const count=`${p.completed_cells||0}/${p.total_cells||job.cells} cells`,percent=Math.floor((p.completed||0)/p.total*100);
      const active=p.active_cells?.filter(cell=>cell.stage!=="testing")||[],current=active[0],stage=p.stage==="preparing"?"Preparing measurements":p.stage==="starting_workers"?"Starting parallel fits":p.stage==="saving"?"Saving results":active.length>1?`Analysing ${active.length} traces (cells ${[...new Set(active.map(cell=>cell.identity))].join(", ")})`:current?`Cell ${current.identity}: ${current.stage==="component_test"?"testing fitted rhythms":"fitting rhythm"}`:p.stage==="component_test"?`Cell ${p.identity}: testing fitted rhythms`:p.stage==="fitting"?`Cell ${p.identity}: fitting rhythm`:"Testing rhythms";
      const traces=p.total!==p.total_cells?` · ${p.completed||0}/${p.total} traces`:"";
      const metric=p.metric&&p.total!==p.total_cells?` (${this.data.metrics.find(m=>m.key===p.metric)?.label||p.metric})`:"";
      const remaining=p.remaining_seconds>0?` · ~${duration(p.remaining_seconds)} left`:"";
      this.$("rhythm-status").textContent=`${count}${traces} · ${percent}% · ${stage}${metric} · ${duration(elapsed)} elapsed${remaining}`;
    }
    watchRhythms(job,data){
      clearTimeout(this.rhythmJobTimer);this.rhythmJobId=job.id;this.rhythmRunning=true;this.$("run-rhythms").disabled=true;
      const poll=async current=>{
        if(data!==this.data)return;
        this.showRhythmProgress(current);
        if(current.state==="failed"){this.rhythmRunning=false;this.syncRhythmMetrics();this.$("rhythm-progress").hidden=true;this.$("rhythm-status").textContent=current.error;return;}
        if(current.state==="completed"){
          try{const result=await this.options.api(this.url("rhythms"));if(data!==this.data)return;this.applyRhythms(result);this.rhythmStatus();if(current.reused)this.$("rhythm-status").textContent+=" · Saved results reused";else if(current.progress?.elapsed_seconds!=null){const seconds=Math.round(current.progress.elapsed_seconds);this.$("rhythm-status").textContent+=` · Finished in ${seconds<60?`${seconds}s`:`${Math.floor(seconds/60)}m ${seconds%60}s`}`;}}
          catch(error){if(data===this.data){this.$("rhythm-progress").hidden=true;this.$("rhythm-status").textContent=error.message;}}
          if(data===this.data){this.rhythmRunning=false;this.syncRhythmMetrics();}return;
        }
        this.rhythmJobTimer=setTimeout(async()=>{try{const next=await this.options.api(`/api/jobs/${job.id}`);poll(next);}catch(error){if(data===this.data){this.$("rhythm-status").textContent=`Progress connection interrupted; analysis may still be running. ${error.message}`;this.rhythmJobTimer=setTimeout(()=>poll(current),2000);}}},1000);
      };poll(job);
    }
    applyRhythms(result){
      this.phaseRequest++;this.phaseAutomatic=false;this.data.results.rhythms=result;this.rhythmIndexSource=null;
      this.setOptions("clock-metric",result.metrics.map(key=>[key,this.data.metrics.find(m=>m.key===key)?.label||key]),result.metrics[0]);
      for(const id of ["map-layer","barcode-order"]){const el=this.$(id);for(const option of [...el.options])if(option.value.startsWith("phase:"))option.remove();for(const metric of result.metrics)el.add(new Option("Tested phase · "+(this.data.metrics.find(m=>m.key===metric)?.label||metric),"phase:"+metric));}
      this.setOptions("tectonics-metric",result.metrics.map(k=>[k,this.data.metrics.find(m=>m.key===k)?.label||k]));this.comparisons.refresh();
      this.draw();this.updateRhythmSettings();this.filters?.refreshApplied();this.outlineFilters?.refreshApplied();
    }
    rhythmFor(identity,prefix){
      const metric=this.rhythmAnalyses?this.rhythmAnalyses.metricFor(prefix||"trace"):prefix==="clock"?this.val("clock-metric"):C.metricKey(this.val((prefix||"trace")+"-metric"),this.val((prefix||"trace")+"-aggregation"),this.data);
      return this.rhythmForMetric(identity,metric);
    }
    rhythmForMetric(identity,metric){
      if(this.rhythmAnalyses&&(!this.plotView||["trace","trace-grid"].includes(this.plotView)))metric=this.rhythmAnalyses.traceMetric(metric);
      const result=this.data?.results?.rhythms;if(!result||result.preview)return null;
      if(this.rhythmIndexSource!==result){this.rhythmIndexSource=result;this.rhythmIndex=new Map(result.rows.map(r=>[`${r.identity}:${r.metric}`,r]));}
      const exact=this.rhythmIndex.get(`${identity}:${metric}`);if(exact)return exact;
      if(metric==="integrated_density")return !this.rhythmAnalyses?this.rhythmIndex.get(`${identity}:integrated_density_mean3`)||null:null;
      if(metric==="integrated_density_mean3")return !this.rhythmAnalyses?this.rhythmIndex.get(`${identity}:integrated_density`)||null:null;
      return null;
    }
    rhythmText(identity,prefix,compact=false){
      const row=this.rhythmFor(identity,prefix);if(!row)return `Cell ${identity} · measurement not analysed`;
      const metric=this.data.metrics.find(m=>m.key===row.metric)?.label||row.metric;
      const state=row.status==="rhythmic"?"Rhythmic":row.status==="not rhythmic"||row.status==="arrhythmic"?"Not rhythmic":"Not tested";
      const probability=row.correction&&row.correction!=="none"&&C.finite(row.q_value)?`q=${fmt(row.q_value)}`:C.finite(row.p_value)?`p=${fmt(row.p_value)}`:"";
      return `${compact?"":`Cell ${identity} · ${metric} · `}${state}${C.finite(row.period_hours)?` · ${state==="Rhythmic"?"":"candidate "}${fmt(row.period_hours)} h`:""}${probability?" · "+probability:""}${row.period_underdetermined&&C.finite(row.period_hours)?" · short record":""}${!C.finite(row.period_hours)&&row.reason?" · "+row.reason.replaceAll("_"," "):""}`;
    }
    updateRhythmAnnotations(){
      if(!this.data)return;const result=this.data.results?.rhythms,visible=this.rhythmResultsVisible()&&!!result?.rows?.length&&!result.preview;
      const colours=this.traceRhythmColours();
      if(this.traceLegend){const overlay=this.val("trace-layout")==="overlay"&&this.traceMetrics().length>1;this.traceLegend.hidden=!colours;this.traceLegend.querySelector(".xp-other-key").textContent=overlay?"Colours distinguish measurements":"Other / not tested";this.traceLegend.title=overlay?"Each measurement has its own colour; an asterisk marks its significant saved rhythm.":"Colours use the saved test for the displayed measurement. Grey cells may be non-significant, untested or not analysed; their labels distinguish these states.";}
      const update=(note,identity,prefix,compact=false)=>{note.hidden=!visible;if(!visible)return;note.textContent=this.rhythmText(identity,prefix,compact);note.title=this.rhythmText(identity,prefix);note.dataset.rhythmStatus=this.rhythmFor(identity,prefix)?.status||"unknown";};
      for(const [id,note] of this.rhythmNotes){const prefix=id==="xp-trace-panel"?"trace":id==="xp-barcode-panel"?"barcode":id==="xp-distribution-panel"?"distribution":id==="xp-clock-panel"?"clock":null;update(note,this.state.identity,prefix);}
      for(const card of this.host.querySelectorAll(".xp-cell-card")){const select=card.querySelector("select");update(card.querySelector(".xp-rhythm-note"),select?Number(select.value):this.state.identity,"trace",true);}
      for(const [identity,{note,tile,button,annotation}] of this.gridTiles||[]){
        update(note,identity,"trace",true);const metric=C.metricKey(this.val("trace-metric"),this.val("trace-aggregation"),this.data),row=visible?this.rhythmForMetric(identity,metric):null;
        tile.dataset.rhythmStatus=row?.status||"unknown";tile.style.setProperty("--xp-trace-colour",this.traceColour(identity,metric,"trace-grid"));tile.style.setProperty("--xp-trace-background",this.style("trace-grid").background);
        button.textContent=`Cell ${identity}${row?.status==="rhythmic"?" *":""}`;
        annotation.hidden=!visible;annotation.textContent=visible?this.traceRhythmLabel(row):"";annotation.title=visible?this.rhythmText(identity,"trace"):"";
        note.title=annotation.title;note.hidden=true;
      }
    }
    async loadPhases(){
      if(!this.data||!this.phaseAutomatic)return;
      const data=this.data,metric=this.val("clock-metric"),request=++this.phaseRequest;
      data.results.rhythms={rows:[],metrics:[metric],message:"Computing fixed-cycle phases...",preview:true};this.drawClock();
      try{
        if(!this.phaseCache.has(metric))this.phaseCache.set(metric,this.options.api(this.url("phases",{metric})));
        const result=await this.phaseCache.get(metric);if(data!==this.data||request!==this.phaseRequest)return;
        data.results.rhythms=result;
        const label=data.metrics.find(m=>m.key===metric)?.label||metric,phaseOption=["phase:"+metric,"Phase preview | "+label];
        for(const id of ["map-layer","barcode-order"]){const el=this.$(id);if(![...el.options].some(o=>o.value===phaseOption[0]))el.add(new Option(phaseOption[1],phaseOption[0]));}
        this.drawClock();this.drawBarcode();this.drawMap();
      }catch(error){if(data!==this.data||request!==this.phaseRequest)return;this.phaseCache.delete(metric);data.results.rhythms.message=error.message;this.drawClock();}
    }
    async loadContacts(){
      if(!this.data||!this.previewContacts||this.contactBusy===this.data)return;
      const data=this.data,frame=this.state.frame;this.contactBusy=data;
      try{
        if(!this.contactCache.has(frame))this.contactCache.set(frame,this.options.api(this.url("contacts",{frame})));
        const result=await this.contactCache.get(frame);if(data!==this.data||frame!==this.state.frame)return;
        data.results.relations=result;this.setOptions("link-source",result.sources.map(s=>[s.key,s.title]));this.drawNetwork();
      }catch(error){if(data!==this.data||frame!==this.state.frame)return;this.contactCache.delete(frame);data.results.relations={sources:[],message:error.message};this.drawNetwork();}
      finally{if(this.contactBusy===data)this.contactBusy=null;if(data===this.data&&frame!==this.state.frame)this.loadContacts();}
    }
    async loadLuts(){
      try{const catalogue=await this.options.api("/api/exploration/luts");
        this.lutTables=new Map(catalogue.maps.map(map=>[map.key,map.table]));
        this.setOptions("image-lut",catalogue.maps.map(map=>[map.key,map.label]),catalogue.default);
      }catch{this.$("image-lut").title="Grayscale available; shared colour lookup tables could not be loaded.";}
    }
    clearLutCache(){this.lutCache.clear();this.lutCacheBytes=0;}
    signalImage(image){
      const lut=this.val("image-lut"),brightness=+this.val("contrast")||1;
      if(lut==="grays"&&brightness===1)return image;
      const key=lut+":"+brightness,cached=this.lutCache.get(image);
      if(cached?.key===key){this.lutCache.delete(image);this.lutCache.set(image,cached);return cached.surface;}
      if(cached){this.lutCacheBytes-=cached.bytes;this.lutCache.delete(image);}
      const surface=document.createElement("canvas");surface.width=image.naturalWidth;surface.height=image.naturalHeight;
      const ctx=surface.getContext("2d",{willReadFrequently:true});ctx.drawImage(image,0,0);
      const pixels=ctx.getImageData(0,0,surface.width,surface.height),rgba=pixels.data,table=this.lutTables.get(lut);
      for(let i=0;i<rgba.length;i+=4){const level=Math.min(255,Math.max(0,Math.round(rgba[i]*brightness))),colour=table?.[level]||[level,level,level];rgba[i]=colour[0];rgba[i+1]=colour[1];rgba[i+2]=colour[2];}
      ctx.putImageData(pixels,0,0);
      const bytes=surface.width*surface.height*4,limit=16*1024*1024;
      if(bytes<=limit){this.lutCache.set(image,{key,surface,bytes});this.lutCacheBytes+=bytes;
        while(this.lutCacheBytes>limit){const first=this.lutCache.keys().next().value;this.lutCacheBytes-=this.lutCache.get(first).bytes;this.lutCache.delete(first);}}
      return surface;
    }
    async decodeImage(source){const img=new Image();await new Promise(resolve=>{img.onload=img.onerror=resolve;img.src=source;});return img.naturalWidth?img:null;}
    async fetchFieldMedia(frame,identity,epoch,signal,current){
      let finished=false;
      // Both requests retain the same frame and source selection. A late preview
      // cannot downgrade decoded detail or install pixels from another request.
      if(!this.fieldSize||Math.max(...this.fieldSize)>256)this.options.api(this.url('frame-image',{frame,preview:true}),{signal})
        .then(async payload=>{const image=await this.decodeImage(payload.image);
          if(image&&!finished&&current()){
            this.imageMeta={...payload,preview:true};this.image=image;this.imageError='';
            this.fieldSize=[payload.width,payload.height];this.drawImage();
          }}).catch(()=>{});
      try{
        const [payload,cell]=await Promise.all([
          this.options.api(this.url('frame',{frame}),{signal}),
          this.fetchCellMedia(frame,identity,epoch,signal)]);
        const image=await this.decodeImage(payload.image);
        if(!image)throw new Error('The full source image could not be decoded.');
        return {payload,image,change:cell?.change};
      }finally{finished=true;}
    }
    cacheImage(frame,payload,img){const bytes=payload.image.length+payload.width*payload.height*4,limit=64*1024*1024;this.imageSampleBytes=bytes;if(!img||bytes>limit)return;const old=this.imageCache.get(frame);if(old){this.imageCacheBytes-=old.bytes;this.imageCache.delete(frame);}this.imageCache.set(frame,{payload,image:img,bytes});this.imageCacheBytes+=bytes;while(this.imageCacheBytes>limit){const first=this.imageCache.keys().next().value;this.imageCacheBytes-=this.imageCache.get(first).bytes;this.imageCache.delete(first);}}
    cancelMedia(){this.detailsController?.abort();this.mediaController?.abort();this.mediaBackgroundController?.abort();this.peerMediaController?.abort();clearTimeout(this.mediaPrefetchTimer);clearTimeout(this.frameDebounce);this.mediaRequest=(this.mediaRequest||0)+1;this.imageQueue=[];}
    cacheCellMedia(frame,row){
      const old=this.cellImageCache.get(frame);this.cellImageBytes=(this.cellImageBytes||0)-(old?.bytes||0);
      row.bytes=row.payload.image.length+(row.payload.change?.length||0)+(row.image.naturalWidth*row.image.naturalHeight+(row.change?.naturalWidth||0)*(row.change?.naturalHeight||0))*4;
      this.cellImageCache.delete(frame);this.cellImageCache.set(frame,row);this.cellImageBytes+=row.bytes;
      this.changeCache.set(frame,{payload:row.payload,image:row.change});
      while(this.cellImageBytes>64*1024*1024&&this.cellImageCache.size>1){const first=this.cellImageCache.keys().next().value;this.cellImageBytes-=this.cellImageCache.get(first).bytes;this.cellImageCache.delete(first);this.changeCache.delete(first);}
    }
    async fetchCellMedia(frame,identity,epoch,signal){
      const url=this.url('cell-frame',{frame,identity}),fallback=this.url('frame-image',{frame});
      const payload=await this.options.api(url,{signal});if(epoch!==this.cellImageEpoch||signal.aborted)return null;
      const pixels=payload.available?payload:await this.options.api(fallback,{signal});
      const [image,change]=await Promise.all([this.decodeImage(pixels.image),payload.change?this.decodeImage(payload.change):null]);
      if(epoch!==this.cellImageEpoch||signal.aborted||!image)return null;
      const accepted=payload.observation?[{identity,...payload.observation}]:[];
      const row={payload:{...pixels,change:payload.change,change_bounds:payload.change_bounds,observation:payload.observation,positions:payload.positions,layers:payload.layers||{accepted},partial_layers:true},image,change};
      this.cacheCellMedia(frame,row);return row;
    }
    queueImages(anchor){
      if(!this.data)return;clearTimeout(this.mediaPrefetchTimer);const epoch=this.cellImageEpoch;
      const frames=[];for(let distance=1;distance<=3;distance++){if(anchor+distance<this.data.frames.length)frames.push(anchor+distance);if(anchor-distance>=0)frames.push(anchor-distance);}
      this.imageQueue=frames.filter(frame=>!this.cellImageCache.has(frame));
      this.mediaPrefetchTimer=setTimeout(()=>{if(epoch===this.cellImageEpoch&&!this.mediaBusy&&!this.imageWorkers)this.prefetchWorker(epoch);},100);
    }
    async prefetchWorker(epoch){
      this.imageWorkers++;const identity=this.state.identity,controller=this.mediaBackgroundController=new AbortController();
      try{while(epoch===this.cellImageEpoch&&!controller.signal.aborted&&!this.mediaBusy&&this.imageQueue.length){const frame=this.imageQueue.shift();if(!this.cellImageCache.has(frame))await this.fetchCellMedia(frame,identity,epoch,controller.signal);}}
      catch{/* The requested frame reports source errors; look-ahead is optional. */}
      finally{if(epoch===this.cellImageEpoch)this.imageWorkers--;}
    }
    choosePeers(){
      const id=this.state.identity;
      const idx=this.data.sample_indices.reduce((best,f,i)=>Math.abs(f-this.state.frame)<Math.abs(this.data.sample_indices[best]-this.state.frame)?i:best,0);
      const origin=this.data.cells.find(c=>c.identity===id),x=origin?.values.centroid_x?.[idx],y=origin?.values.centroid_y?.[idx];
      const distance=c=>{const cx=c.values.centroid_x?.[idx],cy=c.values.centroid_y?.[idx];return C.finite(x)&&C.finite(y)&&C.finite(cx)&&C.finite(cy)?Math.hypot(cx-x,cy-y):1e6+Math.abs(c.identity-id);};
      const nearby=this.data.cells.filter(c=>c.identity!==id).sort((a,b)=>distance(a)-distance(b)).map(c=>c.identity);
      const choices=[...nearby,...this.data.identities.filter(i=>i!==id&&!nearby.includes(i))];
      const saved=this.session?.run===this.data.run&&this.session?.stem===this.data.stem&&this.session?.identity===id?this.session.controls||{}:{};
      for(const [slot,defaultChoice] of [["peer-one",choices[0]],["peer-two",choices[1]]]){
        const preferred=Number(saved["xp-"+slot]);
        const choice=this.data.identities.includes(preferred)&&preferred!==id?preferred:defaultChoice;
        const select=this.$(slot);
        for(const option of select.options)option.disabled=Number(option.value)===id;
        select.closest(".xp-cell-card").hidden=choice==null;
        if(choice!=null)select.value=String(choice);
      }
    }
    requestPeers(preserve=false){
      const epoch=preserve?this.peerEpoch:++this.peerEpoch;
      const identities=[+this.val("peer-one"),+this.val("peer-two")].filter(i=>Number.isFinite(i)&&i!==this.state.identity);
      if(!preserve){this.peerRows.clear();this.peerExact.clear();this.peerStripUnavailable.clear();}
      else for(const old of this.peerRows.keys())if(!identities.includes(old)){this.peerRows.delete(old);this.peerExact.delete(old);this.peerStripUnavailable.delete(old);}
      this.peerIdentities=[...new Set(identities)];
      const added=[];
      for(const identity of this.peerIdentities){
        if(this.peerRows.has(identity))continue;
        added.push(identity);
        this.peerRows.set(identity,new Map());
        this.options.api(this.url("cell",{identity})).then(data=>{
          if(epoch!==this.peerEpoch)return;this.peerExact.set(identity,data);this.drawCellCards();
        }).catch(()=>{});
        this.requestPeerFrame(identity,this.state.frame,epoch);
      }

    }
    async requestPeerFrame(identity,frame,epoch){
      this.peerFramePending??=new Set();const key=JSON.stringify([this.data.run,this.data.stem,identity,frame]);if(this.peerFramePending.has(key))return;this.peerFramePending.add(key);
      const controller=this.peerMediaController||(this.peerMediaController=new AbortController());
      try{const payload=await this.options.api(this.url('cell-frame',{identity,frame}),{signal:controller.signal}),image=payload.available?await this.decodeImage(payload.image):null;if(epoch!==this.peerEpoch||controller.signal.aborted)return;const cache=this.peerRows.get(identity);if(cache&&!cache.has(frame))cache.set(frame,payload.available?{payload,image,bytes:payload.image.length+image.naturalWidth*image.naturalHeight*4}:{missing:true,bytes:0});while(cache?.size>1&&(cache.size>12||[...cache.values()].reduce((sum,row)=>sum+(row.bytes||0),0)>24*1024*1024))cache.delete(cache.keys().next().value);if(frame===this.state.frame)this.drawCellCards();}
      catch{/* Peer trace still displays saved measurements. */}finally{this.peerFramePending.delete(key);}
    }
    async requestChange(frame,identity,epoch){try{const payload=await this.options.api(this.url("change",{frame,identity}));const image=payload.change?await this.decodeImage(payload.change):null;if(epoch!==this.cellImageEpoch||frame!==this.state.frame||identity!==this.state.identity)return;this.changeCache.set(frame,{payload,image});this.drawChange();}catch{/* Selected-frame source state remains visible. */}}
    loadCellDetails(){
      const details=this.pendingDetails;if(!details||details.key===this.detailsRequestKey)return;this.detailsRequestKey=details.key;
      this.detailsController?.abort();const controller=this.detailsController=new AbortController(),{epoch,identity,key}=details,frame=this.state.frame;
      Promise.allSettled([this.options.api(this.url('cell',{identity}),{signal:controller.signal}),this.options.api(this.url('history',{identity,frame}),{signal:controller.signal})]).then(requests=>{
        if(epoch!==this.cellToken||key!==this.cellRequestKey||controller.signal.aborted)return;
        if(requests[0].status==='fulfilled')this.exact=requests[0].value;
        this.history=requests[1].status==='fulfilled'?requests[1].value:null;this.historyError=requests[1].status==='rejected'?requests[1].reason.message:'';this.draw();
      });
    }
    select(patch){if(!this.data||!this.data.identities.length)return;this.state=C.selection(this.data,this.state,patch);const key=JSON.stringify([this.data.run,this.data.stem,this.state.identity]);
      if(key!==this.cellRequestKey){this.cancelMedia();this.cellRequestKey=key;this.exact=null;this.history=null;this.historyError='';this.cellImageCache.clear();this.cellImageBytes=0;this.cellImageEpoch++;this.imageWorkers=0;const epoch=++this.cellToken,identity=this.state.identity,frame=this.state.frame;
        this.changeCache.clear();this.choosePeers();this.peerEpoch++;this.peerRows.clear();this.peerExact.clear();
        this.detailsRequestKey=null;this.pendingDetails={epoch,identity,frame,key};
      }
      this.token++;const cached=this.val('image-mode')==='cell'?this.cellImageCache.get(this.state.frame):this.frameCache.get(this.state.frame);this.mediaBusy=!cached;
      this.frameData=cached?.payload||null;this.frameValues=this.frameCache.get(this.state.frame)?.values||null;this.imageMeta=cached?.payload||null;this.image=cached?.image||null;this.imageError='';
      this.$('cell-select').value=this.state.identity;this.$('frame-slider').value=this.state.frame;this.options.onSelection?.(this.state);this.drawSelection();
      clearTimeout(this.contactDebounce);
      clearTimeout(this.frameDebounce);this.frameDebounce=setTimeout(()=>this.loadSelectedFrame(),0);
    }
    async loadSelectedFrame(){
      if(!this.data)return;const frame=this.state.frame,identity=this.state.identity,epoch=this.cellImageEpoch,mode=this.val('image-mode'),request=this.mediaRequest=(this.mediaRequest||0)+1;
      this.mediaController?.abort();const controller=this.mediaController=new AbortController();
      let row=mode==='cell'?this.cellImageCache.get(frame):this.frameCache.get(frame);
      if(!row||mode==='field'&&!this.changeCache.has(frame)){this.mediaBackgroundController?.abort();this.mediaBusy=true;}
      const current=()=>request===this.mediaRequest&&epoch===this.cellImageEpoch&&frame===this.state.frame&&identity===this.state.identity&&!controller.signal.aborted;
      this.options.api(this.url('values',{frame}),{signal:controller.signal}).then(values=>{if(current()){this.frameValues=values;this.drawTrace();this.scheduleSecondary();}}).catch(()=>{});
      try{
        if(!row){
          if(mode==='cell')row=await this.fetchCellMedia(frame,identity,epoch,controller.signal);
          else {row=await this.fetchFieldMedia(frame,identity,epoch,controller.signal,current);if(current()){this.cacheImage(frame,row.payload,row.image);this.frameCache.set(frame,row);for(const old of this.frameCache.keys())if(!this.imageCache.has(old))this.frameCache.delete(old);}}
        }
        if(mode==='field'&&row&&!this.changeCache.has(frame))await this.fetchCellMedia(frame,identity,epoch,controller.signal);
        if(!current()||!row)return;this.frameData=row.payload;this.imageMeta=row.payload;this.image=row.image;this.fieldSize=[row.payload.width,row.payload.height];this.imageError='';
        this.drawImage();this.drawChange();this.drawCellCards();this.scheduleSecondary();
      }catch(error){if(current()&&error.name!=='AbortError'){this.imageError=error.message;this.drawImage();}}
      finally{if(current()){this.mediaBusy=false;this.scheduleSecondary();clearTimeout(this.contactDebounce);if(this.previewContacts)this.contactDebounce=setTimeout(()=>this.loadContacts(),250);this.loadOverview();this.loadCellDetails();this.peerMediaController?.abort();this.peerMediaController=new AbortController();this.requestPeers(true);for(const [peer,cache] of this.peerRows)if(!cache.has(frame))this.requestPeerFrame(peer,frame,this.peerEpoch);this.queueImages(frame);}}
    }
    selected(){return this.exact?.cell||this.data?.cells.find(c=>c.identity===this.state.identity);}
    vector(cell,prefix,exact=false,view=prefix){const key=C.metricKey(this.comparisons.value(prefix+"-metric",view),this.comparisons.value(prefix+"-aggregation",view),this.data),indices=exact?this.exact.sample_indices:this.data.sample_indices;return C.smooth(cell?.values[key]||indices.map(()=>null),indices,+(this.comparisons.element(prefix+"-smoothing",view)?.value||1));}
    metricLabel(prefix){const key=C.metricKey(this.val(prefix+"-metric"),this.val(prefix+"-aggregation"),this.data),m=this.data.metrics.find(m=>m.key===key);return m?`${m.label}${m.unit?` (${m.unit.replace('{length}',this.data.units?.length||'saved length unit').replace('{interval}',this.data.units?.interval||'frame')})`:""}`:"No compatible metric";}
    drawSelection(){if(!this.data)return;if(!this.data.identities.length){this.drawEmptyCohort();return;}const frame=this.data.frames[this.state.frame];this.$("time-label").textContent=`Frame ${frame.source_frame_index+1} · ${fmt(frame.hours)} h`;const cell=this.selected();const index=this.exact?this.state.frame:this.data.sample_indices.indexOf(this.state.frame);const absent=index>=0&&cell&&!cell.observed[index];this.$("selection-status").textContent=`${this.data.stem} · Cell ${this.state.identity}${absent?" · no observed mask at this time":""}`;
      for(const p of ["trace","trace-grid","barcode","distribution"])this.$(p+"-aggregation-label").hidden=this.val(p+"-metric")!=="light";
      this.drawImage();this.drawChange();this.drawHistory();this.drawTrace();this.drawCellCards();}
    drawSecondary(){if(!this.data.identities.length){this.drawEmptyCohort();this.saveSession();return;}if(this.mediaBusy)return;if(!this.data.preview.pending){this.drawBarcode();this.comparisons.draw("barcode");}else this.blank(this.$("barcode"),"Loading measurements...");this.drawEvents();this.drawClock();this.drawMap();this.comparisons.draw("tissue-map");this.drawNetwork();this.drawDistribution();this.drawTraceGrid();this.updateRhythmAnnotations();this.saveSession();}
    scheduleSecondary(){if(this.mediaBusy||this.secondaryPending)return;this.secondaryPending=requestAnimationFrame(()=>{this.secondaryPending=null;if(this.data&&!this.mediaBusy)this.drawSecondary();});}
    draw(){if(!this.data)return;if(this.secondaryPending){cancelAnimationFrame(this.secondaryPending);this.secondaryPending=null;}this.drawSelection();this.drawSecondary();}
    axes(ctx,w,h,left=65,top=18,bottom=h-35,showTimeLabels=true){ctx.strokeStyle="#2a3b4a";this.stroke(ctx,1);for(let i=0;i<4;i++){const y=top+(bottom-top)*i/3;ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(w-18,y);ctx.stroke();}ctx.fillStyle=muted;ctx.font=`${Math.max(10,this.style(ctx.canvas.id.slice(3)).fontSize-1)}px system-ui`;if(showTimeLabels){const first=this.data.frames[0],last=this.data.frames.at(-1);ctx.fillText(`${fmt(first.hours)} h`,left,h-12);ctx.textAlign="right";ctx.fillText(`${fmt(last.hours)} h`,w-18,h-12);ctx.textAlign="left";}return {left,right:w-18,top,bottom};}
    cursor(ctx,box){const x=box.left+this.fraction(this.state.frame)*(box.right-box.left);ctx.strokeStyle=gold;this.stroke(ctx,1.5);ctx.beginPath();ctx.moveTo(x,box.top);ctx.lineTo(x,box.bottom);ctx.stroke();}
    rhythmResultsVisible(){return this.rhythmAnalyses?!!this.data?.results?.rhythms?.rows?.length:this.$("rhythm-visible").checked;}
    traceRhythmColours(){return this.$("trace-rhythm-colours").checked&&this.rhythmResultsVisible();}
    savedFit(identity,metric){
      if(!this.$("trace-fit").checked)return {state:"off"};
      const row=this.rhythmForMetric(identity,metric),result=this.rhythmAnalyses?.result(row?.metric)||(!this.rhythmAnalyses?this.data.results?.rhythms:null);
      if(!row||!result||result.preview)return {state:"unavailable",message:"No saved fitted rhythm for this measurement."};
      metric=row.metric;const key=JSON.stringify([this.data.run,this.data.stem,result.analysis_id||result.source,metric]);
      this.fitCache??=new Map();let entry=this.fitCache.get(key);
      if(!entry){
        entry={state:"loading",rows:new Map()};this.fitCache.set(key,entry);
        if(this.fitCache.size>32)this.fitCache.delete(this.fitCache.keys().next().value);
        const data=this.data;
        this.options.api(this.url("rhythm-fits",{metric,analysis_id:result.analysis_id})).then(fits=>{
          entry.rows=new Map(fits.rows.map(fit=>[fit.identity,fit]));entry.state="ready";entry.message=fits.message;
        }).catch(error=>{entry.state="unavailable";entry.message=error.message;}).finally(()=>{
          if(data===this.data&&result===(this.rhythmAnalyses?.result(metric)||this.data.results?.rhythms)){this.drawTrace();this.drawCellCards();this.drawTraceGrid();}
        });
      }
      const fit=entry.rows.get(identity);
      return {state:fit?"available":entry.state==="ready"?"unavailable":entry.state,fit,message:entry.message};
    }
    fitCaption(identity,metric){
      const fit=this.savedFit(identity,metric);
      return fit.state==="off"?"":fit.state==="loading"?" · Loading saved fit…":fit.state==="available"?" · Dashed: saved model (analysis settings)":" · "+(fit.message||"No saved model for this cell/metric");
    }
    traceStyleId(){return this.plotView==="trace-grid"?"trace-grid":"trace";}
    traceColour(identity,metric,styleId=this.traceStyleId()){const row=this.rhythmForMetric(identity,metric),palette=this.data.trace_style;return this.traceRhythmColours()&&palette?palette[row?.status==="rhythmic"?"rhythmic":"other"]:this.style(styleId).accent;}
    traceRhythmLabel(row){
      if(!row)return "Not analysed";
      const probability=row.correction&&row.correction!=="none"&&C.finite(row.q_value)?`q=${fmt(row.q_value)}`:C.finite(row.p_value)?`p=${fmt(row.p_value)}`:"";
      const significant=row.status==="rhythmic",state=significant?"Significant":row.status==="not rhythmic"||row.status==="arrhythmic"?"Not rhythmic":"Not tested";
      return [significant&&C.finite(row.period_hours)?`${fmt(row.period_hours)} h${row.period_underdetermined?" (limited)":""} *`:state,probability].filter(Boolean).join(" · ");
    }
    traceInk(){const colour=this.style(this.traceStyleId()).background,rgb=[1,3,5].map(i=>parseInt(colour.slice(i,i+2),16));return rgb[0]*.2126+rgb[1]*.7152+rgb[2]*.0722>145?"#202832":"#dce5ed";}
    plotInk(id){const colour=this.style(id).background,rgb=[1,3,5].map(i=>parseInt(colour.slice(i,i+2),16));return rgb[0]*.2126+rgb[1]*.7152+rgb[2]*.0722>145?"#202832":"#dce5ed";}
    barcodeRhythmColours(view="barcode"){return this.comparisons.element("barcode-rhythm-colours",view).checked&&this.rhythmResultsVisible();}
    matrixColour(value,accent){const rgb=[1,3,5].map(i=>parseInt(accent.slice(i,i+2),16));return `rgb(${rgb.map(v=>Math.round(247+(v-247)*Math.max(0,Math.min(1,value)))).join(",")})`;}
    traceTick(v){const n=Math.abs(v);return n>=100000||n>0&&n<.01?v.toExponential(1):Number(v.toPrecision(3)).toString();}
    traceLeft(ctx,w,lanes,compact=false){ctx.font=`${Math.max(10,this.style(this.traceStyleId()).fontSize-(compact?2:1))}px system-ui`;return Math.min(Math.max(...lanes.flatMap(l=>l.range.map(v=>ctx.measureText(this.traceTick(v)).width)))+9,w*.32);}
    traceLane(ctx,w,{values,indices,identity,metric,label,top,bottom,timeLabels=true,range=null,compact=false,left:sharedLeft=null,heading=true,colour:chosenColour=null,fitted:chosenFit=null,extraSeries=[]}){
      const s=this.style(this.traceStyleId()),colour=chosenColour||this.traceColour(identity,metric),row=this.rhythmResultsVisible()?this.rhythmForMetric(identity,metric):null,ink=this.traceInk(),[lo,hi]=range||this.rangeFor(this.traceStyleId(),values);
      ctx.font=`${Math.max(10,s.fontSize-(compact?2:1))}px system-ui`;
      const left=sharedLeft??this.traceLeft(ctx,w,[{range:[lo,hi]}],compact),box={left,right:w-9,top:top+(heading?19:0),bottom},sy=v=>box.bottom-3-(v-lo)/(hi-lo)*Math.max(1,box.bottom-box.top-6);
      ctx.fillStyle=ink;ctx.textAlign="right";ctx.fillText(this.traceTick(hi),left-5,box.top+8);ctx.fillText(this.traceTick(lo),left-5,box.bottom);ctx.textAlign="left";
      if(heading){
      const title=label||"",available=w-12;ctx.save();ctx.beginPath();ctx.rect(3,top,w-6,17);ctx.clip();ctx.fillStyle=colour;ctx.font=`${row?.status==="rhythmic"?"600 ":""}${Math.max(10,s.fontSize-(compact?2:1))}px system-ui`;
      const annotation=this.rhythmResultsVisible()?this.traceRhythmLabel(row):"",joined=title&&annotation?title+" · "+annotation:title||annotation;
      if(ctx.measureText(joined).width<=available)ctx.fillText(joined,4,top+12);
      else{ctx.fillText(title||annotation,4,top+12);if(title&&annotation){const short=row?.status==="rhythmic"&&C.finite(row.period_hours)?`${fmt(row.period_hours)} h *`:row?.status==="rhythmic"?"Significant *":row?.status==="not rhythmic"||row?.status==="arrhythmic"?"Not rhythmic":row?"Not tested":"Not analysed";if(ctx.measureText(title).width+ctx.measureText(short).width+12<=available){ctx.textAlign="right";ctx.fillText(short,w-4,top+12);ctx.textAlign="left";}}}ctx.restore();
      }
      ctx.strokeStyle=ink;ctx.lineWidth=.7;ctx.beginPath();ctx.moveTo(left,box.top);ctx.lineTo(left,bottom);ctx.lineTo(box.right,bottom);ctx.stroke();
      if(timeLabels){
        const first=this.data.frames[0].hours,last=this.data.frames.at(-1).hours,span=last-first;
        let ticks=[first,last];if(C.finite(span)&&span>0){for(const step of [12,24,48,96,192,384,768]){const candidates=[];for(let t=Math.ceil(first/step)*step;t<=last;t+=step)candidates.push(t);if(candidates.length&&candidates.length<=10&&candidates.length*42<=box.right-box.left){ticks=candidates;break;}}}
        ctx.fillStyle=ink;ctx.font=`${Math.max(10,s.fontSize-2)}px system-ui`;
        for(const t of ticks){const x=left+(span>0?(t-first)/span:0)*(box.right-left),text=fmt(t),tw=ctx.measureText(text).width;ctx.strokeStyle=ink;ctx.beginPath();ctx.moveTo(x,bottom);ctx.lineTo(x,bottom+3);ctx.stroke();ctx.fillText(text,Math.max(left,Math.min(box.right-tw,x-tw/2)),bottom+14);}
        ctx.textAlign="right";ctx.fillText("h",w-1,bottom+14);ctx.textAlign="left";
      }
      const fitted=this.savedFit(identity,metric),fitValues=C.alignFit(fitted.fit,indices,values);
      ctx.canvas.dataset.rhythmFit=fitted.state;ctx.canvas.dataset.rhythmFitKind=fitted.fit?.kind||"";
      ctx.save();ctx.beginPath();ctx.rect(left,box.top,box.right-left,Math.max(1,bottom-box.top));ctx.clip();
      this.traceTimeGuides(ctx,box,ink);
      for(const series of [{values,indices,colour,fitted:chosenFit??fitValues,fitColour:extraSeries.length?colour:ink},...extraSeries])this.traceSeries(ctx,box,sy,series);
      ctx.strokeStyle=gold;ctx.lineWidth=.9;const x=left+this.fraction(this.state.frame)*(box.right-left);ctx.beginPath();ctx.moveTo(x,box.top);ctx.lineTo(x,bottom);ctx.stroke();ctx.restore();
      ctx.canvas.dataset.traceColour=colour;ctx.canvas.dataset.rhythmStatus=row?.status||"unknown";return box;
    }
    traceTimeGuides(ctx,box,ink){
      const interval=+this.val("trace-time-guides"),offset=+this.val("trace-time-guides-offset"),first=this.data.frames[0].hours,last=this.data.frames.at(-1).hours;
      const values=[];if(Number.isFinite(interval)&&interval>0&&Number.isFinite(offset)&&last>first){const begin=Math.ceil((first-offset)/interval),end=Math.floor((last-offset)/interval);if(end-begin+1<=1000)for(let i=begin;i<=end;i++)values.push(offset+i*interval);}
      ctx.canvas.dataset.timeGuides=JSON.stringify(values);ctx.save();ctx.strokeStyle=ink;ctx.globalAlpha=.32;ctx.lineWidth=.7;ctx.setLineDash([4,4]);ctx.beginPath();for(const hour of values){const x=box.left+(hour-first)/(last-first)*(box.right-box.left);ctx.moveTo(x,box.top);ctx.lineTo(x,box.bottom);}ctx.stroke();ctx.restore();
    }
    traceSeries(ctx,box,sy,{values,indices,colour,fitted=[],fitColour=colour}){
      const path=(samples,dashed=false)=>{ctx.strokeStyle=dashed?fitColour:colour;ctx.lineWidth=this.style(this.traceStyleId()).lineWidth;ctx.lineJoin="round";ctx.setLineDash(dashed?[5,3]:[]);ctx.beginPath();let previous=null;
        samples.forEach((v,i)=>{if(!C.finite(v)){previous=null;return;}const frame=indices[i],x=box.left+this.fraction(frame)*(box.right-box.left),y=sy(v);if(previous!==null&&frame===previous+1)ctx.lineTo(x,y);else ctx.moveTo(x,y);previous=frame;});ctx.stroke();ctx.setLineDash([]);};
      path(values);ctx.fillStyle=colour;
      values.forEach((v,i)=>{if(!C.finite(v))return;const before=i>0&&C.finite(values[i-1])&&indices[i-1]===indices[i]-1,after=i+1<values.length&&C.finite(values[i+1])&&indices[i+1]===indices[i]+1;if(!before&&!after){ctx.beginPath();ctx.arc(box.left+this.fraction(indices[i])*(box.right-box.left),sy(v),1.7,0,Math.PI*2);ctx.fill();}});
      if(fitted.some(C.finite))path(fitted,true);
    }
    tracePicks(kind){return Array.from(this.$(`trace-${kind}-list`).querySelectorAll("input:checked")).filter(el=>el.dataset.available!=="false").map(el=>kind==="cells"?+el.value:el.value).filter(value=>kind!=="cells"||this.inCohort(value));}
    bindGridMeasurementControls(repaint){
      const toolbar=this.$("trace-grid-panel").querySelector(".xp-grid-toolbar");
      for(const id of ["trace-measurements-picker","trace-layout","trace-overlay-scale"]){const source=this.$(id),control=(source.tagName==="SELECT"?source.closest("label"):source).cloneNode(true);for(const element of [control,...control.querySelectorAll("[id]")])if(element.id)element.id=element.id==="xp-trace-layout"?"xp-trace-grid-measurement-layout":element.id==="xp-trace-overlay-scale"?"xp-trace-grid-measurement-scale":element.id.replace("xp-trace-","xp-trace-grid-");toolbar.insertBefore(control,this.$("trace-grid-status"));}
      this.$("trace-grid-measurements-picker").onchange=()=>{const checked=new Set(Array.from(this.$("trace-grid-measurements-list").querySelectorAll("input:checked"),el=>el.value));this.$("trace-measurements-list").querySelectorAll("input").forEach(el=>el.checked=checked.has(el.value));repaint();};
      for(const [target,source] of [["trace-grid-measurement-layout","trace-layout"],["trace-grid-measurement-scale","trace-overlay-scale"]])this.$(target).onchange=()=>{this.$(source).value=this.val(target);repaint();};
    }
    setTracePicks(saved){
      const build=(kind,options,selected)=>{const list=this.$(`trace-${kind}-list`);list.replaceChildren();for(const [key,text] of options){const label=document.createElement("label"),input=document.createElement("input");input.type="checkbox";input.value=key;input.checked=selected.includes(key);label.append(input,document.createTextNode(text));list.append(label);}};
      const old=saved?.controls?.["xp-trace-secondary"],extras=saved?.traceMeasurements??(old&&old!=="none"?[old]:[]);
      const catalogue=this.measurementCatalogue(),config={selected:extras,catalogue:{...catalogue,metrics:catalogue.metrics.filter(m=>m.key!=="integrated_density_mean3")}};
      this.tracePicker=this.measurementPicker("trace-measurements-list",config);
      this.gridMeasurementPicker=this.measurementPicker("trace-grid-measurements-list",config);
      for(const prefix of ["trace","trace-grid"])window.PyMicrogliaMeasurementPicker.popup(this.$(prefix+"-measurements-picker"),this.$(prefix+"-measurements-list"));
      build("cells",this.data.identities.map(id=>[id,`Cell ${id}`]),saved?.traceCells??[this.state.identity,...this.data.identities.filter(id=>id!==this.state.identity).slice(0,2)]);
      this.syncTracePicks();
    }
    syncTracePicks(){
      const primary=C.metricKey(this.val("trace-metric"),this.val("trace-aggregation"),this.data);
      this.$("trace-measurements-list").querySelectorAll("input").forEach(input=>{input.disabled=input.dataset.available==="false"||input.value===primary;input.title=input.dataset.available==="false"?"Not measured yet":input.value===primary?"Already shown as the primary measurement":"Add this saved measurement";});
      const measurements=new Map(Array.from(this.$("trace-measurements-list").querySelectorAll("input"),input=>[input.value,input]));
      this.$("trace-grid-measurements-list").querySelectorAll("input").forEach(input=>{const source=measurements.get(input.value);input.checked=source.checked;input.disabled=source.disabled;input.title=source.title;});
      this.$("trace-measurements-count").textContent=`(${Math.max(0,this.traceMetrics().length-1)})`;this.$("trace-cells-count").textContent=`(${this.tracePicks("cells").length})`;
      this.$("trace-grid-measurements-count").textContent=this.$("trace-measurements-count").textContent;this.$("trace-grid-measurement-layout").value=this.val("trace-layout");this.$("trace-grid-measurement-scale").value=this.val("trace-overlay-scale");this.$("trace-grid-measurement-scale").disabled=this.val("trace-layout")!=="overlay";
      this.$("trace-overlay-scale").disabled=this.val("trace-layout")!=="overlay";this.$("trace-grid-overlay-scale").disabled=this.val("trace-grid-layout")!=="overlay";
      this.$("trace-grid-overlay-toggle").checked=this.val("trace-grid-layout")==="overlay";
      this.$("trace-grid-columns").disabled=this.val("trace-grid-layout")!=="grid";
      this.tracePicker?.refresh();this.gridMeasurementPicker?.refresh();
    }
    traceMetrics(){const primary=C.metricKey(this.val("trace-metric"),this.val("trace-aggregation"),this.data);return [...new Set([primary,...this.tracePicks("measurements")])].filter(Boolean);}
    traceMetricTitle(metric){const m=this.data.metrics.find(m=>m.key===metric);return m?`${m.label}${m.unit?` (${m.unit.replace('{length}',this.data.units?.length||'saved length unit').replace('{interval}',this.data.units?.interval||'frame')})`:""}`:metric;}
    cellTraceSeries(cell,indices){return this.traceMetrics().map(metric=>({identity:cell.identity,metric,indices,values:C.smooth(cell.values[metric]||indices.map(()=>null),indices,+this.val("trace-smoothing")),label:this.traceMetricTitle(metric)}));}
    drawTraceSet(ctx,w,h,series,{mode=this.val("trace-layout"),normalized=this.val("trace-overlay-scale")==="normalized",compact=false,headings=true,left:sharedLeft=null}={}){
      const overlay=mode==="overlay"&&series.length>1,s=this.style(this.traceStyleId()),prepared=series.map((item,i)=>{
        const saved=this.savedFit(item.identity,item.metric),model=item.analysis_id?(this.$("trace-fit").checked?item.fitted:[]):C.alignFit(saved.fit,item.indices,item.values),scaled=C.scaleSeries(item.values,model,overlay&&normalized);
        return {...item,...scaled,colour:overlay?C.seriesColours[i%C.seriesColours.length]:this.traceColour(item.identity,item.metric),range:!overlay&&item.range?this.rangeFor(this.traceStyleId(),item.range):!overlay?this.rangeFor(this.traceStyleId(),item.values):scaled.range};
      });
      ctx.canvas.dataset.traceLayout=overlay?"overlay":"stack";ctx.canvas.dataset.traceSeries=JSON.stringify(prepared.map(p=>({identity:p.identity,metric:p.metric,colour:p.colour,fit:this.savedFit(p.identity,p.metric).state})));
      ctx.canvas.dataset.traceScale=overlay&&normalized?"each trace 0–1":"original units";
      ctx.canvas.title=prepared.map(p=>`${p.label}: ${this.traceRhythmLabel(this.rhythmForMetric(p.identity,p.metric))}`).join("\n");
      ctx.canvas.traceHitSeries=[];ctx.canvas.traceLegendIdentities=overlay?prepared.map(p=>p.identity):[];
      if(!prepared.length){ctx.fillStyle=this.traceInk();ctx.fillText("Excluded for the displayed measurements",8,35);return [];}
      const left=sharedLeft??this.traceLeft(ctx,w,prepared,compact),boxes=[];
      if(overlay){
        ctx.font=`${Math.max(10,s.fontSize-(compact?2:1))}px system-ui`;
        prepared.forEach((p,i)=>{const row=this.rhythmResultsVisible()?this.rhythmForMetric(p.identity,p.metric):null,title=p.label+(row?.status==="rhythmic"?` · ${fmt(row.period_hours)} h *`:row?" · "+this.traceRhythmLabel(row):"");let text=title;while(ctx.measureText(text).width>w-25&&text.length>2)text=text.slice(0,-2);if(text!==title)text+="…";ctx.fillStyle=p.colour;ctx.fillRect(4,i*16+8,10,2);ctx.fillText(text,19,i*16+13);});
        const range=normalized?[0,1]:C.range(prepared.flatMap(p=>p.values)),top=prepared.length*16+4;
        const box=this.traceLane(ctx,w,{...prepared[0],top,bottom:h-22,left,range,heading:false,compact,extraSeries:prepared.slice(1)});boxes.push(box);ctx.canvas.traceHitSeries=prepared.map(p=>({...p,box,range}));
      }else for(let i=0;i<prepared.length;i++){const top=i*(h-23)/prepared.length,bottom=(i+1)*(h-23)/prepared.length-3;boxes.push({...this.traceLane(ctx,w,{...prepared[i],top,bottom,left,timeLabels:i===prepared.length-1,compact,heading:headings}),identity:prepared[i].identity});}
      ctx.canvas.dataset.rhythmFit=this.savedFit(prepared[0].identity,prepared[0].metric).state;
      return boxes;
    }
    drawTrace(){
      const cell=this.selected();if(!cell){this.blank(this.$("trace"),"Loading exact selected-cell measurements…");return;}
      this.sizeCardTraces();const [ctx,w,h]=this.canvas("trace"),exact=!!this.exact,indices=exact?this.exact.sample_indices:this.data.sample_indices,series=this.cellTraceSeries(cell,indices),key=series[0]?.metric;
      this.hits.trace=this.drawTraceSet(ctx,w,h,series)[0];
      ctx.canvas.dataset.traceColour=this.traceColour(this.state.identity,key);ctx.canvas.dataset.rhythmStatus=this.rhythmResultsVisible()?this.rhythmForMetric(this.state.identity,key)?.status||"unknown":"unknown";
      this.$("trace-caption").textContent=`${exact?"Exact saved frames":"Sampled overview; exact trace loading"} · ${series.length===1?this.metricLabel("trace"):series.length+" measurements"} · ${this.val("trace-smoothing")==="1"?"raw":this.val("trace-smoothing")+" consecutive-frame display mean"}${this.val("trace-layout")==="overlay"&&series.length>1?" · "+ctx.canvas.dataset.traceScale:""}${this.fitCaption(this.state.identity,key)}`;
    }
    sizeCardTraces(){
      const count=this.traceMetrics().length,stack=this.val("trace-layout")==="stack",height=stack&&count>2?count*85:Math.max(170,this.style("trace").height,this.val("trace-layout")==="overlay"?count*16+110:0);
      for(const id of ["trace","peer-one-trace","peer-two-trace"]){const canvas=this.$(id),body=canvas.closest(".xp-card-body");body.classList.toggle("is-multi-stack",height>210);if(count>1||this.styles.trace?.height)canvas.dataset.traceHeight=height;else{delete canvas.dataset.traceHeight;canvas.style.removeProperty("height");}}
    }
    syncTraceGridControls(origin="trace"){
      const target=origin==="trace"?"trace-grid":"trace";
      for(const setting of ["metric","aggregation","smoothing"])this.$(target+"-"+setting).value=this.val(origin+"-"+setting);
      this.$(target+"-fit").checked=this.$(origin+"-fit").checked;
      this.syncMetricPickers();
    }
    resetTraceGrid(){
      this.gridAbort?.abort();this.gridObserver?.disconnect();
      this.gridRequestKey=null;this.gridRows=new Map();this.gridTiles=new Map();this.gridVisible=new Set();this.gridPreparedKey=null;this.gridPrepared=new Map();
      this.$("trace-grid")?.replaceChildren();if(this.$("trace-grid-status"))this.$("trace-grid-status").textContent="";
      if(this.$("trace-grid-caption"))this.$("trace-grid-caption").textContent="";
    }
    createTraceGrid(){
      if(this.gridTiles.size)return;
      const container=this.$("trace-grid"),fragment=document.createDocumentFragment();
      if(window.IntersectionObserver)this.gridObserver=new IntersectionObserver(entries=>{
        for(const entry of entries){const id=Number(entry.target.dataset.identity);if(entry.isIntersecting){this.gridVisible.add(id);this.drawTraceGridCell(id);}else this.gridVisible.delete(id);}
      },{rootMargin:"300px 0px"});
      for(const identity of this.sourceData.identities){
        const tile=document.createElement("div");tile.className="xp-grid-cell";tile.dataset.identity=identity;
        const button=document.createElement("button");button.type="button";button.textContent=`Cell ${identity}`;button.title=`Select cell ${identity} in every linked view`;button.onclick=()=>this.select({identity});
        const heading=document.createElement("div");heading.className="xp-grid-heading";const annotation=document.createElement("span");annotation.className="xp-grid-rhythm";annotation.hidden=true;heading.append(button,annotation);
        const canvas=document.createElement("canvas");canvas.style.height="120px";canvas.dataset.identity=identity;canvas.setAttribute("aria-label",`Measurement trace for cell ${identity}`);canvas.title=`Cell ${identity}: click to select a time`;canvas.tabIndex=0;
        canvas.onclick=event=>{const box=canvas.getBoundingClientRect(),x=(event.clientX-box.left)*Number(canvas.dataset.logicalWidth)/box.width,plot=canvas.gridBox;if(!plot)return;this.select({identity,frame:this.nearestFrame(Math.max(0,Math.min(1,(x-plot.left)/(plot.right-plot.left))))});};
        canvas.onkeydown=event=>{if(event.key==="Enter"||event.key===" "){this.select({identity});event.preventDefault();}else if(["ArrowLeft","ArrowRight"].includes(event.key)){this.select({identity,frame:this.state.frame+(event.key==="ArrowLeft"?-1:1)});event.preventDefault();}};
        const note=document.createElement("span");note.className="xp-rhythm-note";note.hidden=true;note.dataset.rhythmIdentity=identity;
        tile.append(heading,canvas,note);fragment.append(tile);this.gridTiles.set(identity,{tile,canvas,note,button,annotation});
        if(!this.gridObserver)this.gridVisible.add(identity);
      }
      container.append(fragment);for(const {canvas} of this.gridTiles.values())this.gridObserver?.observe(canvas);
    }
    async loadTraceGrid(metric,key){
      this.gridAbort?.abort();const abort=new AbortController();this.gridAbort=abort;
      const data=this.data;this.gridRequestKey=key;this.gridRows.clear();this.gridPreparedKey=null;
      this.$("trace-grid-status").textContent=`Loading ${data.identities.length} cell traces...`;
      try{
        let offset=0;
        do{
          const page=await this.options.api(this.url("traces",{metric,metrics:this.traceMetrics().join(","),offset}),{signal:abort.signal});
          if(data!==this.data||key!==this.gridRequestKey||abort.signal.aborted)return;
          for(const row of page.cells)this.gridRows.set(row.identity,row);
          offset=page.next_offset;this.gridPreparedKey=null;
          this.$("trace-grid-status").textContent=`${this.gridRows.size} / ${page.total_cells} cells${offset!=null?" · loading...":""}`;
          this.drawTraceGrid();
        }while(offset!=null);
      }catch(error){if(data!==this.data||key!==this.gridRequestKey||abort.signal.aborted)return;this.$("trace-grid-status").textContent=error.message;}
    }
    prepareTraceGrid(){
      const key=JSON.stringify([this.gridRequestKey,this.val("trace-smoothing"),this.val("trace-values"),this.rhythmAnalyses?.revision,this.plotView]);if(key===this.gridPreparedKey)return;
      this.gridPreparedKey=key;this.gridPrepared.clear();this.gridRanges=new Map();
      for(const [identity,row] of this.gridRows){
        const series=this.cellTraceSeries({identity,values:row.values_by_metric||{[this.traceMetrics()[0]]:row.values}},row.indices);
        this.gridPrepared.set(identity,{series,hasValues:series.some(p=>p.values.some(C.finite))});
        for(const item of series){const r=C.range(item.values),old=this.gridRanges.get(item.metric);if(item.values.some(C.finite))this.gridRanges.set(item.metric,old?[Math.min(old[0],r[0]),Math.max(old[1],r[1])]:r);}
      }
    }
    drawTraceGrid(){
      if(!this.data)return;
      const metrics=this.traceMetrics(),key=JSON.stringify([this.data.run,this.data.stem,metrics]);
      this.createTraceGrid();if(key!==this.gridRequestKey)this.loadTraceGrid(metrics[0],key);
      const layout=this.val("trace-grid-layout"),comparison=layout==="overlay",columns=this.val("trace-grid-columns");this.$("trace-grid").hidden=comparison;this.$("trace-comparison").hidden=!comparison;
      this.$("trace-grid").style.gridTemplateColumns=columns==="auto"?"":`repeat(${columns},minmax(0,1fr))`;
      this.prepareTraceGrid();this.syncTracePicks();
      for(const [identity,{tile}] of this.gridTiles){tile.classList.toggle("is-selected",identity===this.state.identity);tile.hidden=!this.inCohort(identity);}
      this.$("trace-grid-caption").textContent=`${metrics.length===1?this.metricLabel("trace"):metrics.length+" measurements"} · ${this.val("trace-smoothing")==="1"?"raw":this.val("trace-smoothing")+" consecutive-frame display mean"} · exact saved frames; gaps stay missing${comparison&&this.val("trace-grid-layout")==="overlay"?" · "+(this.val("trace-grid-overlay-scale")==="normalized"?"Each trace 0–1":"Original units"):""}${this.$("trace-fit").checked?" · Dashed: available saved models (analysis settings)":""}`;
      if(comparison)this.drawTraceComparison();else{for(const [identity,{canvas,tile}] of this.gridTiles){if(tile.hidden)continue;const rect=canvas.getBoundingClientRect();if(rect.bottom>=-300&&rect.top<=innerHeight+300)this.gridVisible.add(identity);}for(const identity of this.gridVisible)this.drawTraceGridCell(identity);}
    }
    gridCanvas(canvas,height){
      const w=Math.max(1,Math.round(canvas.getBoundingClientRect().width)),h=height(w),ratio=window.devicePixelRatio||1,s=this.style("trace-grid");
      canvas.style.height=`${h}px`;canvas.dataset.logicalWidth=w;canvas.dataset.logicalHeight=h;
      if(canvas.width!==Math.round(w*ratio))canvas.width=Math.round(w*ratio);if(canvas.height!==Math.round(h*ratio))canvas.height=Math.round(h*ratio);
      const ctx=canvas.getContext("2d");ctx.setTransform(ratio,0,0,ratio,0,0);ctx.fillStyle=s.background;ctx.fillRect(0,0,w,h);ctx.font=`${Math.max(10,s.fontSize-2)}px system-ui`;return [ctx,w,h];
    }
    drawTraceGridCell(identity){
      if(!this.data||!this.gridTiles.has(identity)||this.val("trace-grid-layout")==="overlay"||this.gridTiles.get(identity).tile.hidden)return;
      this.prepareTraceGrid();const {canvas}=this.gridTiles.get(identity),row=this.gridPrepared.get(identity),count=this.traceMetrics().length;
      const [ctx,w,h]=this.gridCanvas(canvas,w=>this.val("trace-layout")==="stack"?Math.max(Number(this.styles["trace-grid"]?.height)||Math.max(92,Math.min(145,Math.round(w*.43))),count>1?count*85:0):Math.max(Number(this.styles["trace-grid"]?.height)||120,count*16+90));
      if(!row){ctx.fillStyle=muted;ctx.fillText("Loading saved trace…",10,h/2,w-20);return;}
      const series=row.series.map(p=>({...p,range:this.val("trace-grid-scale")==="recording"?this.gridRanges.get(p.metric):null}));
      const singleColumn=this.val("trace-grid-columns")==="1",left=singleColumn?this.traceLeft(ctx,w,this.data.identities.flatMap(id=>(this.gridPrepared.get(id)?.series||[]).map(p=>({range:this.val("trace-grid-scale")==="recording"?this.gridRanges.get(p.metric)||[0,1]:this.rangeFor("trace-grid",p.values)}))),true):null;
      canvas.gridBox=this.drawTraceSet(ctx,w,h,series,{compact:true,headings:count>1,left})[0];
      if(!row.hasValues){ctx.fillStyle=this.traceInk();ctx.textAlign="center";ctx.fillText(this.val("trace-smoothing")==="1"?"No measured values":`No complete ${this.val("trace-smoothing")}-frame mean`,w/2,h/2,w-30);ctx.textAlign="left";}
    }
    drawTraceComparison(){
      const canvas=this.$("trace-comparison-canvas"),identities=this.tracePicks("cells"),series=identities.flatMap(identity=>(this.gridPrepared.get(identity)?.series||[]).map(p=>({...p,label:`Cell ${identity} · ${this.traceMetricTitle(p.metric)}`,range:this.val("trace-grid-scale")==="recording"?this.gridRanges.get(p.metric):null}))),mode=this.val("trace-grid-layout");
      const [ctx,w,h]=this.gridCanvas(canvas,()=>Math.max(280,Number(this.styles["trace-grid"]?.height)||0,series.length*16+150));
      canvas.traceBoxes=this.drawTraceSet(ctx,w,h,series,{mode,normalized:this.val("trace-grid-overlay-scale")==="normalized"});
      if(!series.length){ctx.fillStyle=this.traceInk();ctx.fillText(identities.length?"Loading saved traces…":"Choose cells in Compare cells",12,40);}
    }
    selectTraceComparison(event){
      const canvas=event.currentTarget,rect=canvas.getBoundingClientRect(),x=(event.clientX-rect.left)*Number(canvas.dataset.logicalWidth)/rect.width,y=(event.clientY-rect.top)*Number(canvas.dataset.logicalHeight)/rect.height,legend=canvas.traceLegendIdentities||[];
      if(y<legend.length*16){this.select({identity:legend[Math.floor(y/16)]});return;}
      const box=canvas.traceBoxes?.[0];if(!box)return;
      const frame=this.nearestFrame(Math.max(0,Math.min(1,(x-box.left)/(box.right-box.left))));let closest=null,distance=12;
      for(const series of canvas.traceHitSeries||[]){const index=series.indices.indexOf(frame),value=series.values[index];if(!C.finite(value))continue;const [lo,hi]=series.range,position=box.bottom-3-(value-lo)/(hi-lo)*Math.max(1,box.bottom-box.top-6),delta=Math.abs(y-position);if(delta<distance){closest=series.identity;distance=delta;}}
      this.select({frame,...(closest===null?{}:{identity:closest})});
    }
    rhythmRows(metric=this.val("clock-metric")){return (this.data.results?.rhythms?.rows||[]).filter(r=>r.metric===metric&&this.inCohort(r.identity));}
    phaseFor(identity,metric=this.val("clock-metric")){return this.rhythmRows(metric).find(r=>r.identity===identity)?.phase_fraction;}
    barcodeOrderOptions(view="barcode"){return {engine:this.comparisons.value("barcode-order",view),metric:C.metricKey(this.comparisons.value("barcode-metric",view),this.comparisons.value("barcode-aggregation",view),this.data),time_mean:+this.comparisons.value("barcode-smoothing",view),scale:this.comparisons.value("barcode-scale",view),...(this.cohortIds?{identities:this.data.identities.join(",")}:{})};}
    barcodeOrderKey(view="barcode"){return JSON.stringify(this.barcodeOrderOptions(view));}
    barcodeRows(view="barcode"){
      const cells=this.data.cells.slice(),options=this.barcodeOrderOptions(view),engine=options.engine;
      this.barcodeOrderStatus="";
      if(engine.startsWith("phase:"))cells.sort((a,b)=>(this.phaseFor(a.identity,engine.slice(6))??2)-(this.phaseFor(b.identity,engine.slice(6))??2));
      else if(engine==="group")cells.sort((a,b)=>a.group.localeCompare(b.group));
      else if((this.data.barcode_ordering?.engines||[]).some(e=>e.key===engine)){
        const key=this.barcodeOrderKey(view),data=this.data;let entry=this.barcodeOrderCache.get(key);
        if(!entry){
          entry={pending:true};this.barcodeOrderCache.set(key,entry);
          if(this.barcodeOrderCache.size>32)this.barcodeOrderCache.delete(this.barcodeOrderCache.keys().next().value);
          const expected=new Set(cells.map(c=>c.identity));
          this.options.api(this.url("barcode-order",options)).then(result=>{
            if(result.identities.length!==expected.size||new Set(result.identities).size!==expected.size||result.identities.some(id=>!expected.has(id)))throw new Error("The ordering result does not match the displayed cells.");
            entry.identities=result.identities;entry.pending=false;
          }).catch(error=>{entry.error=error.message;entry.pending=false;}).finally(()=>{if(data===this.data&&this.comparisons.views.has(view))this.drawBarcode(view);});
        }
        if(entry.identities){const byIdentity=new Map(cells.map(c=>[c.identity,c]));return entry.identities.map(id=>byIdentity.get(id));}
        this.barcodeOrderStatus=entry.error?`Ordering unavailable: ${entry.error} Showing cell identity order.`:"Organising rows...";
      }
      return cells;
    }
    drawBarcode(view="barcode"){
      if(!this.data?.identities.length){this.blank(this.comparisons.element("barcode",view),"No cells match this filter.");return;}
      const [ctx,w,h]=this.canvas(view),cells=this.barcodeRows(view);
      const ink=this.plotInk(view),rhythmColours=this.barcodeRhythmColours(view),metric=C.metricKey(this.comparisons.value("barcode-metric",view),this.comparisons.value("barcode-aggregation",view),this.data),palette=this.data.trace_style;
      const box={left:48,right:w-12,top:15,bottom:h-28},rh=(box.bottom-box.top)/Math.max(1,cells.length),all=cells.map(c=>this.vector(c,"barcode",false,view)),global=C.range(all.flat()),labelStep=Math.max(1,Math.ceil((this.style(view).fontSize+3)/rh)),selected=cells.findIndex(c=>c.identity===this.state.identity);
      ctx.imageSmoothingEnabled=false;
      cells.forEach((cell,row)=>{
        const values=all[row],[lo,hi]=this.comparisons.value("barcode-scale",view)==="cell"?C.range(values):global;
        const tested=this.rhythmForMetric(cell.identity,this.rhythmAnalyses?.metricFor(view)||metric),significant=rhythmColours&&tested?.status==="rhythmic",colour=rhythmColours&&palette?palette[significant?"rhythmic":"other"]:this.accent(ctx);
        values.forEach((v,i)=>{if(!C.finite(v))return;
          const f=this.data.sample_indices[i],prev=this.data.sample_indices[i-1],next=this.data.sample_indices[i+1],at=this.fraction(f),left=prev==null?0:(this.fraction(prev)+at)/2,right=next==null?1:(at+this.fraction(next))/2;
          const x=box.left+left*(box.right-box.left),end=box.left+right*(box.right-box.left),y=box.top+row*rh;
          ctx.fillStyle=this.matrixColour((v-lo)/(hi-lo),colour);ctx.fillRect(x,y,end-x,rh-1/(window.devicePixelRatio||1));
        });
        ctx.fillStyle=colour;ctx.fillRect(box.left-5,box.top+row*rh,3,Math.max(1,rh-1));
        if(row===selected||(row%labelStep===0&&Math.abs(row-selected)>=labelStep)){
          ctx.fillStyle=rhythmColours?colour:ink;ctx.textBaseline="middle";ctx.fillText(String(cell.identity)+(significant?" *":""),5,box.top+(row+.5)*rh);ctx.textBaseline="alphabetic";
        }
        if(row===selected){ctx.strokeStyle=ink;ctx.lineWidth=1;ctx.strokeRect(box.left,box.top+row*rh,box.right-box.left,rh);}
      });
      ctx.fillStyle=ink;ctx.fillText(`${fmt(this.data.frames[0].hours)} h`,box.left,h-9);ctx.textAlign="right";ctx.fillText(`${fmt(this.data.frames.at(-1).hours)} h`,box.right,h-9);ctx.textAlign="left";
      this.cursor(ctx,box);this.hits[view]={...box,rowHeight:rh,cells};
      this.comparisons.element("barcode",view).dataset.rhythmColours=String(rhythmColours);
      const key=this.comparisons.element("barcode-panel",view).querySelector(".colour-key"),ramp=key.querySelector("i"),note=key.querySelector(".muted");
      ramp.style.background=`linear-gradient(90deg,#f7f7f7,${rhythmColours&&palette?palette.rhythmic:this.accent(ctx)})`;
      note.textContent=rhythmColours?"* Red = significant rhythm · grey = other / not tested · blank = missing":"Blank = missing observation";
      const selectionKey=JSON.stringify([this.state.identity,this.barcodeOrderKey(view),cells.map(c=>c.identity)]),scroll=this.comparisons.element("barcode",view).parentElement;
      if(selectionKey!==this.comparisons.views.get(view).selectionKey&&selected>=0){this.comparisons.views.get(view).selectionKey=selectionKey;const y=box.top+selected*rh;if(y<scroll.scrollTop)scroll.scrollTop=Math.max(0,y-10);else if(y+rh>scroll.scrollTop+scroll.clientHeight)scroll.scrollTop=y+rh-scroll.clientHeight+10;}
      const orderLabel=this.comparisons.element("barcode-order",view).selectedOptions[0]?.textContent||"Cell identity";
      this.comparisons.element("barcode-caption",view).textContent=(this.barcodeOrderStatus||orderLabel)+" · "+this.data.preview.message+(this.data.preview.stride!==1?" Barcode values are sampled; temporal means are available on the exact trace.":"");
    }
    drawCellEvents(){const [ctx,w,h]=this.canvas("events"),box=this.axes(ctx,w,h,95,22,h-35),cell=this.selected(),exact=!!this.exact,indices=exact?this.exact.sample_indices:this.data.sample_indices;ctx.fillStyle=muted;ctx.fillText("Observed",8,53);if(cell)cell.observed.forEach((seen,i)=>{const x=box.left+this.fraction(indices[i])*(box.right-box.left);ctx.fillStyle=seen?this.accent(ctx):"#d08488";ctx.fillRect(x,40,Math.max(2,(box.right-box.left)/indices.length),15);});if(this.val("event-mode")!=="observations"){for(const event of this.data.results?.events?.rows||[]){if(event.identity!=null&&event.identity!==this.state.identity)continue;if(this.val("event-mode")==="states"&&event.source!=="states")continue;const start=event.frame_index??event.start_frame;if(start==null)continue;const end=event.end_frame??start;ctx.fillStyle=gold;const x=box.left+this.fraction(Math.min(start,this.data.frames.length-1))*(box.right-box.left),right=box.left+this.fraction(Math.min(end,this.data.frames.length-1))*(box.right-box.left);ctx.fillRect(x,95,Math.max(3,right-x),18);ctx.fillText(event.label||event.event||"Saved event",x+4,89);}for(const win of this.data.windows||[]){const a=win.frames?.[0],b=win.frames?.[1];if(a==null||b==null)continue;ctx.fillStyle="#495b6c";const x=box.left+this.fraction(a)*(box.right-box.left),right=box.left+this.fraction(Math.min(b,this.data.frames.length-1))*(box.right-box.left);ctx.fillRect(x,145,right-x,12);ctx.fillStyle=muted;ctx.fillText(win.name,x,139);}}this.cursor(ctx,box);this.hits.events=box;}
    drawClock(){
      const [ctx,w,h]=this.canvas("phase-clock"),rows=this.rhythmRows(),result=this.data.results?.rhythms;
      this.hits["phase-clock"]=[];this.$("clock-caption").textContent=result?.meaning||"";
      if(!rows.length){this.blank(this.$("phase-clock"),result?.message||"No available phases.");return;}
      const cx=w/2,cy=h/2,r=Math.max(30,Math.min(w/2-28,h/2-30));
      ctx.strokeStyle="#385062";this.stroke(ctx,1);ctx.beginPath();ctx.arc(cx,cy,r,0,Math.PI*2);ctx.stroke();ctx.fillStyle=muted;ctx.textAlign="center";ctx.fillText("0 / 1 cycle",cx,15);ctx.fillText("\u00bd",cx,h-9);
      for(const row of rows){const cell=this.data.cells.find(c=>c.identity===row.identity);if(this.val("clock-scope")==="group"&&cell?.group!==this.selected()?.group)continue;if(!C.finite(row.phase_fraction))continue;
        const angle=row.phase_fraction*Math.PI*2-Math.PI/2,x=cx+Math.cos(angle)*r,y=cy+Math.sin(angle)*r;ctx.fillStyle=row.identity===this.state.identity?gold:this.accent(ctx);ctx.beginPath();ctx.arc(x,y,row.identity===this.state.identity?7:4,0,Math.PI*2);ctx.fill();this.hits["phase-clock"].push({x,y,identity:row.identity});
      }
      const row=rows.find(r=>r.identity===this.state.identity);ctx.fillStyle=white;ctx.fillText(`Cell ${this.state.identity}`,cx,cy-4);ctx.fillStyle=muted;ctx.fillText(row?`${fmt(row.period_hours)} h | ${fmt(row.phase_fraction)} cycle`:"No available phase",cx,cy+18);ctx.textAlign="left";
    }
    async loadTimeline(){
      const data=this.data,key=JSON.stringify([data.run,data.stem,this.val("timeline-order")]);if(key===this.timelineKey)return;this.timelineKey=key;const epoch=++this.timelineEpoch;
      try{const result=await this.options.api(this.url("timeline",{order:this.val("timeline-order")}));if(data!==this.data||epoch!==this.timelineEpoch)return;this.timeline=result;this.timelineError="";this.drawEvents();}catch(error){if(data!==this.data||epoch!==this.timelineEpoch)return;this.timelineError=error.message;this.drawEvents();}
    }
    drawGantt(){
      this.loadTimeline();const [ctx,w,h]=this.canvas("events"),ink=this.plotInk("events"),cells=(this.timeline?.cells||[]).filter(cell=>this.inCohort(cell.identity));this.hits.events=null;
      if(!cells.length){this.blank(this.$("events"),!this.data.identities.length?"No cells match this plot's exclusions.":this.timelineError||"Loading all cell track histories...");this.$("events-caption").textContent="";return;}
      const box={left:65,right:w-12,top:15,bottom:h-29},rowHeight=(box.bottom-box.top)/cells.length,s=this.style("events");ctx.font=`${Math.min(12,s.fontSize)}px system-ui`;
      for(const [row,cell] of cells.entries()){
        const rowTop=box.top+row*rowHeight,y=rowTop+rowHeight/2,sx=frame=>box.left+this.fraction(frame)*(box.right-box.left);
        if(cell.identity===this.state.identity){ctx.fillStyle="#e6f1f0";ctx.fillRect(0,box.top+row*rowHeight,w,rowHeight);}
        ctx.fillStyle=ink;ctx.textBaseline="middle";ctx.fillText(`Cell ${cell.identity}`,4,y);ctx.textBaseline="alphabetic";
        ctx.strokeStyle=this.accent(ctx);ctx.lineWidth=1;ctx.setLineDash([3,3]);ctx.beginPath();ctx.moveTo(sx(cell.start_frame),y);ctx.lineTo(sx(cell.end_frame),y);ctx.stroke();ctx.setLineDash([]);
        for(const [start,end] of cell.segments){ctx.fillStyle=this.accent(ctx);const left=start===0?box.left:(sx(start-1)+sx(start))/2,right=end===this.data.frames.length-1?box.right:(sx(end)+sx(end+1))/2;ctx.fillRect(left,rowTop,Math.max(2,right-left),rowHeight);}
        for(const gap of cell.gaps||[]){
          const left=(sx(gap.start_frame-1)+sx(gap.start_frame))/2,right=(sx(gap.end_frame)+sx(gap.end_frame+1))/2;
          ctx.fillStyle=this.timeline.gap_legend[gap.reason].colour;ctx.fillRect(left,rowTop,Math.max(1,right-left),rowHeight);
          ctx.save();ctx.beginPath();ctx.rect(left,rowTop,Math.max(1,right-left),rowHeight);ctx.clip();ctx.strokeStyle=ink;ctx.lineWidth=.5;for(let x=left-rowHeight;x<right+rowHeight;x+=5){ctx.beginPath();ctx.moveTo(x,rowTop+rowHeight);ctx.lineTo(x+rowHeight,rowTop);ctx.stroke();}ctx.restore();
        }
        if(cell.termination_current&&String(cell.termination?.silent_nonborder_ending).toLowerCase()==="true"){const x=sx(cell.end_frame);ctx.strokeStyle=ink;ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(x-3,y-4);ctx.lineTo(x+3,y+4);ctx.moveTo(x-3,y+4);ctx.lineTo(x+3,y-4);ctx.stroke();}
      }
      ctx.strokeStyle=ink;ctx.lineWidth=.7;ctx.beginPath();ctx.moveTo(box.left,box.top);ctx.lineTo(box.left,box.bottom);ctx.lineTo(box.right,box.bottom);ctx.stroke();ctx.fillStyle=ink;ctx.fillText(`${fmt(this.data.frames[0].hours)} h`,box.left,h-8);ctx.textAlign="right";ctx.fillText(`${fmt(this.data.frames.at(-1).hours)} h`,box.right,h-8);ctx.textAlign="left";
      this.cursor(ctx,box);this.hits.events={...box,rowHeight,cells};this.$("events-caption").textContent=this.timelineError||[this.timeline.meaning,this.timeline.history_message].filter(Boolean).join(" ");
      const axis=this.$("timeline-axis"),first=this.data.frames[0].hours,last=this.data.frames.at(-1).hours;axis.style.width=`${w}px`;axis.innerHTML=Array.from({length:5},(_,i)=>`<span>${fmt(first+(last-first)*i/4)} h</span>`).join("");
      const items={observed:{label:"Observed mask",colour:this.accent(ctx)},...this.timeline.gap_legend};this.$("timeline-legend").innerHTML=Object.entries(items).map(([key,item])=>`<span data-reason="${escape(key)}"><i style="background:${escape(item.colour)}" class="${key!=="observed"?"hatched":""}"></i>${escape(item.label)}</span>`).join("");this.timelineDetail();
      const scroll=this.$("events").parentElement,index=cells.findIndex(c=>c.identity===this.state.identity),selectionKey=JSON.stringify([this.state.identity,this.timelineKey]);if(index>=0&&selectionKey!==this.timelineSelectionKey){this.timelineSelectionKey=selectionKey;const y=box.top+index*rowHeight;if(y<scroll.scrollTop)scroll.scrollTop=Math.max(0,y-5);else if(y+rowHeight>scroll.scrollTop+scroll.clientHeight)scroll.scrollTop=y+rowHeight-scroll.clientHeight+5;}
    }
    timelineDetail(cell=null,frame=this.state.frame){
      if(this.val("event-mode")!=="gantt"||!this.timeline)return;
      cell=cell||this.timeline.cells.find(c=>c.identity===this.state.identity);if(!cell)return;
      const gap=cell.gaps?.find(g=>frame>=g.start_frame&&frame<=g.end_frame),clock=this.data.frames[frame];if(!clock)return;
      let label=gap?(gap.labels.length?gap.labels.join("; "):"Gap with no recorded explanation"):frame<cell.start_frame?"Before first observed mask":frame>cell.end_frame?"After last observed mask":"Observed mask";
      const start=gap?.start_frame??frame,end=gap?.end_frame??frame,a=this.data.frames[start],b=this.data.frames[end],parts=[`Cell ${cell.identity} · source frame ${clock.source_frame_index+1} · ${fmt(clock.hours)} h`,label];
      if(gap){parts.push(`Gap: source frames ${a.source_frame_index+1}–${b.source_frame_index+1} (${fmt(a.hours)}–${fmt(b.hours)} h)`);if(gap.hosts.length)parts.push(`Recorded host: ${gap.hosts.map(id=>`Cell ${id}`).join(", ")}`);if(gap.prior_owners.length||gap.next_owners.length)parts.push(`Recorded owners before/after: ${gap.prior_owners.join(", ")||"none"} / ${gap.next_owners.join(", ")||"none"}`);}
      if(frame>=cell.end_frame&&cell.termination_current&&cell.termination?.termination_mechanism)parts.push(`Recorded ending: ${String(cell.termination.termination_mechanism).replaceAll("_"," ")}`);
      if(frame===cell.start_frame&&cell.start_event!=="first_observed")parts.push(`Saved start: ${cell.start_event.replaceAll("_"," ")} (${cell.start_status||"status not recorded"})`);
      if(frame>=cell.end_frame&&cell.end_event!=="last_observed")parts.push(`Saved end: ${cell.end_event.replaceAll("_"," ")} (${cell.end_status||"status not recorded"})`);
      this.$("timeline-detail").textContent=parts.join(" · ");this.$("events").title=parts.join("\n");
      this.$("timeline-detail").dataset.identity=cell.identity;this.$("timeline-detail").dataset.frame=frame;this.$("timeline-detail").dataset.reason=gap?.reason||"observed_or_outside";
    }
    drawEvents(){const gantt=this.val("event-mode")==="gantt";for(const id of ["timeline-axis","timeline-legend","timeline-detail"])this.$(id).hidden=!gantt;if(gantt){this.drawGantt();return;}this.$("events-caption").textContent="Selected cell: saved events and observation gaps";this.drawCellEvents();}
    tissueKey(view="tissue-map"){return JSON.stringify([this.comparisons.value("map-layer",view),this.comparisons.value("tectonics-metric",view),this.data.results?.rhythms?.analysis_id||this.data.results?.rhythms?.source,this.cohortIds?this.data.identities:null]);}
    async loadTissueMap(key,view="tissue-map"){
      if(this.tissueCache.has(key))return;const data=this.data,layer=this.comparisons.value("map-layer",view).slice("tectonics:".length),metric=this.comparisons.value("tectonics-metric",view),entry={};this.tissueCache.set(key,entry);
      try{const payload=await this.options.api(this.url("tissue",{layer,...(metric?{metric,analysis_id:this.rhythmAnalyses?.result(metric)?.analysis_id||"00000000000000000000000000000000"}:{}),...(this.cohortIds?{identities:this.data.identities.join(",")}:{})}));if(data!==this.data)return;entry.payload=payload;const image=new Image();image.src=payload.image;await image.decode();if(data!==this.data)return;entry.image=image;if(this.comparisons.views.has(view)&&key===this.tissueKey(view))this.comparisons.views.get(view).aspect=payload.width/payload.height;this.drawMap();this.comparisons.draw("tissue-map");}catch(error){if(data!==this.data)return;entry.error=error.message;this.drawMap();this.comparisons.draw("tissue-map");}
    }
    drawTectonics(view="tissue-map"){
      const key=this.tissueKey(view),entry=this.tissueCache.get(key);if(!entry)this.loadTissueMap(key,view);if(entry?.payload)this.comparisons.views.get(view).aspect=entry.payload.width/entry.payload.height;const [ctx,w,h]=this.canvas(view),ink=this.plotInk(view);this.hits[view]=[];
      if(!entry?.image){this.blank(this.comparisons.element("tissue-map",view),entry?.error||"Preparing saved-mask tissue map...");this.comparisons.element("map-caption",view).textContent=entry?.error||"";return;}
      const payload=entry.payload;if(payload.message){this.blank(this.comparisons.element("tissue-map",view),payload.message);this.comparisons.element("map-caption",view).textContent=payload.label;return;}
      const scale=Math.min(w/payload.width,(h-35)/payload.height),dx=(w-payload.width*scale)/2,dy=(h-35-payload.height*scale)/2;ctx.imageSmoothingEnabled=false;ctx.drawImage(entry.image,dx,dy,payload.width*scale,payload.height*scale);
      const selected=this.observation();if(selected)this.strokeContours(ctx,selected,(x,y)=>({x:dx+x*scale,y:dy+y*scale}),ink,1.8);
      this.hits[view]=this.positions().map(p=>({...p,x:dx+p.x*scale,y:dy+p.y*scale}));
      const left=45,right=w-45;for(let i=0;i<payload.legend.length;i++){ctx.fillStyle=`rgb(${payload.legend[i].join(",")})`;ctx.fillRect(left+i*(right-left)/payload.legend.length,h-27,(right-left)/payload.legend.length+1,7);}
      ctx.fillStyle=ink;ctx.font=`${Math.max(10,this.style(view).fontSize-2)}px system-ui`;ctx.fillText(fmt(payload.limits[0]),left,h-6);ctx.textAlign="right";ctx.fillText(fmt(payload.limits[1]),right,h-6);ctx.textAlign="left";
      this.comparisons.element("map-caption",view).textContent=payload.scope+" · "+payload.label+" · outline: selected cell at the shared frame";
    }
    observation(){return this.frameData?.frame_index===this.state.frame?this.frameData.layers.accepted.find(o=>o.identity===this.state.identity):undefined;}
    strokeContours(ctx,item,transform,colour,width=2){ctx.strokeStyle=colour;this.stroke(ctx,width);for(const contour of item.contours){ctx.beginPath();contour.forEach(([x,y],i)=>{const p=transform(x,y);i?ctx.lineTo(p.x,p.y):ctx.moveTo(p.x,p.y);});ctx.closePath();ctx.stroke();}}
    drawImagePreview(ctx,w,h){
      const sourceFrame=this.data.frames[this.state.frame].source_frame_index+1;
      const row=this.history?.exact?this.history.observations[this.state.frame]:this.history?.observations.find(o=>o.frame_index===this.state.frame);
      const item=row?.observation;
      if(!item){
        const message=this.imageError||((row&&row.observation===null)?"No saved mask at this frame.":"Loading saved source frame...");
        this.blank(this.$("cell-image"),message);
        this.$("image-caption").textContent=`Source frame ${sourceFrame} \u00b7 ${message}`;
        return;
      }
      const field=this.val("image-mode")==="field"&&this.fieldSize;
      const b=field?[0,0,...this.fieldSize]:item.bounds,margin=field?0:15;
      const x0=Math.max(0,b[0]-margin),y0=Math.max(0,b[1]-margin);
      const x1=this.fieldSize?Math.min(this.fieldSize[0],b[2]+margin):b[2]+margin;
      const y1=this.fieldSize?Math.min(this.fieldSize[1],b[3]+margin):b[3]+margin;
      const scale=Math.min(w/(x1-x0),h/(y1-y0)),dx=(w-(x1-x0)*scale)/2,dy=(h-(y1-y0)*scale)/2;
      this.strokeContours(ctx,item,(x,y)=>({x:dx+(x-x0)*scale,y:dy+(y-y0)*scale}),this.outlineColour(this.state.identity,white),3.5);
      ctx.fillStyle=muted;ctx.fillText("Saved outline; loading source image",12,20);
      this.$("image-caption").textContent=`Source frame ${sourceFrame} \u00b7 exact saved outline; source image loading`;
    }
    drawFieldPreview(ctx,w,h){
      const d=this.imageMeta,scale=Math.min(w/d.width,h/d.height);
      const width=d.width*scale,height=d.height*scale;
      ctx.imageSmoothingEnabled=true;
      ctx.drawImage(this.signalImage(this.image),(w-width)/2,(h-height)/2,width,height);
      ctx.fillStyle='rgba(0,0,0,.72)';ctx.fillRect(6,6,Math.min(w-12,205),43);
      ctx.fillStyle=white;ctx.font='12px system-ui';ctx.fillText('Quick preview',12,23);
      ctx.fillText(this.imageError?'Full detail unavailable':'Loading full detail...',12,41);
      this.$("image-caption").textContent=`Frame ${d.source_frame_index+1} - Quick preview - ${this.imageError||"loading full detail..."}`;
    }
    drawImage(){const [ctx,w,h]=this.canvas("cell-image");this.hits["cell-image"]=[];if(this.imageMeta?.preview&&this.image){this.drawFieldPreview(ctx,w,h);return;}if(!this.imageMeta||!this.image||(this.imageMeta.image_bounds&&this.val("image-mode")==="field")){this.drawImagePreview(ctx,w,h);return;}const d=this.imageMeta,row=this.history?.observations.find(o=>o.frame_index===this.state.frame),item=this.observation()||row?.observation,crop=this.val("image-mode")==="cell"&&item,source=d.image_bounds||[0,0,d.width,d.height];const b=crop?item.bounds:source,margin=crop?15:0;const x0=Math.max(source[0],b[0]-margin),y0=Math.max(source[1],b[1]-margin),x1=Math.min(source[2],b[2]+margin),y1=Math.min(source[3],b[3]+margin),scale=Math.min(w/(x1-x0),h/(y1-y0)),dx=(w-(x1-x0)*scale)/2,dy=(h-(y1-y0)*scale)/2;const project=(x,y)=>({x:dx+(x-x0)*scale,y:dy+(y-y0)*scale});ctx.imageSmoothingEnabled=false;ctx.drawImage(this.signalImage(this.image),x0-source[0],y0-source[1],x1-x0,y1-y0,dx,dy,(x1-x0)*scale,(y1-y0)*scale);ctx.filter="none";
      const accepted=(d.layers?.accepted||(item?[{...item,identity:this.state.identity}]:[])).filter(o=>this.inCohort(o.identity)),base=!!d.layers?.base;for(const o of this.$("outline-layer").options)o.disabled=["base","both","measurement"].includes(o.value)&&!base;if(this.$("outline-layer").selectedOptions[0].disabled)this.$("outline-layer").value="accepted";
      const mode=this.val("outline-layer");for(const [key,colour] of [["accepted",this.accent(ctx)],["base",gold],["background","#75a9ed"]]){if(key==="background"?mode!=="measurement":mode!==key&&mode!=="both"&&mode!=="measurement")continue;for(const o of (key==="accepted"?accepted:d.layers?.[key]||[]).filter(o=>this.inCohort(o.identity))){if(crop&&o.identity!==this.state.identity)continue;this.strokeContours(ctx,o,project,this.outlineColour(o.identity,key==="accepted"&&o.identity===this.state.identity?white:colour),o.identity===this.state.identity?2.5:1.4);}}
      for(const o of accepted){const p=project(...o.centre);this.hits["cell-image"].push({...p,identity:o.identity});if(!crop){ctx.fillStyle=this.outlineColour(o.identity,o.identity===this.state.identity?gold:white);ctx.font=`${Math.max(10,this.style(ctx.canvas.id.slice(3)).fontSize-1)}px system-ui`;ctx.fillText(String(o.identity),p.x+4,p.y-3);}}
      if(this.$("show-trails").checked&&this.exact){const n=Math.max(1,Math.min(200,+this.val("trail-length")||12)),vals=this.exact.cell.values;let previous=null;for(let f=Math.max(0,this.state.frame-n);f<=this.state.frame;f++){let p=null;const record=this.history?.observations.find(o=>o.frame_index===f)?.observation;if(record)p=project(...record.centre);else if(this.exact.cell.observed[f]&&C.finite(vals.centroid_x?.[f])&&C.finite(vals.centroid_y?.[f])&&this.data.centroid_pixels)p=project(vals.centroid_x[f],vals.centroid_y[f]);if(p&&previous){ctx.strokeStyle=this.greyOutlineIds?.has(this.state.identity)?this.greyOutlineColour:`rgba(239,199,123,${.1+.8*(f-(this.state.frame-n))/n})`;this.stroke(ctx,2);ctx.beginPath();ctx.moveTo(previous.x,previous.y);ctx.lineTo(p.x,p.y);ctx.stroke();}previous=p;}}
      this.$("image-caption").textContent=`Frame ${d.source_frame_index+1} ? ${x1-x0} ? ${y1-y0} px${item?"":" ? cell absent"}`;}
    drawChange(){const [ctx,w,h]=this.canvas("cell-change"),change=this.changeCache.get(this.state.frame),source=this.imageMeta,raw=this.image;if(!change){this.blank(this.$("cell-change"),"Loading saved mask change…");return;}if(!change.payload.change_bounds){this.blank(this.$("cell-change"),"No mask in either frame");return;}const [x0,y0,x1,y1]=change.payload.change_bounds,bounds=source?.image_bounds||[0,0,source?.width||0,source?.height||0],scale=Math.min(w/(x1-x0),h/(y1-y0)),dx=(w-(x1-x0)*scale)/2,dy=(h-(y1-y0)*scale)/2;if(raw&&!source?.preview&&x0>=bounds[0]&&y0>=bounds[1]&&x1<=bounds[2]&&y1<=bounds[3]){ctx.imageSmoothingEnabled=false;ctx.globalAlpha=.5;ctx.drawImage(this.signalImage(raw),x0-bounds[0],y0-bounds[1],x1-x0,y1-y0,dx,dy,(x1-x0)*scale,(y1-y0)*scale);ctx.filter="none";ctx.globalAlpha=1;}if(change.image){ctx.imageSmoothingEnabled=false;ctx.drawImage(change.image,dx,dy,(x1-x0)*scale,(y1-y0)*scale);}const item=this.observation()||this.history?.observations.find(o=>o.frame_index===this.state.frame)?.observation;if(item)this.strokeContours(ctx,item,(x,y)=>({x:dx+(x-x0)*scale,y:dy+(y-y0)*scale}),this.outlineColour(this.state.identity,white),2.5);if(this.state.frame===0){ctx.fillStyle=white;ctx.fillText("First analysed frame: no preceding mask",9,20);}}
    drawCardImage(canvasId,row,identity){
      const [ctx,w,h]=this.canvas(canvasId);
      if(row?.payload.preview){this.blank(this.$(canvasId),'Loading full cell detail...');return;}
      if(!row?.image){this.blank(this.$(canvasId),row?.missing?"No saved mask at this frame":"Loading cell image?");return;}
      const d=row.payload,source=d.image_bounds||[0,0,d.width,d.height];
      const item=d.observation||(identity===this.state.identity?(this.observation()||this.history?.observations.find(o=>o.frame_index===this.state.frame)?.observation):null);
      if(!d.image_bounds&&!item){this.blank(this.$(canvasId),"No saved mask at this frame");return;}
      const b=d.image_bounds||(item?[Math.max(0,item.bounds[0]-15),Math.max(0,item.bounds[1]-15),Math.min(d.width,item.bounds[2]+15),Math.min(d.height,item.bounds[3]+15)]:source);
      const scale=Math.min(w/(b[2]-b[0]),h/(b[3]-b[1])),dx=(w-(b[2]-b[0])*scale)/2,dy=(h-(b[3]-b[1])*scale)/2;
      ctx.imageSmoothingEnabled=false;
      ctx.drawImage(this.signalImage(row.image),b[0]-source[0],b[1]-source[1],b[2]-b[0],b[3]-b[1],dx,dy,(b[2]-b[0])*scale,(b[3]-b[1])*scale);
      if(item)this.strokeContours(ctx,item,(x,y)=>({x:dx+(x-b[0])*scale,y:dy+(y-b[1])*scale}),this.outlineColour(identity,white),2);
      else{ctx.fillStyle=gold;ctx.fillText("No saved mask at this frame",8,19);}
    }
    drawPeerTrace(canvasId,identity){
      const [ctx,w,h]=this.canvas(canvasId),exact=this.peerExact.get(identity),cell=exact?.cell||this.data.cells.find(c=>c.identity===identity);
      if(!cell){this.blank(this.$(canvasId),"Loading measurements…");return;}
      const indices=exact?.sample_indices||this.data.sample_indices,series=this.cellTraceSeries(cell,indices),key=series[0]?.metric;
      this.hits[canvasId]=this.drawTraceSet(ctx,w,h,series,{compact:true})[0];
      ctx.canvas.dataset.traceColour=this.traceColour(identity,key);ctx.canvas.dataset.rhythmStatus=this.rhythmResultsVisible()?this.rhythmForMetric(identity,key)?.status||"unknown":"unknown";
    }
    drawCellCards(){if(!this.data||!this.data.identities.length)return;this.$("selected-card-label").textContent=`Cell ${this.state.identity} · selected`;let selected=this.cellImageCache.get(this.state.frame);if(!selected&&this.image&&this.imageMeta)selected={image:this.image,payload:this.imageMeta};this.drawCardImage("selected-mini",selected,this.state.identity);for(const slot of ["peer-one","peer-two"]){const card=this.$(slot).closest(".xp-cell-card");if(card.hidden)continue;const identity=+this.val(slot);this.drawCardImage(slot+"-image",this.peerRows.get(identity)?.get(this.state.frame),identity);this.drawPeerTrace(slot+"-trace",identity);}}
    drawHistory(){
      const [ctx,w,h]=this.canvas("outline-history");
      const sourceFrame=this.data.frames[this.state.frame].source_frame_index+1;
      const currentLabel=this.$("history-current");
      const marker=(present)=>{
        const y=h-22-this.fraction(this.state.frame)*(h-44);
        ctx.strokeStyle="#425b6a";this.stroke(ctx,1);ctx.beginPath();ctx.moveTo(w-18,18);ctx.lineTo(w-18,h-22);ctx.stroke();
        ctx.strokeStyle=present?white:gold;this.stroke(ctx,2.5);ctx.beginPath();ctx.moveTo(w-34,y);ctx.lineTo(w-5,y);ctx.stroke();
        ctx.fillStyle=present?white:gold;ctx.fillText(String(sourceFrame),w-64,Math.max(20,Math.min(h-20,y+4)));
      };
      if(!this.history){this.blank(this.$("outline-history"),this.historyError||"Loading saved outline history...");if(currentLabel)currentLabel.textContent=`Source frame ${sourceFrame} \u00b7 loading outline`;marker(false);return;}
      const history=this.history,stride=Math.max(1,history.stride||1);
      let rows=history.observations.filter(o=>o.observation&&(o.frame_index%stride===0||o.frame_index===history.count-1));
      if(this.val("history-window")!=="all")rows=rows.filter(o=>Math.abs(o.frame_index-this.state.frame)<=24);
      const exact=history.exact?history.observations[this.state.frame]:history.observations.find(o=>o.frame_index===this.state.frame);
      const selected=exact?.observation||this.observation();
      if(currentLabel)currentLabel.textContent=`Source frame ${sourceFrame} \u00b7 ${selected?"white = selected outline":"no observed outline"}`;
      if(!rows.length&&!selected){this.blank(this.$("outline-history"),"No observed outlines in the selected range.");marker(false);return;}
      rows.sort((a,b)=>a.frame_index-b.frame_index);
      const allRows=selected?[...rows,{frame_index:this.state.frame,observation:selected}]:rows;
      const centred=this.val("history-position")==="centred",ref=rows.length?rows[Math.floor(rows.length/2)].observation.centre:selected.centre;
      const all=allRows.flatMap(r=>r.observation.contours.flat().map(([x,y])=>[x-(centred?r.observation.centre[0]:ref[0]),y-(centred?r.observation.centre[1]:ref[1])]));
      const span=Math.max(15,...all.map(p=>Math.max(Math.abs(p[0]),Math.abs(p[1]))));
      const {yaw,pitch,zoom}=this.rotation,s=1.6*zoom*Math.min(w,h)/(span*8),count=Math.max(1,this.data.frames.length-1);
      const project=(x,y,f)=>{const a=x*Math.cos(yaw)-y*Math.sin(yaw),b=x*Math.sin(yaw)+y*Math.cos(yaw),z=(this.fraction(f)-.5)*span*5;return {x:w/2+a*s,y:h/2+(b*Math.cos(pitch)-z*Math.sin(pitch))*s};};
      for(const r of rows){
        if(r.frame_index===this.state.frame)continue;
        const origin=centred?r.observation.centre:ref;
        this.strokeContours(ctx,r.observation,(x,y)=>project(x-origin[0],y-origin[1],r.frame_index),this.outlineColour(this.state.identity,heat(.2+.8*r.frame_index/count,this.accent(ctx))),1);
      }
      if(selected){const origin=centred?selected.centre:ref;this.strokeContours(ctx,selected,(x,y)=>project(x-origin[0],y-origin[1],this.state.frame),this.outlineColour(this.state.identity,white),3.5);}
      marker(Boolean(selected));
      ctx.fillStyle=muted;ctx.fillText(`${fmt(this.data.frames[0].hours)} h \u2192 ${fmt(this.data.frames.at(-1).hours)} h`,15,h-15);
      this.$("history-caption").textContent=history.message;
    }
    allPositions(){const frame=this.frameData?.frame_index===this.state.frame?this.frameData:null;if(frame?.partial_layers&&frame.positions)return frame.positions;const geometry=frame&&!frame.partial_layers?frame.layers.accepted:null;if(geometry)return geometry.map(o=>({identity:o.identity,x:o.centre[0],y:o.centre[1]}));return (this.frameValues?.cells||[]).filter(c=>C.finite(c.values.centroid_x?.[0])&&C.finite(c.values.centroid_y?.[0])).map(c=>({identity:c.identity,x:c.values.centroid_x[0],y:c.values.centroid_y[0]}));}
    positions(){return this.allPositions().filter(p=>this.inCohort(p.identity));}
    mapPoints(ctx,w,h,positions){const [xmin,xmax]=C.range(positions.map(p=>p.x)),[ymin,ymax]=C.range(positions.map(p=>p.y)),s=Math.min((w-90)/(xmax-xmin),(h-70)/(ymax-ymin));return positions.map(p=>({...p,x:45+(p.x-xmin)*s,y:35+(p.y-ymin)*s}));}
    drawMap(view="tissue-map"){if(!this.data?.identities.length){this.blank(this.comparisons.element("tissue-map",view),"No cells match this filter.");return;}if(this.comparisons.value("map-layer",view).startsWith("tectonics:")){this.drawTectonics(view);return;}const [ctx,w,h]=this.canvas(view),positions=this.positions();this.hits[view]=[];if(!positions.length){this.blank(this.comparisons.element("tissue-map",view),"No saved cell positions for this frame.");return;}const layer=this.comparisons.value("map-layer",view),key=C.metricKey(layer,"total",this.data),value=id=>layer.startsWith("phase:")?this.phaseFor(id,layer.slice(6)):this.frameValues?.cells.find(c=>c.identity===id)?.values[key]?.[0];const [lo,hi]=C.range(positions.map(p=>value(p.identity)));const points=this.mapPoints(ctx,w,h,positions);for(const p of points){const v=value(p.identity);ctx.fillStyle=layer==="identity"?this.accent(ctx):C.finite(v)?heat((v-lo)/(hi-lo),this.accent(ctx)):"#495b6c";ctx.beginPath();ctx.arc(p.x,p.y,p.identity===this.state.identity?10:6,0,Math.PI*2);ctx.fill();if(p.identity===this.state.identity){ctx.strokeStyle=white;this.stroke(ctx,2);ctx.stroke();}ctx.fillStyle=muted;ctx.fillText(String(p.identity),p.x+10,p.y+3);}this.hits[view]=points;this.comparisons.element("map-caption",view).textContent=`Exact saved positions at selected frame · ${layer==="identity"?"cell identities":this.comparisons.element("map-layer",view).selectedOptions[0]?.textContent} · grey = unavailable`;}
    drawNetwork(){
      const [ctx,w,h]=this.canvas("relationships"),result=this.data.results?.relations,source=result?.sources.find(s=>s.key===this.val("link-source"));this.hits.relationships=[];
      if(!source||(source.preview&&source.frame_index!==this.state.frame)){
        this.blank(this.$("relationships"),this.previewContacts&&!result?.message?"Computing contacts at selected frame...":this.previewContacts&&result?.message?.startsWith("No saved")?"Computing contacts at selected frame...":result?.message||"No available relationships.");this.$("network-caption").textContent="";return;
      }
      let links=source.rows.filter(r=>(r.frame_index==null||r.frame_index===this.state.frame)&&this.inCohort(r.identity_a)&&this.inCohort(r.identity_b));
      const neighbourhood=this.val("network-scope")==="neighbours";
      if(neighbourhood)links=links.filter(r=>r.identity_a===this.state.identity||r.identity_b===this.state.identity);
      const visible=(source.identities||this.positions().map(p=>p.identity)).filter(id=>this.inCohort(id)),identities=neighbourhood?[...new Set([...(visible.includes(this.state.identity)?[this.state.identity]:[]),...links.flatMap(r=>[r.identity_a,r.identity_b])])]:[...new Set([...visible,...links.flatMap(r=>[r.identity_a,r.identity_b])])];
      if(!identities.length){this.blank(this.$("relationships"),"No observed cells at this frame.");return;}
      const positions=this.positions(),hasPositions=identities.every(id=>positions.some(p=>p.identity===id));
      const points=hasPositions?this.mapPoints(ctx,w,h,positions.filter(p=>identities.includes(p.identity))):identities.map((identity,i)=>({identity,x:w/2+Math.cos(i/Math.max(1,identities.length)*Math.PI*2)*Math.min(w,h)*.35,y:h/2+Math.sin(i/Math.max(1,identities.length)*Math.PI*2)*Math.min(w,h)*.35}));
      ctx.strokeStyle="#516575";this.stroke(ctx,1.5);
      for(const link of links){const a=points.find(p=>p.identity===link.identity_a),b=points.find(p=>p.identity===link.identity_b);ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();}
      for(const p of points){ctx.fillStyle=p.identity===this.state.identity?gold:this.accent(ctx);ctx.beginPath();ctx.arc(p.x,p.y,p.identity===this.state.identity?8:5,0,Math.PI*2);ctx.fill();ctx.fillStyle=white;ctx.fillText(String(p.identity),p.x+10,p.y+4);}
      this.hits.relationships=points;this.$("network-caption").textContent=`${source.title} | ${links.length} links | ${source.meaning}`;
      if(neighbourhood&&!links.length){ctx.fillStyle=muted;ctx.textAlign="center";ctx.fillText(visible.includes(this.state.identity)?"No contacts for this cell at this frame.":"Selected cell has no observed mask at this frame.",w/2,h-12);ctx.textAlign="left";}
    }
    drawDistribution(view="distribution"){this.groupComparison?.draw(view);if(view==="distribution")this.comparisons.draw("distribution");}

  }
  window.PyMicrogliaExploration={mount:(host,options)=>new Explorer(host,options)};
})();
