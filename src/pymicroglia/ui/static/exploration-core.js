"use strict";
((root) => {
  const finite = x => typeof x === "number" && Number.isFinite(x);
  function smooth(values, indices, n=1) {
    if(n===1)return values.slice();
    const half=Math.floor(n/2);
    return values.map((v,i)=>{
      if(i<half||i+half>=values.length)return null;
      const segment=values.slice(i-half,i+half+1);
      if(!segment.every(finite)||indices[i+half]-indices[i-half]!==n-1)return null;
      return segment.reduce((a,b)=>a+b,0)/n;
    });
  }
  const range = values => {const valid=values.filter(finite);if(!valid.length)return [0,1];const lo=Math.min(...valid),hi=Math.max(...valid);return lo===hi?[lo-.5,hi+.5]:[lo,hi];};
  const mean = values => {const a=values.filter(finite);return a.length?a.reduce((x,y)=>x+y,0)/a.length:null;};
  function metricKey(metric,aggregation,data) {
    if(metric!=="light")return metric;
    const key=aggregation==="mean"?"signal_mean":"integrated_density";
    return data.metrics.some(m=>m.key===key)?key:null;
  }
  function selection(data, current={}, patch={}) {
    const s={...current,...patch};
    if(!data.identities.includes(Number(s.identity)))s.identity=data.identities[0];
    s.identity=Number(s.identity);s.frame=Math.max(0,Math.min(data.frames.length-1,Math.round(Number(s.frame)||0)));
    return {...s,run:data.run,stem:data.stem};
  }
  function alignFit(fit, indices, values) {
    const points=new Map((fit?.indices||[]).map((frame,i)=>[frame,fit.values[i]]));
    return indices.map((frame,i)=>finite(values[i])&&finite(points.get(frame))?points.get(frame):null);
  }
  // One measured range transforms the observations and their saved fit equally.
  // Missing values remain missing; a constant trace sits halfway up the axis.
  function scaleSeries(values, fitted=[], normalized=false) {
    const [lo,hi]=range(values),convert=v=>finite(v)?(normalized?(v-lo)/(hi-lo):v):null;
    return {values:values.map(convert),fitted:fitted.map(convert),range:normalized?[0,1]:[lo,hi]};
  }
  const seriesColours=["#177a98","#c44a70","#387b42","#7954a3","#ae681b","#3869c5","#976742","#227e79"];
  function rhythmPlan(config,schema) {
    const plan={...schema.defaults,...config},component=Boolean(plan.fft_component_test);
    const method=plan.fit_method||schema.defaults.fit_method,test=component?"fft_component":plan.significance_method||schema.tests.find(t=>t.key===method)?.key||"lomb";
    return {fit_method:method,test,detrend:plan.detrend,multiple_testing:plan.multiple_testing,
      period_min_hours:Number(plan.period_min_hours),period_max_hours:Number(plan.period_max_hours),
      rhythmic_alpha:Number(plan.rhythmic_alpha),min_observations:Number(plan.min_observations),min_cycles:Number(plan.min_cycles),
      metrics:[...(plan.metrics||[])].sort()};
  }
  function sameRhythmPlan(a,b) {return JSON.stringify(a)===JSON.stringify(b);}
  const api={finite,smooth,range,mean,metricKey,selection,alignFit,scaleSeries,seriesColours,rhythmPlan,sameRhythmPlan};
  root.PyMicrogliaExplorationCore=api;
  if(typeof module!=="undefined")module.exports=api;
})(typeof window!=="undefined"?window:globalThis);
