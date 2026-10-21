"use strict";
(() => {
  const R=window.CellReview;
  function pixels(item,centered) {
    const set=new Set(); if(!item) return set;
    const [ax,ay]=centered?item.anchor:[0,0];
    for(let i=0;i<item.runs.length;i+=3) {
      const [y,x0,x1]=item.runs.slice(i,i+3);
      for(let x=x0;x<x1;x++) set.add(`${x-ax},${y-ay}`);
    } return set;
  }
  function compare(previous,current,centered) {
    if(!previous || !current) return null;
    const old=pixels(previous,centered), now=pixels(current,centered);
    return {gained:[...now].filter(p=>!old.has(p)), lost:[...old].filter(p=>!now.has(p)),
            retained:[...now].filter(p=>old.has(p)).length};
  }
  const view=R.changes={compare,pixels,init() {
    R.$("change-mode").onchange=event=>R.select({changeMode:event.target.value});
  },render(image) {
    const cell=R.cell(), current=R.observed(), previous=R.observed(R.state.cellId,R.state.frame-1);
    const location=current ?? R.location(cell,R.state.frame);
    const size=cell?.crop_size_px ?? 41, raw=R.$("raw-crop"), changed=R.$("change-crop");
    const sourceView=R.drawRaw(raw,image,location,size);R.contours(sourceView,current);R.scaleBar(raw,sourceView);
    // The native-coordinate crop includes both observed masks, even after a
    // large displacement, rather than silently hiding the lost pixels.
    let comparisonSize=size;
    if(current && previous && R.state.changeMode==="original") {
      const [ax,ay]=current.anchor, [x0,y0,x1,y1]=previous.bounds;
      comparisonSize=Math.max(size,2*(Math.max(ax-x0,x1-ax,ay-y0,y1-ay)+5)+1);
    }
    const target=R.drawRaw(changed,image,location,comparisonSize);
    target.ctx.fillStyle="rgba(0,0,0,.45)";target.ctx.fillRect(0,0,changed.width,changed.height);
    const centered=R.state.changeMode==="centered", delta=compare(previous,current,centered);
    this.last=delta;R.$("change-caption").textContent=centered?"Cell-centred mask changes":"Changes in recording coordinates";
    if(delta) {
      const anchor=centered?current.anchor:[0,0];
      for(const [keys,colour] of [[delta.gained,R.colours.cyan],[delta.lost,R.colours.amber]]) {
        target.ctx.fillStyle=colour;
        for(const key of keys) {const [x,y]=key.split(",").map(Number);target.ctx.fillRect((x+anchor[0]-target.left)*target.scale,(y+anchor[1]-target.top)*target.scale,target.scale,target.scale);}
      }
      R.$("change-status").textContent=`Cell ${cell.id} · +${delta.gained.length} gained / −${delta.lost.length} lost pixels · ${delta.retained} retained`;
    } else R.$("change-status").textContent=!current?"No saved mask for this cell at this frame.":"No mask in the preceding frame. A comparison is not drawn across that gap.";
    R.contours(target,current);R.scaleBar(changed,target);
  }};
  R.views.push(view);
})();
