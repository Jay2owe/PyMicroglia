"""Reuse immutable component geometry and backward ownership-history links."""
from collections import OrderedDict
from contextlib import contextmanager,ExitStack
import hashlib
import os
from pathlib import Path
import sys
import numpy as np
from skimage.measure import label

GEOMETRY='7ddff7a3843a6036488499afb0fe3762b6f5da5c96e7e27754a2e36243d5f155'
PERSISTENT='adf323348cb6735dd284d09aaaa4b75f56cc59ea9449d710fe7166977a403f84'


def digest(module):return hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()


@contextmanager
def geometry(module):
    stats=dict(supported=False,frames=0,overlaps=0,disjoint=0)
    if digest(module)!=GEOMETRY:yield stats;return
    old_components,old_overlap=module.frame_components,module.overlap_pixels
    validated=OrderedDict();retained=0
    def components(frame,frame_index,minimum_area=1):
        stats['frames']+=1
        if frame.ndim!=2 or frame.dtype.kind not in 'iu' or (frame.dtype.kind=='u' and frame.size and int(frame.max())>np.iinfo(np.int64).max):
            return old_components(frame,frame_index,minimum_area)
        numbered=label(frame,background=0,connectivity=2)
        positions=np.flatnonzero(numbered);values=numbered.flat[positions]
        order=np.argsort(values,kind='stable');positions=positions[order];values=values[order]
        cuts=np.flatnonzero(values[1:]!=values[:-1])+1
        numbers={};result=[]
        for pixels in np.split(positions,cuts):
            if not len(pixels):continue
            identity=int(frame.flat[pixels[0]]);number=numbers.get(identity,0)+1;numbers[identity]=number
            if len(pixels)<minimum_area:continue
            yy,xx=np.divmod(pixels,frame.shape[1])
            result.append(module.Component(frame_index,identity,number,pixels.copy(),len(pixels),float(xx.mean()),float(yy.mean())))
        result.sort(key=lambda item:(item.observed_identity,item.local_component))
        return result
    def checked(component):
        nonlocal retained
        key=id(component)
        if key in validated:
            validated.move_to_end(key);return validated[key][1]
        pixels=np.asarray(component.pixels)
        good=pixels.ndim==1 and pixels.dtype.kind in 'iu' and np.all(pixels[1:]>pixels[:-1])
        result=pixels if good else None;cost=pixels.nbytes+512
        if cost<=16*1024**2:
            while validated and retained+cost>16*1024**2:
                _,(_,_,size)=validated.popitem(last=False);retained-=size
            validated[key]=(component,result,cost);retained+=cost
        return result
    def overlap(first,second):
        stats['overlaps']+=1;a,b=checked(first),checked(second)
        if a is None or b is None or a.dtype!=b.dtype:return old_overlap(first,second)
        if not len(a) or not len(b) or a[-1]<b[0] or b[-1]<a[0]:
            stats['disjoint']+=1;return 0
        if len(a)>len(b):a,b=b,a
        lo=np.searchsorted(a,b[0]);hi=np.searchsorted(a,b[-1],side='right');a=a[lo:hi]
        return int(np.count_nonzero(b[np.searchsorted(b,a)]==a))
    old_tracklets=module.build_tracklets
    def tracklets(labels,maximum_step_px=15.0,maximum_gap_frames=2,minimum_area=1):
        # The frozen loop scores every tracklet ever made for every component.
        # Only same-name tracklets ending 1..gap frames earlier can score, so
        # index those; the frozen set order still breaks exact ties.
        stats['tracklet_calls']=stats.get('tracklet_calls',0)+1
        frames=[];tracklets={};next_number=1;by_identity={}
        for frame_index,frame in enumerate(labels):
            current=module.frame_components(frame,frame_index,minimum_area)
            available=set(tracklets);order=None
            for component in sorted(current,key=lambda item:-item.area):
                identity=component.observed_identity
                numbers=by_identity.get(identity,[])
                # Last frames only grow on a match, so an expired tracklet can never score again.
                numbers[:]=[n for n in numbers if frame_index-tracklets[n].components[-1].frame<=maximum_gap_frames]
                choices=[]
                for number in numbers:
                    if number not in available:continue
                    track=tracklets[number]
                    gap=frame_index-track.last.frame
                    if track.observed_identity!=identity or gap<1 or gap>maximum_gap_frames:continue
                    shared=module.overlap_pixels(track.last,component)
                    separation=module.distance(track.last,component)
                    if shared==0 and separation>maximum_step_px*gap:continue
                    choices.append(((shared,-separation,-gap),track))
                if choices:
                    best=max(row[0] for row in choices)
                    tied=[track for key,track in choices if key==best]
                    if len(tied)>1 or any(key[1]!=key[1] for key,_ in choices):
                        # Ties (and NaN keys) depend on the frozen iteration order.
                        stats['tracklet_ties']=stats.get('tracklet_ties',0)+1
                        order={number:index for index,number in enumerate(list(available))}
                        choices.sort(key=lambda row:order[row[1].number])
                        tied=[max(choices,key=lambda row:row[0])[1]]
                    track=tied[0]
                    available.remove(track.number)
                else:
                    track=module.Tracklet(next_number,identity)
                    tracklets[next_number]=track
                    by_identity.setdefault(identity,numbers)
                    by_identity[identity].append(next_number)
                    next_number+=1
                component.tracklet=track.number
                component.local_age=len(track.components)+1
                track.components.append(component)
            frames.append(current)
        return frames,tracklets
    aliases=[];root=Path(module.__file__).resolve().parent
    # A module inside root has one of root's file names; resolving every
    # other loaded module on each helper reload was a measurable cost.
    local={name.lower() for name in os.listdir(root)}
    for child in [m for m in list(sys.modules.values()) if type(m).__module__ != 'Cython.Shadow']:
        path=getattr(child,'__file__',None)
        if path is None or os.path.basename(path).lower() not in local or Path(path).resolve().parent!=root:continue
        for name,value in list(vars(child).items()):
            replacement=(components if value is old_components else overlap if value is old_overlap
                         else tracklets if value is old_tracklets else None)
            if replacement is not None:aliases.append((child,name,value));setattr(child,name,replacement)
    stats['supported']=True
    try:yield stats
    finally:
        for child,name,value in reversed(aliases):setattr(child,name,value)
        validated.clear()


@contextmanager
def paths(module):
    stats=dict(supported=False,calls=0,links=0,reused=0,peak_links=0)
    if digest(module)!=PERSISTENT:yield stats;return
    original=module.physical_chain_backward;old_run=module.persistent_quarantine
    cache=OrderedDict();scope=None
    def clear():
        nonlocal scope
        cache.clear();scope=None
    def chain(frames,frame,start,cfg):
        nonlocal scope
        stats['calls']+=1
        if scope is not frames:clear();scope=frames
        radius=float(cfg['owner_match_radius_px']);result=[start];reference=start
        for prior in range(frame-1,-1,-1):
            key=(id(reference),prior,radius.hex())
            item=cache.pop(key,None)
            if item is None:
                match=module.best_match(frames[prior],reference,radius)
                if match is not None:
                    ratio=match.area/max(reference.area,1)
                    if ratio<.25 or ratio>4.:match=None
                item=(reference,match);stats['links']+=1
            else:stats['reused']+=1
            cache[key]=item
            if len(cache)>32768:cache.popitem(last=False)
            stats['peak_links']=max(stats['peak_links'],len(cache))
            match=item[1]
            if match is None:break
            result.append(match);reference=match
        return result
    def run(*args,**kwargs):
        clear()
        try:return old_run(*args,**kwargs)
        finally:clear()
    module.physical_chain_backward=chain;module.persistent_quarantine=run;stats['supported']=True
    try:yield stats
    finally:
        module.physical_chain_backward=original;module.persistent_quarantine=old_run;clear()


@contextmanager
def install_loader(history):
    original=history._issue_module;seen=set();counters=[]
    with ExitStack() as stack:
        def load(*args,**kwargs):
            module=original(*args,**kwargs)
            helper=sys.modules.get('component_tenure_audit')
            if helper is not None and id(helper) not in seen:
                counter=stack.enter_context(geometry(helper));counters.append(counter)
                if counter['supported']:seen.add(id(helper))
            if hasattr(module,'physical_chain_backward') and id(module) not in seen:
                counter=stack.enter_context(paths(module));counters.append(counter)
                if counter['supported']:seen.add(id(module))
            return module
        history._issue_module=load
        try:yield counters
        finally:history._issue_module=original
