"""Hoist immutable outline measurements out of dormant-identity loops.

apply_dormancy never writes its working frame inside the identity loop, so
the host components are labelled once per call. Each component's host score
was np.mean of the dilated old host over the component: an integer count
over the component size, divided once in float64. One count table per
identity gives the same quotients, and np.argmax keeps the loop's choice of
the first highest score (a later equal score never replaced it); only the
winning component's mask is formed. The frame's observations are listed
without masks, and only the regions named by the dormant identity get one.

substantial_duplicate read whole-frame distance transforms only through two
threshold tests. A pixel outside a part's box widened by ceil(threshold) is
farther than the threshold from every pixel of the part, so each test has the
same answer from a distance transform of that widened box.
"""
from contextlib import contextmanager
import hashlib
import inspect
import math

import numpy as np


HASHES={
    'apply_dormancy':'0bc40703392072302483fffdfdc370fcef5baa883458292e0b62355bbc06b025',
    'add_reciprocal_dormant_swaps':'68f045097e97c67b71d5e7abe06e9ade9a767759b24b28046bf4c05316356fc0'}
DUPLICATE='022426385b419b360bf1dea28316b2d9de6a09562b8bef6d1ca1aed7678f015b'


def _window(box,pad,shape):
    ys,xs=box
    return (slice(max(ys.start-pad,0),min(ys.stop+pad,shape[0])),
            slice(max(xs.start-pad,0),min(xs.stop+pad,shape[1])))


def boxed_duplicate(original,stats):
    """substantial_duplicate with each distance transform limited to the part's widened box."""
    scope=original.__globals__
    def substantial_duplicate(frame,identity,minimum_area,minimum_distance,maximum_fragment_gap):
        plain=np.asarray(frame)
        if (plain.ndim!=2 or not math.isfinite(float(minimum_distance))
                or not math.isfinite(float(maximum_fragment_gap))):
            stats['duplicate_fallback']+=1
            return original(frame,identity,minimum_area,minimum_distance,maximum_fragment_gap)
        ndi=scope['ndi']
        numbered,count=ndi.label(plain==identity,structure=scope['STRUCTURE'])
        sizes=np.bincount(numbered.ravel(),minlength=count+1)
        kept=[c for c in range(1,count+1) if int(sizes[c])>=minimum_area]
        if len(kept)<2:
            return False
        primary=max(kept,key=lambda c:int(sizes[c]))
        boxes=ndi.find_objects(numbered)
        stats['duplicate_boxed']+=1
        others=None
        for part in kept:
            if part==primary:
                continue
            rows,cols=_window(boxes[part-1],max(0,math.ceil(minimum_distance)),plain.shape)
            window=numbered[rows,cols]
            held=window==primary
            # No primary pixel in the widened box: the nearest is beyond the threshold.
            if held.any() and float(ndi.distance_transform_edt(~held)[window==part].min())<minimum_distance:
                continue
            if others is None:
                others=(plain>0)&(plain!=identity)
                anywhere=bool(np.any(others))
            if anywhere:
                rows,cols=_window(boxes[part-1],max(0,math.ceil(maximum_fragment_gap)),plain.shape)
                near=others[rows,cols]
                if near.any() and float(ndi.distance_transform_edt(~(numbered[rows,cols]==part))[near].min()) \
                        <=maximum_fragment_gap:
                    return True
        return False
    return substantial_duplicate


@contextmanager
def install(module):
    changes=[]
    stats=dict(supported=[],duplicate_boxed=0,duplicate_fallback=0)
    for name,digest in HASHES.items():
        original=getattr(module,name,None)
        if original is None:continue
        source=inspect.getsource(original)
        if hashlib.sha256(source.encode()).hexdigest()!=digest:continue
        if name=='apply_dormancy':
            source=source.replace('    for identity, state in list(states.items()):',
                '    _host_numbered, _host_count = ndi.label(result > 0, structure=STRUCTURE) if states else (None, 0)\n'
                '    _host_sizes = np.bincount(_host_numbered.ravel(), minlength=_host_count + 1)[1:] if _host_count else None\n'
                '    frame_observations = observations(result) if states else []\n'
                '    for identity, state in list(states.items()):')
            # Only the dormant identity's own regions read a mask; it is the
            # same pixels placed in an empty frame (observation_mask).
            for old,new in (('            if item.identity != identity:\n                continue\n',
                             '            if item.identity != identity:\n                continue\n'
                             '            _item_mask = observation_mask(item, result.shape)\n'),
                            ('                & item.mask))','                & _item_mask))'),
                            ('            red = float(np.mean(lag[item.mask] >= 1.5))',
                             '            red = float(np.mean(lag[_item_mask] >= 1.5))')):
                assert source.count(old)==1
                source=source.replace(old,new)
            source=source.replace('        host_candidates = binary_components(result > 0)\n','')
            source=source.replace('''        for mask in host_candidates:
            score = float(np.mean(expanded[mask]))
            if score > host_score:
                host_score, host = score, mask
''','''        if _host_count:
            _host_scores = (np.bincount(_host_numbered[expanded], minlength=_host_count + 1)[1:]
                            / _host_sizes)
            _best = int(np.argmax(_host_scores))
            host_score, host = float(_host_scores[_best]), _host_numbered == _best + 1
''')
            source=source.replace('for item in observations(result, with_masks=True):','for item in frame_observations:')
        else:
            source=source.replace('    for proposal in proposals:',
                '    frame_observations = None\n    for proposal in proposals:')
            source=source.replace('        for item in observations(current, with_masks=True):',
                '        if frame_observations is None:\n'
                '            frame_observations = observations(current, with_masks=True)\n'
                '        for item in frame_observations:')
        # Retain live module globals: later scientific contexts still apply.
        namespace={}
        exec(compile(source,'<reused-ownership-observations>','exec'),original.__globals__,namespace)
        setattr(module,name,namespace[name]);changes.append((name,original))
        stats['supported'].append(name)
    original=getattr(module,'substantial_duplicate',None)
    if original is not None and hashlib.sha256(inspect.getsource(original).encode()).hexdigest()==DUPLICATE:
        module.substantial_duplicate=boxed_duplicate(original,stats)
        changes.append(('substantial_duplicate',original));stats['supported'].append('substantial_duplicate')
    try:yield stats
    finally:
        for name,original in reversed(changes):setattr(module,name,original)


@contextmanager
def install_loader(history):
    from contextlib import ExitStack
    original=history._issue_module
    counters=[]
    with ExitStack() as stack:
        def load(*args,**kwargs):
            module=original(*args,**kwargs)
            if hasattr(module,'apply_dormancy'):
                counters.append(stack.enter_context(install(module)))
            return module
        history._issue_module=load
        try:yield counters
        finally:history._issue_module=original
