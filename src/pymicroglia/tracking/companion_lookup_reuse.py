"""Index immutable companion evidence once per excursion discovery."""
from contextlib import contextmanager
from collections import defaultdict
import hashlib
import inspect
import types


HASHES={'discover':'ec09d3747df3189891de330872c182fc87250772a6dc2edbd523512c6cdad29e',
        '_companion':'4d2d01efef409880c3d06ddd86fcb6f25eeb8eee0bb1fbb3f27c533e3a59a167'}


class CompanionIndex:
    def __init__(self,attached,module,counters):
        self.attached=attached;self.module=module;self.counters=counters
        self.residents=defaultdict(dict);self.owners=defaultdict(list);self.groups=[]
        # Preserve the first row at duplicate frame keys, as DataFrame.iloc[0].
        for row in attached.itertuples(index=False):
            self.residents[int(row.track_id)].setdefault(row.frame,row)
        visible=attached[attached.physically_visible.astype(bool)]
        for track,group in visible.groupby('track_id'):
            first=int(group.frame.astype(int).min())
            rows=defaultdict(list)
            for row in group.itertuples(index=False):rows[int(row.candidate_owner)].append(row)
            self.groups.append((int(track),first,rows))
            for owner,points in rows.items():
                self.owners[owner].append((int(track),first,points))
        counters['indexed_rows']+=len(attached)

    def query(self,resident_track,owner_b,middle_frames,minimum_support,minimum_overlap,maximum_distance):
        self.counters['queries']+=1
        resident=self.residents.get(int(resident_track),{})
        middle=set(middle_frames);candidates=[]
        groups=(self.owners.get(int(owner_b),()) if minimum_support>0 else
                [(track,first,rows.get(int(owner_b),())) for track,first,rows in self.groups])
        for track,first,points in groups:
            if track==int(resident_track):continue
            support=len(points)
            if support<minimum_support:continue
            overlap=0
            for point in points:
                frame=int(point.frame)
                if frame not in middle or frame not in resident:continue
                if self.module._scaled_step(resident[frame],point)<=maximum_distance:overlap+=1
            if overlap>=minimum_overlap:candidates.append((overlap,support,track,first))
        if not candidates:return 0,0,0,-1
        overlap,support,track,first=max(candidates)
        return track,support,overlap,first


@contextmanager
def install(module):
    originals={name:getattr(module,name) for name in HASHES}
    stats=dict(supported=False,queries=0,indexed_rows=0)
    if any(hashlib.sha256(inspect.getsource(originals[name]).encode()).hexdigest()!=digest
           for name,digest in HASHES.items()):
        yield stats;return
    stats['supported']=True
    original=originals['discover']
    def discover(*args,**kwargs):
        measured=None
        def companion(attached,*values,**options):
            nonlocal measured
            if measured is None:measured=CompanionIndex(attached,module,stats)
            if measured.attached is not attached:return originals['_companion'](attached,*values,**options)
            return measured.query(*values,**options)
        namespace=dict(original.__globals__);namespace['_companion']=companion
        function=types.FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__)
        function.__kwdefaults__=original.__kwdefaults__
        try:return function(*args,**kwargs)
        finally:namespace.clear();measured=None
    module.discover=discover
    try:yield stats
    finally:module.discover=original
