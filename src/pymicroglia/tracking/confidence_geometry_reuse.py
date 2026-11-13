"""Measure the same confidence samples inside each occupied bounding box."""
from contextlib import contextmanager
import hashlib
import inspect
from pathlib import Path
import math
import numpy as np
from scipy import ndimage as ndi

SOURCE = '30f9cef93954b2843bc964de7bd9a916fc41c148d06c0a13895215b91a09e552'


def regions(frame):
    frame = np.asarray(frame)
    if (frame.ndim != 2 or frame.dtype.kind not in 'iu' or
            (frame.size and (frame.min() < 0 or frame.max() > frame.size))):
        # Unusual sparse/negative identity spaces preserve the original path.
        for identity in sorted(set(map(int, np.unique(frame))) - {0}):
            yield identity, (slice(0,frame.shape[0]),slice(0,frame.shape[1])), frame==identity
        return
    for identity, region in enumerate(ndi.find_objects(frame), 1):
        if region is not None:
            yield identity, region, frame[region] == identity


def _plain(value):
    value=np.asarray(value)
    return value.view(np.ndarray) if isinstance(value,np.memmap) else value


# binary_dilation's default structure for a 2-D mask.
CROSS=ndi.generate_binary_structure(2,1)


def _boxes(frame):
    """regions()'s identities and boxes from one maximum, or None where it keeps its original path."""
    if frame.ndim!=2 or frame.dtype.kind not in 'iu' or not frame.size:
        return None
    top=int(frame.max())
    if top>frame.size or (frame.dtype.kind=='i' and int(frame.min())<0):
        return None
    # find_objects reads the maximum itself only when max_label is below one.
    return [(identity,region) for identity,region in enumerate(ndi.find_objects(frame,max_label=top),1)
            if region is not None]


def confidence_rows(labels, raw, ring_width):
    rows=[]
    for t, frame in enumerate(labels):
        # Disk-backed frames are read through plain views of the same memory.
        frame=_plain(frame)
        intensity=_plain(raw[t])
        boxes=_boxes(frame) if ring_width>0 else None
        if boxes is not None:
            # Each identity's box lies inside its ring window, so its mask is
            # a view of the window's mask.
            for identity,region in boxes:
                ys,xs=region
                bounds=(slice(max(0,ys.start-ring_width),min(frame.shape[0],ys.stop+ring_width)),
                        slice(max(0,xs.start-ring_width),min(frame.shape[1],xs.stop+ring_width)))
                window=frame[bounds]
                mask=window==identity
                ring=ndi.binary_dilation(mask,structure=CROSS,iterations=ring_width)&~mask
                background=intensity[bounds][ring & (window==0)]
                local_mask=mask[ys.start-bounds[0].start:ys.stop-bounds[0].start,
                                xs.start-bounds[1].start:xs.stop-bounds[1].start]
                median=float(np.median(background)) if background.size else 0.0
                mean=float(intensity[region][local_mask].mean())
                rows.append(dict(identity=identity,t=t,imagej_frame=t+1,area_px=int(local_mask.sum()),
                                 mean_intensity=mean,local_background_median=median,
                                 local_contrast_ratio=(mean+1.0)/(median+1.0)))
            continue
        for identity, region, local_mask in regions(frame):
            if ring_width > 0:
                ys,xs=region
                bounds=(slice(max(0,ys.start-ring_width),min(frame.shape[0],ys.stop+ring_width)),
                        slice(max(0,xs.start-ring_width),min(frame.shape[1],xs.stop+ring_width)))
                mask=frame[bounds]==identity
                ring=ndi.binary_dilation(mask,iterations=ring_width)&~mask
                background=intensity[bounds][ring & (frame[bounds]==0)]
            else:
                mask=frame==identity
                ring=ndi.binary_dilation(mask,iterations=ring_width)&~mask
                background=intensity[ring & (frame==0)]
            median=float(np.median(background)) if background.size else 0.0
            mean=float(intensity[region][local_mask].mean())
            rows.append(dict(identity=identity,t=t,imagej_frame=t+1,area_px=int(local_mask.sum()),
                             mean_intensity=mean,local_background_median=median,
                             local_contrast_ratio=(mean+1.0)/(median+1.0)))
    return rows


def _foreground(plain):
    """np.nonzero(plain) of a non-empty 2-D frame and the values there.

    Both list pixels in row-major order; the row-major positions of the
    nonzero pixels split by the row length into the same rows and columns
    (intp), which is about four times faster than the 2-D nonzero."""
    flat=plain.ravel()
    where=np.flatnonzero(flat!=0)
    ys,xs=np.divmod(where,plain.shape[1])
    return ys,xs,flat[where]


def frame_coordinates(frame):
    # A disk-backed frame is compared through a plain view of the same memory.
    plain=_plain(frame)
    if plain.ndim==2 and plain.dtype.kind in 'iu':
        if not plain.size:
            return
        # Every foreground pixel in row-major order, stably sorted by name: each
        # name's run holds np.nonzero(frame == name)'s pixels in the same order,
        # and names come out ascending, as sorted(np.unique(frame)) gives them.
        ys,xs,names=_foreground(plain)
        order=np.argsort(names,kind='stable')
        ys,xs,names=ys[order],xs[order],names[order]
        cuts=np.flatnonzero(names[1:]!=names[:-1])+1
        for start,stop in zip([0,*cuts.tolist()],[*cuts.tolist(),len(names)]):
            if start<stop:
                yield int(names[start]),np.column_stack((ys[start:stop],xs[start:stop]))
        return
    for identity,region,mask in regions(plain):
        coordinates=np.column_stack(np.nonzero(mask))
        coordinates[:,0]+=region[0].start
        coordinates[:,1]+=region[1].start
        yield identity,coordinates


def frame_centres(frame):
    """(name, np.mean(coordinates, axis=0)) for frame_coordinates' names, in its order.

    The mean adds integer coordinates in float64 and divides by the count. Every
    partial sum of a frame's coordinates is an integer far below 2**53, so it is
    exact in any order: one weighted bincount per axis gives the same sums, and
    the same division gives the same centres."""
    plain=_plain(frame)
    if (plain.ndim!=2 or plain.dtype.kind not in 'iu' or plain.size>=1<<26
            or (plain.size and ((plain.dtype.kind=='i' and int(plain.min())<0) or int(plain.max())>1<<20))):
        for identity,coordinates in frame_coordinates(frame):
            yield identity,np.mean(coordinates,axis=0)
        return
    if not plain.size:
        return
    ys,xs,names=_foreground(plain)
    if not len(ys):
        return
    names=names.astype(np.intp)
    counts=np.bincount(names)
    rows=np.bincount(names,weights=ys)
    columns=np.bincount(names,weights=xs)
    for identity in np.flatnonzero(counts[1:]).tolist():
        identity+=1
        count=int(counts[identity])
        yield identity,np.array([rows[identity],columns[identity]])/count


_smaller_held={}
# Positions, radii and the threshold below this keep float rounding far under the 1 px margin.
_MODERATE=float(1<<40)


def near_smaller(row, smaller, near_distance):
    """bool(gaps and min(gaps) <= near_distance) of the confidence table's edge gaps.

    With finite values min(gaps) <= near is any gap <= near, and a smaller object
    farther than both radii plus the threshold plus 1 px along either axis has a
    gap above the threshold, so only the other objects need their gap; those gaps
    are the original expression. Non-finite, very large or unusual values take
    the original list."""
    def original():
        gaps = [max(
            float(np.linalg.norm(row["position"] - other["position"]))
            - float(row["radius"]) - float(other["radius"]), 0.0)
            for other in smaller]
        return bool(gaps and min(gaps) <= near_distance)
    if not smaller:
        return False
    held=_smaller_held.get('list')
    if held is not smaller:
        try:
            positions=np.array([other["position"] for other in smaller],dtype=float)
            radii=np.array([float(other["radius"]) for other in smaller],dtype=float)
        except (TypeError,ValueError):
            positions=radii=None
        _smaller_held.clear()
        _smaller_held.update(list=smaller,positions=positions,radii=radii)
    positions,radii=_smaller_held['positions'],_smaller_held['radii']
    try:
        position=np.asarray(row["position"],dtype=float);radius=float(row["radius"])
    except (TypeError,ValueError):
        return original()
    if (positions is None or positions.ndim!=2 or position.shape!=positions.shape[1:]
            or not (np.isfinite(positions).all() and np.isfinite(radii).all() and np.isfinite(position).all()
                    and math.isfinite(radius) and math.isfinite(near_distance))
            or max(np.abs(positions).max(),np.abs(radii).max(),abs(radius),abs(near_distance),
                   np.abs(position).max())>=_MODERATE):
        return original()
    reach=radius+radii+near_distance+1.0
    close=np.flatnonzero((np.abs(positions-position)<=reach[:,None]).all(axis=1))
    for j in close.tolist():
        other=smaller[j]
        gap=max(float(np.linalg.norm(row["position"] - other["position"]))
                - float(row["radius"]) - float(other["radius"]), 0.0)
        if gap<=near_distance:
            return True
    return False


@contextmanager
def install(module, history=None, *extra_aliases):
    stats=dict(supported=False)
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()!=SOURCE:
        yield stats
        return
    names=('identity_confidence_table','multilayer_identity_confidence_table')
    originals={name:getattr(module,name) for name in names}
    namespace=dict(vars(module),_confidence_rows=confidence_rows,_frame_coordinates=frame_coordinates,_near_smaller=near_smaller)
    source=inspect.getsource(originals[names[0]])
    start=source.index('    rows: list[dict] = []')
    stop=source.index('    if not rows:',start)
    source=source[:start]+'    rows = _confidence_rows(labels, raw, ring_width)\n'+source[stop:]
    exec(compile(source,'<cropped-confidence-samples>','exec'),namespace)
    source=inspect.getsource(originals[names[1]])
    old='        for identity in sorted(set(map(int, np.unique(frame))) - {0}):\n            coordinates = np.column_stack(np.nonzero(frame == identity))'
    assert source.count(old)==1
    source=source.replace(old,'        for identity, coordinates in _frame_coordinates(frame):')
    old=('            gaps = [max(\n'
         '                float(np.linalg.norm(row["position"] - other["position"]))\n'
         '                - float(row["radius"]) - float(other["radius"]), 0.0)\n'
         '                for other in smaller]\n'
         '            nearby[identity].append(bool(gaps and min(gaps) <= near_distance))\n')
    assert source.count(old)==1
    source=source.replace(old,'            nearby[identity].append(_near_smaller(row, smaller, near_distance))\n')
    exec(compile(source,'<cropped-confidence-coordinates>','exec'),namespace)
    aliases=[]
    for name,original in originals.items():
        setattr(module,name,namespace[name])
        for target in (history,*extra_aliases):
            if target is not None and getattr(target,name,None) is original:
                aliases.append((target,name,original))
                setattr(target,name,namespace[name])
    stats['supported']=True
    try:
        yield stats
    finally:
        for name,original in originals.items():setattr(module,name,original)
        for target,name,original in aliases:setattr(target,name,original)
