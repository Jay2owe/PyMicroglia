"""Measure the existing three-outline comparison in bounded image regions.

Channels share one expansion and one identity assignment. Numeric measurement
is delegated to expanded_outlines.analyse; the region includes every pixel and
every competing outline that can affect its aperture or background ring.
"""
from __future__ import annotations
from pathlib import Path
import inspect,json,shutil
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from .expanded_outlines import analyse,expand_frame
from .fixed_apertures import digest
from .mask_io import clock

METHOD_VERSION='2026-10-09-expanded-channels-bounded-v1'


def blocked_foreground(labels,foreground,validity=None,*,output,source_frame_offset=0,unclaimed=None):
    """Exclude every untracked cell and every unobserved camera pixel."""
    import contextlib
    offset,_=clock(source_frame_offset,.5)
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    with contextlib.ExitStack() as stack:
        movies=[stack.enter_context(tifffile.TiffFile(path)) for path in (labels,foreground)]
        tracked,cells=movies
        valid=stack.enter_context(tifffile.TiffFile(validity)) if validity is not None else None
        unknown=stack.enter_context(tifffile.TiffFile(unclaimed)) if unclaimed is not None else None
        count=len(tracked.pages);shape=tracked.pages[0].shape
        if len(cells.pages)!=count+offset or (valid is not None and len(valid.pages)!=count+offset):
            raise ValueError('Foreground and camera validity must cover all source frames')
        if unknown is not None and len(unknown.pages)!=count:raise ValueError('Unclaimed masks must align with tracked labels')
        if any(opened.pages[0].shape!=shape for opened in movies+([valid] if valid is not None else [])):raise ValueError('All masks must share camera dimensions')
        output.parent.mkdir(parents=True,exist_ok=True)
        def frames():
            for t in range(count):
                own=tracked.pages[t].asarray()
                blocked=((cells.pages[t+offset].asarray()>0)&(own==0))
                if valid is not None:blocked|=(valid.pages[t+offset].asarray()==0)
                if unknown is not None:blocked|=(unknown.pages[t].asarray()>0)&(own==0)
                yield blocked.astype(np.uint8)
        tifffile.imwrite(output,frames(),shape=(count,*shape),dtype=np.uint8,
            photometric='minisblack',compression='zlib',metadata={'axes':'TYX'})
    return output

def analyse_frame(pixels,labels,identity,*,blocked=None,options=None,_window=None):
    """Exact aperture/background values; distant-neighbour distance is a bound."""
    options=dict(options or {})
    defaults=inspect.signature(analyse).parameters
    margin=float(options.get('margin_px',defaults['margin_px'].default))
    outer=float(options.get('background_outer_px',defaults['background_outer_px'].default))
    pad=int(np.ceil(2*margin+outer+2))
    if _window is None:
        windows=ndi.find_objects(labels)
        if identity<1 or identity>len(windows) or windows[identity-1] is None:
            raise ValueError('identity must exist in the supplied frame')
        window=windows[identity-1]
    else:window=_window
    y0,y1=max(0,window[0].start-pad),min(labels.shape[0],window[0].stop+pad)
    x0,x1=max(0,window[1].start-pad),min(labels.shape[1],window[1].stop+pad)
    crop=np.s_[y0:y1,x0:x1]
    result=analyse(np.asarray(pixels[crop])[None],np.asarray(labels[crop])[None],identities=[identity],
        blocked=None if blocked is None else np.asarray(blocked[crop])[None],**options)
    row=result['table'].iloc[0].to_dict()
    # Another object outside the computational region cannot affect expansion
    # or background. Its distance remains explicitly unknown rather than infinity.
    if row['nearest_other_px']>=pad:
        row['nearest_other_px']=np.nan
        row['nearest_other_lower_bound_px']=float(pad)
    else:row['nearest_other_lower_bound_px']=row['nearest_other_px']
    return row,result['expanded_labels'][0],(y0,y1,x0,x1)

def measure(channels,labels,*,output_dir,source_frame_offset=0,frame_times=None,
            identities=(),blocked=None,options=None,claim='',frame_interval_h=.5):
    """Stream original, exclusive expanded and mean-background-corrected totals."""
    options=dict(options or {})
    allowed={'margin_px','background_inner_px','background_outer_px','min_background_px','background_statistic'}
    if set(options)-allowed:raise ValueError(f'Unknown options: {sorted(set(options)-allowed)}')
    if not channels:raise ValueError('At least one named quantitative channel is required')
    offset,cadence=clock(source_frame_offset,frame_interval_h)
    target=Path(output_dir);target.mkdir(parents=True,exist_ok=True)
    if any(target.iterdir()):raise FileExistsError(target)
    label_path=Path(labels)
    input_paths={'labels':label_path,**{f'channel:{name}':Path(path) for name,path in channels.items()}}
    if blocked is not None:input_paths['blocked']=Path(blocked)
    if frame_times is not None:input_paths['frame_times']=Path(frame_times)
    inputs=[dict(role=role,path=str(path.resolve()),sha256=digest(path)) for role,path in input_paths.items()]
    import contextlib
    rows=[]
    with contextlib.ExitStack() as stack:
        label_file=stack.enter_context(tifffile.TiffFile(label_path))
        channel_files={name:stack.enter_context(tifffile.TiffFile(path)) for name,path in channels.items()}
        count=len(next(iter(channel_files.values())).pages)
        if any(len(opened.pages)!=count for opened in channel_files.values()) or len(label_file.pages)+offset!=count:
            raise ValueError('Channel and label frame counts must match the declared leading offset')
        shape=tuple(label_file.pages[0].shape)
        if any(tuple(opened.pages[0].shape)!=shape for opened in channel_files.values()):
            raise ValueError('All channels must have the same camera dimensions as the labels')
        if shutil.disk_usage(target).free<int(np.prod(shape)*count*8)+256*1024**2:
            raise OSError('Insufficient space for outline and measurement outputs')
        blocked_file=stack.enter_context(tifffile.TiffFile(blocked)) if blocked is not None else None
        if blocked_file is not None and len(blocked_file.pages)!=len(label_file.pages):
            raise ValueError('Blocked foreground must align to label frames')
        chosen=tuple(dict.fromkeys(map(int,identities)))
        if not chosen:
            found=set()
            for page in label_file.pages:found.update(int(i) for i in np.unique(page.asarray()) if i)
            chosen=tuple(sorted(found))
        if not chosen:raise ValueError('There are no identities to measure')
        if frame_times is None:hours=np.arange(count)*cadence
        else:
            times=pd.read_csv(frame_times)
            if len(times)!=count or not np.array_equal(times.source_frame,np.arange(count)):
                raise ValueError('frame_times must cover every zero-based source frame')
            hours=times.hours.to_numpy(float)
            if not np.isfinite(hours).all() or np.any(np.diff(hours)<=0):raise ValueError('Acquisition hours must be finite and increasing')
        params=inspect.signature(analyse).parameters
        settings={name:options.get(name,params[name].default) for name in allowed}
        settings.update(source_frame_offset=offset,frame_interval_h=cadence)
        expanded_path=target/'expanded_labels.tif'
        writer=stack.enter_context(tifffile.TiffWriter(expanded_path,bigtiff=True))
        def frames():
            for source_index in range(count):
                index=source_index-offset
                a=label_file.pages[index].asarray() if index>=0 else np.zeros(shape,np.uint16)
                obstacle=blocked_file.pages[index].asarray().astype(bool) if blocked_file is not None and index>=0 else None
                grown=expand_frame(a,settings['margin_px'],obstacle)
                assert np.array_equal(grown[a>0],a[a>0])
                windows=ndi.find_objects(a)
                pixels={name:opened.pages[source_index].asarray() for name,opened in channel_files.items()}
                for identity in chosen:
                    window=windows[identity-1] if 0<identity<=len(windows) else None
                    present=window is not None
                    for name,image in pixels.items():
                        if present:
                            row,local,bounds=analyse_frame(image,a,identity,blocked=obstacle,options=options,_window=window)
                            y0,y1,x0,x1=bounds
                            assert np.array_equal(local==identity,grown[y0:y1,x0:x1]==identity)
                        else:
                            row=dict(mask_present=False,status='missing_mask',area_px=np.nan,expanded_area_px=np.nan,
                                integrated_density=np.nan,expanded_integrated_density=np.nan,corrected_expanded_integrated_density=np.nan)
                        row.update(identity=identity,channel=name,source_frame_index=source_index,frame_index=index,hours=float(hours[source_index]))
                        rows.append(row)
                if index>=0:yield grown
        writer.write(frames(),shape=(count-offset,*shape),dtype=label_file.pages[0].dtype,
            photometric='minisblack',compression='zlib',compressionargs={'level':1},metadata={'axes':'TYX'})
    table_path=target/'cell_frame.csv';pd.DataFrame(rows).to_csv(table_path,index=False)
    report=dict(method_version=METHOD_VERSION,code_sha256=digest(__file__),upstream_code_sha256=digest(inspect.getfile(analyse)),
        inputs=inputs,outputs=[dict(path=str(p.resolve()),sha256=digest(p)) for p in (expanded_path,table_path)],
        settings=settings,source_frame_offset=offset,source_frames=count,labels_frames=count-offset,
        identities=list(chosen),channels=list(channels),claim=claim,
        quantitative_processing='No intensity filtering; source values are measured by the existing expanded-outline arithmetic.',
        overlap_policy='Nearest original outline, with every supplied identity competing and original pixels retained.',
        background_policy='Existing local finite-pixel ring excludes all expanded outlines and blocked foreground; no fallback.',
        nearest_other_policy='Exact when found within the safe image region; otherwise a declared lower bound.')
    (target/'manifest.json').write_text(json.dumps(report,indent=2));return report
