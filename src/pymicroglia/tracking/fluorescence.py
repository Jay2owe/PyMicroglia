"""Native fluorescence input for Motion; learned refinement follows identities."""
from pathlib import Path
import json
import tempfile
import numpy as np
import tifffile
from auto_organotypic.rigid_registration import to_canvas
from .prepare import sha256
from .prepare_stream import build

METHOD_VERSION = '2026-10-14-motion-first-red-minus-green-v1'

def subtract(red, green):
    """Unsigned subtraction must saturate at zero rather than wrap around."""
    if red.shape != green.shape or red.dtype != np.uint16 or green.dtype != np.uint16:
        raise ValueError('Red and Green must be aligned uint16 camera images')
    return np.maximum(red.astype(np.int32)-green.astype(np.int32),0).astype(np.uint16)

def prepare(manifest, folder, *, stem, progress=None, scratch_dir=None, frame_interval_min=30.):
    """Subtract original channels, apply retained poses, and prepare Motion evidence.

    Signal-isolation pixels are for detection only. No U-Net, learned foreground
    gate, photon median rescaling, or leading-frame exclusion enters this route.
    """
    roles={r['role']:r for r in manifest['outputs']}
    required={'red_original','green_original','source_to_canvas','frame_clock'}
    if required-set(roles):raise ValueError('Original channels, poses and clock are required')
    for role in required:
        if sha256(roles[role]['path']) != roles[role]['sha256']:
            raise ValueError('Retained fluorescence source changed: '+role)
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    shape=tuple(manifest['canvas_shape'])
    poses=np.load(roles['source_to_canvas']['path'])
    if len(poses)!=shape[0]:raise ValueError('Camera poses and frames differ')
    signal=folder/'red_minus_green_tracking.tif'
    with tifffile.TiffFile(roles['red_original']['path']) as red, tifffile.TiffFile(roles['green_original']['path']) as green:
        if tuple(red.series[0].shape)!=tuple(manifest['native_shape']) or red.series[0].shape!=green.series[0].shape:
            raise ValueError('Original channel shapes differ from the source manifest')
        def frames():
            for t in range(shape[0]):
                image=to_canvas(subtract(red.pages[t].asarray(),green.pages[t].asarray()),poses[t],shape[1:])
                if progress is not None:progress('Red minus Green input',t+1,shape[0])
                yield image
        tifffile.imwrite(signal,frames(),shape=shape,dtype=np.uint16,photometric='minisblack',
            compression='zlib',compressionargs={'level':1},bigtiff=True,metadata={'axes':'TYX'})
    with tempfile.TemporaryDirectory(prefix='fluorescence-motion-',dir=scratch_dir) as temporary:
        import os
        (folder/'runtime_storage.json').write_text(json.dumps({'temporary_directory':temporary,'owner_pid':os.getpid()}))
        with tifffile.TiffFile(signal) as opened:
            raw=opened.series[0].asarray(out=str(Path(temporary)/'signal.dat'))
        try:
            prepared=build(stem,raw,None,folder/'inputs',frame_interval_min=frame_interval_min,
                dataset='vessel95_full_native_red_green',scratch_dir=temporary,
                save_measurement_raw=False,native_signal=True,progress=progress,registered_raw_path=signal)
        finally:raw._mmap.close()
    prepared.update(tracking_signal=str(signal.resolve()),frame_clock=roles['frame_clock']['path'],
        native_red=roles['red_original']['path'],native_green=roles['green_original']['path'],
        source_to_canvas=roles['source_to_canvas']['path'],method_version=METHOD_VERSION)
    (folder/'signal_isolation.json').write_text(json.dumps(dict(method_version=METHOD_VERSION,
        operation='max(original Red - original Green, 0), then shared registered camera pose',
        leading_frames_excluded=0,upstream_learned_mask=False,intensity_scale=1.0,
        signal_sha256=sha256(signal),inputs={r:roles[r] for r in required}),indent=2))
    return prepared

def foreground(labels, output, *, unclaimed=None):
    """Use Motion's complete foreground ledger to protect neighbouring cells."""
    from contextlib import ExitStack
    with ExitStack() as stack:
        tracked=stack.enter_context(tifffile.TiffFile(labels))
        other=stack.enter_context(tifffile.TiffFile(unclaimed)) if unclaimed else None
        shape=tracked.series[0].shape
        if other is not None and other.series[0].shape!=shape:raise ValueError('Motion ledgers differ in shape')
        def frames():
            for t,page in enumerate(tracked.pages):
                image=page.asarray()>0
                if other is not None:image|=other.pages[t].asarray()>0
                yield image
        tifffile.imwrite(output,frames(),shape=shape,dtype=bool,photometric='minisblack',
            compression='zlib',metadata={'axes':'TYX'})
    return str(Path(output).resolve())

def refine(signal, labels, output, *, weights, method='single_frame', options=None, threads=4):
    """Optional frame-wise refinement of Motion identities on Red-minus-Green."""
    from .refinement_methods import propose
    rows=[]
    with tifffile.TiffFile(signal) as raw,tifffile.TiffFile(labels) as tracked:
        if raw.series[0].shape!=tracked.series[0].shape:raise ValueError('Refinement needs aligned complete Motion masks')
        def frames():
            for t,page in enumerate(tracked.pages):
                baseline=page.asarray()
                refined,changes,description=propose(raw.pages[t].asarray().astype(np.float32)[None],baseline[None],
                    weights=weights,method=method,threads=threads,options=options)
                rows.append({'source_frame':t,'changed_pixels':int(np.count_nonzero(refined[0]!=baseline)),
                    'method':description})
                yield refined[0]
        tifffile.imwrite(output,frames(),shape=tracked.series[0].shape,dtype=tracked.series[0].dtype,
            photometric='minisblack',compression='zlib',metadata={'axes':'TYX'})
    Path(output).with_suffix('.json').write_text(json.dumps(dict(method=method,options=options,
        weights=str(weights),weights_sha256=sha256(weights),stage='after Motion',frames=rows),indent=2))
    return str(Path(output).resolve())
