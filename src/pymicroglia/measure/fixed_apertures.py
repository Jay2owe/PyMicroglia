"""Fixed-size intensity footprints translated along supplied cell centroids."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates

METHOD_VERSION = '2026-10-06-fixed-centroid-apertures-v1'


def footprint(area_px: float, shape: str, area_fraction: float = 1.) -> np.ndarray:
    """Return fixed sampling offsets; every sample has unit area weight."""
    if not np.isfinite(area_px) or area_px <= 0 or not 0 < area_fraction <= 1:
        raise ValueError('Positive reference area and 0 < area_fraction <= 1 required')
    target = area_px * area_fraction
    if shape == 'circle':
        radius = np.sqrt(target / np.pi)
        axis = np.arange(-int(np.ceil(radius)), int(np.ceil(radius)) + 1)
        y, x = np.meshgrid(axis, axis, indexing='ij')
        keep = y*y + x*x <= radius*radius
        return np.column_stack([y[keep], x[keep]]).astype(float)
    if shape == 'square':
        side = max(1, int(np.floor(np.sqrt(target) + .5)))
        axis = np.arange(side, dtype=float) - (side - 1) / 2
        y, x = np.meshgrid(axis, axis, indexing='ij')
        return np.column_stack([y.ravel(), x.ravel()])
    raise ValueError('shape must be circle or square')


def sample(frame: np.ndarray, y: float, x: float, offsets: np.ndarray) -> dict:
    """Bilinear translation preserves total footprint weight at fractional centres.

    No clipping: an aperture extending beyond the field is missing, rather
    than becoming smaller. Pixel interpolation is spatial measurement only.
    """
    coordinates = offsets + [y, x]
    area = len(offsets)
    result = dict(aperture_area_px=area, signal_mean=np.nan,
                  integrated_density=np.nan, status='missing_centroid')
    if not np.isfinite(coordinates).all():
        return result
    height, width = frame.shape
    if ((coordinates < 0).any() or (coordinates[:, 0] > height - 1).any()
            or (coordinates[:, 1] > width - 1).any()):
        result['status'] = 'outside_field'
        return result
    values = map_coordinates(np.asarray(frame, dtype=float), coordinates.T,
                             order=1, mode='constant', cval=np.nan, prefilter=False)
    if not np.isfinite(values).all():
        result['status'] = 'invalid_pixels'
        return result
    result.update(signal_mean=float(values.mean()), integrated_density=float(values.sum()),
                  status='ok')
    return result


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def measure(movies, *, output_dir, shapes=('circle',), area_fraction=1., raw_scale=1., claim=''):
    """Measure native TIFFs using centroid CSVs keyed by identity and frame.

    Each movie declares stem, raw, centroids, and optional raw_sha256 and
    masks. Centroid CSVs require identity, frame_index, hours, area_px,
    centroid_y_px and centroid_x_px. Optional component identifies that
    cell in the masks for overlap diagnostics. Existing output is refused.
    """
    import tifffile
    from .signal import centered_frame_mean
    shapes = tuple(shapes)
    if not shapes or len(set(shapes)) != len(shapes) or not set(shapes) <= {'circle','square'}:
        raise ValueError('Select distinct circle and/or square shapes')
    out = Path(output_dir).resolve()
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    outputs = []; inputs = []; tables = []
    for movie in movies:
        raw = Path(movie['raw']).resolve(); centroids = Path(movie['centroids']).resolve()
        raw_hash = digest(raw)
        if movie.get('raw_sha256') and raw_hash != movie['raw_sha256']:
            raise ValueError('Raw source fingerprint mismatch: '+str(raw))
        inputs += [dict(path=str(raw),sha256=raw_hash),dict(path=str(centroids),sha256=digest(centroids))]
        c = pd.read_csv(centroids)
        required = {'identity','frame_index','hours','area_px','centroid_y_px','centroid_x_px'}
        if not required <= set(c): raise ValueError('Centroid table is missing '+str(required-set(c)))
        if c.duplicated(['identity','frame_index']).any(): raise ValueError('Duplicate centroid keys')
        reference = c[c.area_px.gt(0)].groupby('identity').area_px.median()
        if set(c.identity) != set(reference.index): raise ValueError('Cell without a positive reference area')
        offsets = {(int(identity), shape): footprint(area, shape, area_fraction)
                   for identity,area in reference.items() for shape in shapes}
        masks = None
        if movie.get('masks'):
            maskpath = Path(movie['masks']).resolve()
            masks = tifffile.TiffFile(maskpath)
            inputs.append(dict(path=str(maskpath),sha256=digest(maskpath)))
        rows=[]
        try:
            with tifffile.TiffFile(raw) as tf:
                pages=tf.series[0].pages
                if c.frame_index.min()<0 or c.frame_index.max()>=len(pages):
                    raise ValueError('Centroid frame index outside source')
                for frame, group in c.groupby('frame_index',sort=True):
                    pixels=pages[int(frame)].asarray().astype(np.float32)*raw_scale
                    labels=masks.series[0].pages[int(frame)].asarray() if masks else None
                    if labels is not None and labels.shape!=pixels.shape: raise ValueError('Masks and raw differ in shape')
                    for row in group.to_dict('records'):
                        identity=int(row['identity']);y=row['centroid_y_px'];x=row['centroid_x_px']
                        for shape in shapes:
                            stencil=offsets[identity,shape];measured=sample(pixels,y,x,stencil)
                            qc=dict(own_mask_fraction=np.nan,other_foreground_fraction=np.nan)
                            component=row.get('qc_component',row.get('component',0))
                            if labels is not None and measured['status']=='ok' and component>0:
                                coords=(stencil+[y,x]).T;component=int(component)
                                own=map_coordinates((labels==component).astype(float),coords,order=1,prefilter=False)
                                other=map_coordinates(((labels>0)&(labels!=component)).astype(float),coords,order=1,prefilter=False)
                                qc.update(own_mask_fraction=float(own.mean()),other_foreground_fraction=float(other.mean()))
                            rows.append(dict(stem=movie['stem'],identity=identity,frame_index=int(frame),
                                hours=row['hours'],shape=shape,reference_mask_area_px=float(reference.loc[identity]),
                                area_fraction=area_fraction,centroid_y_px=y,centroid_x_px=x,**measured,**qc))
        finally:
            if masks: masks.close()
        table=pd.DataFrame(rows);table['integrated_density_mean3']=np.nan
        for _,indices in table.groupby(['identity','shape']).groups.items():
            ordered=table.loc[indices].sort_values('frame_index')
            table.loc[ordered.index,'integrated_density_mean3']=centered_frame_mean(
                ordered.integrated_density.to_numpy(float),ordered.frame_index.to_numpy(int))
        path=out/(movie['stem']+'_cell_frame.csv');table.to_csv(path,index=False)
        outputs.append(dict(path=str(path),sha256=digest(path)));tables.append(table)
    combined=pd.concat(tables,ignore_index=True)
    combined_path=out/'cell_frame.csv';combined.to_csv(combined_path,index=False)
    outputs.append(dict(path=str(combined_path),sha256=digest(combined_path)))
    manifest=dict(method_version=METHOD_VERSION,claim=claim,shapes=shapes,area_fraction=area_fraction,
        raw_scale=raw_scale,spatial_sampling='bilinear; fixed unit-weight offset stencil',
        missing_policy='no centroid interpolation; no edge clipping; no variable mask exclusions',
        inputs=inputs,outputs=outputs,rows=len(combined),valid_rows=int(combined.status.eq('ok').sum()))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    from pymicroglia._results import register_outputs
    register_outputs(out, stage='measure_fixed_apertures', inputs=inputs,
                     settings=dict(method_version=METHOD_VERSION,shapes=shapes,
                                   area_fraction=area_fraction,raw_scale=raw_scale))
    return manifest
