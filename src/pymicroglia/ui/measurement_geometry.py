"""Display the exact expanded aperture and finite background pixels used for totals."""
import math
from pathlib import Path
import numpy as np


def parameters(movie):
    display = movie.get('display', {})
    if display.get('outline_settings'):
        return display['outline_settings']
    return next((row['parameters'] for row in movie.get('modules', [])
                 if row.get('module')=='expanded_intensity' and row.get('status')=='done'), None)


def display_options(movie):
    if parameters(movie) is None:
        return movie.get('display', {})
    return {'preferred_metric':'corrected_expanded_integrated_density',
            'comparison_metrics':['original_integrated_density','expanded_integrated_density',
                                  'corrected_expanded_integrated_density'], **movie.get('display', {})}


def layers(root, movie, current, frame, identity=None):
    from scipy.ndimage import find_objects
    from .exploration_images import plane, declared_source, geometry
    from ..measure.expanded_outlines import expand_frame, background_mask
    from ..measure.mask_geometry import observation
    options = parameters(movie)
    if options is None:
        return None
    display = movie.get('display', {})
    def resolve(value):
        path = Path(value)
        return path if path.is_absolute() else root/path
    blocked = np.zeros(current.shape, bool)
    if display.get('blocked'):
        blocked |= plane(resolve(display['blocked']), frame) > 0
    else:
        unknown = declared_source(root, movie, 'unclaimed')
        if unknown is not None:
            blocked |= plane(unknown, frame)>0
        valid = declared_source(root, movie, 'valid_mask')
        if valid is not None:
            import tifffile
            with tifffile.TiffFile(valid) as opened:
                shape = opened.series[0].shape
            valid_frame = 0 if len(shape)==2 or shape[0]==1 else frame
            blocked |= ~plane(valid, valid_frame).astype(bool)
    base = plane(resolve(display['base_labels']), frame) if display.get('base_labels') else current
    grown = current if display.get('measurement_labels') else expand_frame(base, options['margin_px'], blocked)
    raw = declared_source(root, movie, 'raw')
    offset = int(movie.get('spec',{}).get('source_frame_offset',0))
    finite = np.isfinite(plane(raw, frame+offset))
    backgrounds = []
    windows = find_objects(grown)
    chosen = [identity] if identity is not None else [i+1 for i, window in enumerate(windows) if window is not None]
    pad = int(math.ceil(options['background_outer_px']))+2
    for cell in chosen:
        if not 0 < cell <= len(windows) or windows[cell-1] is None:
            continue
        window = windows[cell-1]
        y0,y1=max(0,window[0].start-pad),min(grown.shape[0],window[0].stop+pad)
        x0,x1=max(0,window[1].start-pad),min(grown.shape[1],window[1].stop+pad)
        crop = np.s_[y0:y1,x0:x1]
        ring = background_mask(grown[crop], cell, options['background_inner_px'],
                               options['background_outer_px'], blocked[crop]) & finite[crop]
        yy, xx = np.nonzero(ring)
        if len(yy):
            backgrounds.append(dict(identity=cell, **observation(yy+y0, xx+x0)))
    return {'accepted':geometry(grown, identity), 'base':geometry(base, identity), 'background':backgrounds}
