"""Optional exclusive expanded outlines using the existing measurement arithmetic."""
from __future__ import annotations
import numpy as np
import pandas as pd
from ..declare import Column, Output, measurement
from ..expanded_channels import analyse_frame
from ..expanded_outlines import analyse
import inspect

METHOD_VERSION = '2026-10-13-expanded-intensity-option-v1'
DEFAULTS = {name: inspect.signature(analyse).parameters[name].default for name in (
    'margin_px', 'background_inner_px', 'background_outer_px', 'min_background_px', 'background_statistic')}
PRODUCES = tuple(Column(key, label, unit, 'reporter') for key, label, unit in (
    ('original_integrated_density', 'Original-outline total', 'camera units'),
    ('expanded_integrated_density', 'Expanded-outline total', 'camera units'),
    ('corrected_expanded_integrated_density', 'Expanded total, background corrected', 'camera units'),
    ('expanded_area_px', 'Expanded outline area', 'px'),
    ('expanded_background_per_pixel', 'Expanded-outline local background', 'camera units'),
    ('expanded_background_px', 'Expanded-outline background pixels', 'px'),
    ('expansion_clipped_px', 'Expansion prevented by other cells or obstacles', 'px')))


@measurement(name='expanded_intensity', description='Exclusive expanded total with optional local-background readout',
             requires=('labels', 'raw'), defaults=DEFAULTS, produces=PRODUCES,
             writes=(Output('expanded_intensity', grain=('identity', 'frame_index'), fold=True),))
def measure(context):
    from scipy import ndimage as ndi
    options = {**DEFAULTS, **context.module_params('expanded_intensity')}
    rows = []
    for frame in range(context.n_frames):
        labels = context.labels[frame]
        blocked = ~context.valid_frame(frame)
        if context.unclaimed is not None:
            blocked |= context.unclaimed[frame] > 0
        windows = ndi.find_objects(labels)
        for identity in context.identities:
            window = windows[identity-1] if 0 < identity <= len(windows) else None
            if window is None:
                continue
            row, _, _ = analyse_frame(context.raw[frame], labels, identity,
                                      blocked=blocked, options=options, _window=window)
            rows.append(dict(identity=identity, frame_index=frame,
                original_integrated_density=row['integrated_density'],
                expanded_integrated_density=row['expanded_integrated_density'],
                corrected_expanded_integrated_density=row['corrected_expanded_integrated_density'],
                expanded_area_px=row['expanded_area_px'],
                expanded_background_per_pixel=row['background_per_pixel'],
                expanded_background_px=row['background_px'], expansion_clipped_px=row['clipped_expansion_px']))
    return {'expanded_intensity': pd.DataFrame(rows, columns=['identity', 'frame_index', *[c.name for c in PRODUCES]])}
