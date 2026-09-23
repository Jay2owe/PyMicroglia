"""Prepare saved radial occupancy without drawing or new fits."""
import numpy as np
import pandas as pd
from .prepared import PreparedViews

def _selected_identities(frame: pd.DataFrame, requested: str | int) -> list[int]:
    """Resolve the shared --cells convention against observed support.

    ``requested`` is the figure's already-resolved ``cells`` option: a count of
    the longest-observed identities, or an explicit comma-separated list. It is
    passed in rather than read here so that the figure that honours the flag is
    the figure that declares it.
    """
    if 'observed_frames' in frame:
        support = frame.groupby('identity')['observed_frames'].max().sort_values(ascending=False)
    else:
        support = frame.groupby('identity').size().sort_values(ascending=False)
    requested = str(requested)
    try:
        return [int(value) for value in support.index[:max(1, int(requested))]]
    except ValueError:
        wanted = [int(piece) for piece in [v.strip() for v in requested.split(',') if v.strip()]]
        missing = sorted(set(wanted) - set((int(value) for value in support.index)))
        if missing:
            raise ValueError(f'--cells includes identities not in this table: {missing}')
        return wanted

def prepare(source,options):
    sholl = source.table('sholl.csv')
    metric = options.get('metrics')
    if metric not in {'occupancy', 'intersections'}:
        raise ValueError('--metrics must be occupancy or intersections')
    scaling = options.get('scaling')
    if 'scaling' in sholl.columns:
        available = sorted(sholl['scaling'].dropna().unique())
        if scaling not in available:
            raise ValueError(f"--scaling must be one of {', '.join(available)}; this run has no {scaling!r} rings")
        sholl = sholl[sholl['scaling'] == scaling].copy()
    elif scaling != 'cell':
        raise ValueError("this run's sholl.csv has no 'scaling' column, so it holds one unlabelled ring width; re-run the analysis or drop --scaling")
    cells = _selected_identities(sholl, options.get('cells'))
    data = sholl[sholl['identity'].isin(cells)].copy()
    full_ring_count = int(data['ring'].max()) + 1
    requested_rings = options.get('ring_count', options.get('rings'))
    if requested_rings is not None:
        if requested_rings < 1:
            raise ValueError('--rings needs a positive number of inner rings')
        data = data[data['ring'] < min(int(requested_rings), full_ring_count)].copy()
    unit_per_pixel = float(source.scale.microns_per_pixel) if source.scale.calibrated else 1.0
    radius_inner_px = data['radius_inner'].to_numpy(float) / unit_per_pixel
    radius_outer_px = data['radius_outer'].to_numpy(float) / unit_per_pixel
    expected_annulus_px = np.pi * (radius_outer_px ** 2 - radius_inner_px ** 2)
    data['annulus_support_fraction'] = np.minimum(data['annulus_px'].to_numpy(float) / np.maximum(expected_annulus_px, 1e-12), 1.0)
    data = data[data['annulus_support_fraction'] >= 0.5].copy()
    required=[column for column in ['identity','frame_index','hours','scaling','ring','radius_inner','radius_outer','occupancy','intersections','annulus_px','annulus_support_fraction'] if column in data]
    grids=[]
    for identity in cells:
        group=data.loc[data.identity.eq(identity)]
        pivot=group.pivot(index='ring',columns='hours',values=metric)
        if pivot.empty:continue
        radii=((pivot.index.to_numpy(float)+.5)/full_ring_count*100 if scaling=='cell' else group.groupby('ring')[['radius_inner','radius_outer']].mean().mean(axis=1).reindex(pivot.index).to_numpy(float))
        grids.append(dict(identity=identity,values=pivot.to_numpy(float),hours=pivot.columns.to_numpy(float),radii=radii))
    if not grids:raise ValueError('No supported radial bands in the selected cells')
    table=data[required]
    return PreparedViews({'kymograph':dict(table=table,grids=grids,metric=metric,maximum=1. if metric=='occupancy' else max(float(data[metric].max()),1.),
        ylabel='Distance from soma (% of current 95% reach)' if scaling=='cell' else f'Distance from soma ({source.scale.length_unit})')},
        wording=dict(subtitle=f'{len(cells)} selected cells; radial bands with less than half their full geometric area available are omitted.')),None
