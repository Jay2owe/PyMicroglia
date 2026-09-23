"""Original low-overlap event selection, ranked denominators and matched crops."""
from __future__ import annotations
import numpy as np
import pandas as pd
import tifffile
from skimage.segmentation import find_boundaries
from ..visualisation.labels import semantic_label
from .prepared import PreparedViews

def commas(value):return [x.strip() for x in str(value).split(',') if x.strip()]

def _event_definition(metric: str) -> str:
    if metric == 'jaccard':
        return "the fraction of the same cell's footprint retained from the previous frame"
    return semantic_label(metric).lower()

def _events_and_rates(data: pd.DataFrame, metric: str, quantile: float) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    if metric not in data:
        raise ValueError(f'--metrics names {metric!r}, which is not in cell_frame.csv')
    if not 0.0 < quantile < 1.0:
        raise ValueError('--quantile must be greater than 0 and less than 1')
    values = pd.to_numeric(data[metric], errors='coerce')
    finite = values.replace([np.inf, -np.inf], np.nan).notna()
    if not finite.any():
        raise ValueError(f'{metric} has no measurable transitions')
    threshold = float(values[finite].quantile(quantile))
    measured = data.loc[finite].copy()
    measured[metric] = values[finite]
    measured['detected_event'] = measured[metric] <= threshold
    rates = measured.groupby('identity', as_index=False).agg(measurable_transitions=(metric, 'size'), detected_transitions=('detected_event', 'sum'), first_measurable_hour=('hours', 'min'), last_measurable_hour=('hours', 'max'))
    rates['event_rate'] = rates['detected_transitions'] / rates['measurable_transitions']
    rates = rates.sort_values(['event_rate', 'measurable_transitions', 'identity'], ascending=[False, False, True], kind='mergesort').reset_index(drop=True)
    rates['event_rate_rank'] = np.arange(1, len(rates) + 1)
    events = measured[measured['detected_event']].copy()
    events = events.merge(rates[['identity', 'measurable_transitions', 'detected_transitions', 'event_rate', 'event_rate_rank']], on='identity', how='left', validate='many_to_one')
    events['event_threshold'] = threshold
    events['event_threshold_quantile'] = quantile
    events['event_metric'] = metric
    events['event_definition'] = _event_definition(metric)
    return (threshold, events, rates)

def _select_examples(events: pd.DataFrame, data: pd.DataFrame, metric: str, wanted: str) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    area = data[['identity', 'frame_index', 'area_px', 'hours']].rename(columns={'frame_index': 'from_frame_index', 'area_px': 'before_area_px', 'hours': 'before_hours'})
    candidates = events.merge(area, on=['identity', 'from_frame_index'], how='left', validate='many_to_one')
    candidates['largest_pair_area_px'] = candidates[['area_px', 'before_area_px']].max(axis=1)
    candidates = candidates.sort_values([metric, 'largest_pair_area_px', 'identity', 'frame_index'], ascending=[True, False, True, True], kind='mergesort')
    strongest = candidates.groupby('identity', sort=False, as_index=False).head(1)
    try:
        return strongest.head(max(0, int(wanted))).reset_index(drop=True)
    except ValueError:
        identities = [int(value) for value in commas(wanted)]
        chosen = strongest[strongest['identity'].astype(int).isin(identities)].copy()
        missing = sorted(set(identities) - set(chosen['identity'].astype(int)))
        if missing:
            raise ValueError('--cells includes identities with no detected transition: ' + ', '.join(map(str, missing)))
        order = {identity: index for index, identity in enumerate(identities)}
        chosen['requested_order'] = chosen['identity'].astype(int).map(order)
        return chosen.sort_values('requested_order').reset_index(drop=True)

def _crop_box(centre: tuple[float, float], side: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    side = min(int(side), int(shape[0]), int(shape[1]))
    top = int(round(float(centre[0]) - side / 2))
    left = int(round(float(centre[1]) - side / 2))
    top = min(max(top, 0), shape[0] - side)
    left = min(max(left, 0), shape[1] - side)
    return (top, top + side, left, left + side)

def _matched_crops(raw: np.ndarray, labels: np.ndarray, picked: pd.DataFrame, *, metric: str, padding_fraction: float) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[tuple[str, str]], pd.DataFrame]:
    if padding_fraction < 0:
        raise ValueError('--crop-padding cannot be negative')
    prepared: list[dict] = []
    largest_span = 1
    for _, event in picked.iterrows():
        identity = int(event['identity'])
        before_frame = int(event['from_frame_index'])
        after_frame = int(event['frame_index'])
        before_mask = labels[before_frame] == identity
        after_mask = labels[after_frame] == identity
        occupied = np.argwhere(before_mask | after_mask)
        if not occupied.size:
            continue
        low = occupied.min(axis=0)
        high = occupied.max(axis=0)
        largest_span = max(largest_span, int(np.max(high - low + 1)))
        prepared.append({'event': event, 'identity': identity, 'before_frame': before_frame, 'after_frame': after_frame, 'before_mask': before_mask, 'after_mask': after_mask, 'centre': tuple((low + high) / 2.0)})
    crop_side = max(3, int(np.ceil(largest_span * (1.0 + 2.0 * padding_fraction))))
    before_images: list[np.ndarray] = []
    after_images: list[np.ndarray] = []
    before_masks: list[np.ndarray] = []
    after_masks: list[np.ndarray] = []
    titles: list[tuple[str, str]] = []
    records = []
    for pair, item in enumerate(prepared):
        top, bottom, left, right = _crop_box(item['centre'], crop_side, labels.shape[1:])
        before_frame = item['before_frame']
        after_frame = item['after_frame']
        event = item['event']
        before_images.append(raw[before_frame, top:bottom, left:right])
        after_images.append(raw[after_frame, top:bottom, left:right])
        before_masks.append(item['before_mask'][top:bottom, left:right])
        after_masks.append(item['after_mask'][top:bottom, left:right])
        retained = 100.0 * float(event[metric])
        before_hour = float(event['before_hours'])
        titles.append((f"Cell {item['identity']} · before ({before_hour:g} h)", f"After ({float(event['hours']):g} h) · {retained:.0f}% retained"))
        records.append({'pair': pair, 'identity': item['identity'], 'before_frame_index': before_frame, 'after_frame_index': after_frame, 'before_hours': before_hour, 'after_hours': float(event['hours']), metric: float(event[metric]), 'crop_top': top, 'crop_bottom': bottom, 'crop_left': left, 'crop_right': right, 'crop_size_px': bottom - top})
    return (before_images, after_images, before_masks, after_masks, titles, pd.DataFrame(records))

def prepare(source,options):
    data=source.table('cell_frame');metric=str(options['metrics']);quantile=float(options['quantile'])
    threshold,events,rates=_events_and_rates(data,metric,quantile)
    picked=_select_examples(events,data,metric,str(options['cells']))
    measured=data.dropna(subset=[metric,'hours','identity']).copy()
    events=measured.loc[measured[metric]<=threshold].copy()
    order=rates.sort_values('event_rate_rank').identity.tolist()
    events['row_order']=events.identity.map({identity:i for i,identity in enumerate(order)})
    events['event_threshold']=threshold
    events=events.merge(rates[['identity','measurable_transitions','detected_transitions','event_rate','event_rate_rank']],on='identity',how='left',validate='many_to_one')
    events['event_threshold_quantile']=quantile;events['event_metric']=metric;events['event_definition']=_event_definition(metric)
    kept=[c for c in ['identity','frame_index','from_frame_index','hours',metric,'area_px','event_threshold','event_threshold_quantile','event_metric','event_definition','measurable_transitions','detected_transitions','event_rate','event_rate_rank','row_order'] if c in events]
    events=events[kept]
    spans=measured.groupby('identity',sort=False).hours.agg(['min','max']).reset_index();spans['row_order']=spans.identity.map({identity:i for i,identity in enumerate(order)})
    population=len(events)/int(rates.measurable_transitions.sum())
    rates=rates[['identity','detected_transitions','measurable_transitions']].copy()
    rates['rate']=rates.detected_transitions/rates.measurable_transitions
    rates=rates.sort_values(['rate','measurable_transitions','identity'],ascending=[False,False,True],kind='mergesort').reset_index(drop=True)
    rates['rank']=np.arange(1,len(rates)+1);rates['rate_percent']=100*rates.rate
    tiles=[];display_rows=[];crop_table=pd.DataFrame();low=high=np.nan
    raw_path=source.input_path('raw');label_path=source.input_path('labels')
    if raw_path and label_path and not picked.empty:
        labels=tifffile.imread(label_path);raw_full=tifffile.imread(raw_path)
        offsets=(data.source_imagej_frame-data.imagej_frame).dropna().astype(int).unique()
        if len(offsets)!=1:raise ValueError('raw-to-label frame offset is not constant')
        raw=raw_full[int(offsets[0]):int(offsets[0])+labels.shape[0]]
        if len(raw)!=len(labels):raise ValueError('raw and label stacks do not cover the same aligned frames')
        before,after,before_masks,after_masks,titles,crop_table=_matched_crops(raw,labels,picked,metric=metric,padding_fraction=float(options['crop_padding']))
        images=[image for pair in zip(before,after) for image in pair]
        if images:
            low,high=np.percentile(np.concatenate([np.asarray(image,dtype=float).ravel() for image in images]),[2.,99.5])
        for i,(a,b,ma,mb,title) in enumerate(zip(before,after,before_masks,after_masks,titles)):
            for state,image,mask,title_text in [('before',a,ma,title[0]),('after',b,mb,title[1])]:
                display_rows.append(dict(title=title_text,rows=int(image.shape[0]),columns=int(image.shape[1]),vmin=float(low),vmax=float(high),mask_px=int(np.count_nonzero(mask)),pair=i,state=state,outline_role='motility' if state=='before' else 'outline',before_reference_px=int(np.count_nonzero(ma)) if state=='after' else 0))
            tiles.append(dict(before=a,after=b,before_border=find_boundaries(ma,mode='outer'),after_border=find_boundaries(mb,mode='outer'),titles=title))
    tile_table=pd.DataFrame(display_rows).merge(crop_table,on='pair',how='left',validate='many_to_one') if display_rows else pd.DataFrame(columns=['pair','state'])
    return PreparedViews({'raster':dict(table=events,spans=spans,count=len(order)),
        'rates':dict(table=rates,reference=population),
        'tiles':dict(table=tile_table,tiles=tiles,low=low,high=high)},
        auxiliary={'event_rates_by_cell.csv':rates,'matched_crop_metadata.csv':crop_table,'matched_tile_display.csv':tile_table},
        wording=dict(title='Large cell-footprint changes through time',subtitle=f'{metric} at or below {threshold:.3g}: the lowest {quantile:.0%} of measurable transitions.',footnote='Rates use all measurable transitions as the denominator. Matched crops share a pixel scale and contrast range; orange repeats the preceding outline and teal shows the subsequent outline.')),None
