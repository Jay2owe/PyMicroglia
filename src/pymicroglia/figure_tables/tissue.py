"""Prepare all six original tissue maps before drawing."""
import numpy as np
import pandas as pd
import tifffile
from . import territory_values as values
from .tissue_rhythms import analyse_rhythm_maps
from .prepared import PreparedViews
from .metric_maps import cell_metric
from ..visualisation.labels import describe


def _pixels(array,low,high,mask=None,column='value'):
    shown=np.asarray(array,dtype=float).copy()
    if mask is not None:shown[~mask]=np.nan
    yy,xx=np.where(np.isfinite(shown))
    table=pd.DataFrame(dict(row=yy,column=xx,value=shown[yy,xx],display_minimum=float(low),display_maximum=float(high))).rename(columns={'value':column})
    return shown,table


def prepare(source,options):
    path=source.input_path('labels')
    if path is None:raise ValueError('Tissue maps require the recorded labels')
    labels=tifffile.imread(path);owners=source.stack('owner_count.tif')
    if labels.ndim!=3 or owners.shape!=labels.shape[1:]:raise ValueError('Owner count and label fields must match')
    frame=source.table('cell_frame');hours=_frame_hours(source.table('frame_summary'),len(labels));span=float(hours[-1]-hours[0])
    assignment=values.metric_assignment(options['map_assignment']);luts=_panel_luts(list(options['map_luts']))
    field={**source.field,'width':source.scale.length(source.field['width']),'height':source.scale.length(source.field['height'])}
    views={};aux={};all_pixels=[]
    def record(key,array,table,column,label,limits,**extras):
        normal=_plotted_pixels(key,table,column);all_pixels.append(normal);aux[key+'_pixels.csv']=table
        views[key]=dict(table=normal,array=array,field=field,label=label,limits=limits,cmap=luts[key] or 'viridis',unit=source.scale.length_unit,**extras)
    limit=max(span,np.finfo(float).eps)
    first=values.first_coverage_time(labels,hours);array,table=_pixels(first,0,limit,column='first_covered_hours')
    table['first_covered_frame_index']=np.argmax(labels>0,axis=0)[table.row.to_numpy(int),table.column.to_numpy(int)]
    record('first_coverage',array,table,'first_covered_hours','First covered (h)',(0,limit))
    cumulative=values.cumulative_occupancy(labels,hours)[-1];array,table=_pixels(cumulative,0,limit,np.any(labels>0,axis=0))
    table['frame_index']=len(labels)-1;table['hours']=float(hours[-1]);record('cumulative_occupancy',array,table,'value','Cumulative occupied time (h)',(0,limit))
    maximum=int(np.nanmax(owners));limits=(1.,float(max(maximum,1)+(maximum==1)))
    array,table=_pixels(owners,*limits,owners>=1,column='owner_count');table['minimum_owner_count']=1
    record('unique_cells',array,table,'owner_count','Distinct cells',limits)
    speed=_speed_values(frame,options['map_summary']);array,table,cells,limits=cell_metric(labels,speed,metric=SPEED_METRIC,assignment=assignment,range_mode=options['map_range'])
    aux['speed_cells.csv']=cells.assign(summary_operation=options['map_summary']);record('speed',array,table,'value',str(options['map_summary']).capitalize()+' cell speed ('+describe(SPEED_METRIC).unit_text(source.interval)+')',limits)
    rhythm,fits,_,params=analyse_rhythm_maps(source,options,frame,[INTENSITY_PERIOD_METRIC])
    array,table,cells,limits=cell_metric(labels,rhythm[INTENSITY_PERIOD_METRIC],metric=INTENSITY_PERIOD_METRIC,assignment=assignment,range_mode='full',limits=tuple(params['period_search_hours']),allow_empty=True)
    intensity=fits.loc[fits.metric.eq('corrected_mean')];supported,exploratory=_period_support_masks(labels,intensity)
    yy=table.row.to_numpy(int);xx=table.column.to_numpy(int)
    table['has_supported_contributor']=supported[yy,xx];table['has_exploratory_contributor']=exploratory[yy,xx]
    table['period_support']=np.select([table.has_supported_contributor&table.has_exploratory_contributor,table.has_supported_contributor,table.has_exploratory_contributor],['mixed support','supported period','exploratory period'],default='excluded')
    # Contour geometry belongs to preparation, so a single-view redraw does no science.
    from matplotlib.figure import Figure
    outlines=[]
    if supported.any():
        ax=Figure().add_subplot();contour=ax.contour(supported.astype(float),levels=[.5],origin='upper',extent=(0,field['width'],field['height'],0));outlines=[segment.copy() for segment in contour.allsegs[0]]
    record('significant_period',array,table,'value','Summary of contributing cell periods (h)',limits,outlines=outlines)
    aux['significant_period_cells.csv']=cells.merge(intensity[['identity','period_map_displayed','period_map_supported','period_map_status','period_map_exclusion']],on='identity',how='left',validate='one_to_one')
    aux['rhythm_map_fits.csv']=fits;aux['statistics.csv']=fits.copy()
    events=source.table('history_merge_split_events',optional=True)
    counts,events_used=values.split_event_areas(labels,events) if events is not None else (np.zeros(labels.shape[1:]),pd.DataFrame())
    maximum=int(counts.max());limits=(1.,float(maximum+(maximum==1)))
    if maximum:
        array,table=_pixels(counts,*limits,counts>0,column='split_events');message=''
    else:
        array=np.full(labels.shape[1:],np.nan);table=pd.DataFrame(columns=['row','column','split_events','display_minimum','display_maximum']);limits=(0,1);message='Split-event record unavailable' if events is None else 'No recorded contact-separation events'
    record('splitting_events',array,table,'split_events','Split transitions per pixel',limits,message=message);aux['splitting_events.csv']=events_used
    aux['figure_data.csv']=pd.concat(all_pixels,ignore_index=True,sort=False)
    return PreparedViews(views,auxiliary=aux,wording=dict(title='Cell coverage and motion across the tissue field',footnote=f'Overlap rule: {values.MAP_ASSIGNMENT_LABELS[assignment]}. Period colours summarize contributing cells; they are not a rhythm detected at each pixel. Black boundaries mark supported contributors; other significant estimates remain exploratory. Split events are tracker contact separations, not evidence of cell division.')),None

SPEED_METRIC = "speed"

INTENSITY_PERIOD_METRIC = "reporter_period_hours"

PANEL_ORDER = (
    "first_coverage",
    "cumulative_occupancy",
    "unique_cells",
    "speed",
    "significant_period",
    "splitting_events",
)

def _frame_hours(frame_summary: pd.DataFrame, n_frames: int) -> np.ndarray:
    """One complete, consistent elapsed-hour coordinate for the label movie."""
    required = {"frame_index", "hours"}
    missing = sorted(required - set(frame_summary.columns))
    if missing:
        raise ValueError("frame_summary.csv is missing " + ", ".join(missing))
    clock = frame_summary[["frame_index", "hours"]].copy()
    clock["frame_index"] = pd.to_numeric(clock["frame_index"], errors="coerce")
    clock["hours"] = pd.to_numeric(clock["hours"], errors="coerce")
    if clock["frame_index"].duplicated().any():
        raise ValueError("frame_summary.csv needs one time for each frame")
    clock = clock.sort_values("frame_index")
    expected = np.arange(n_frames)
    if (not np.array_equal(clock["frame_index"].to_numpy(), expected)
            or not np.isfinite(clock["hours"]).all()
            or np.any(np.diff(clock["hours"].to_numpy(float)) <= 0)):
        raise ValueError(
            "tissue tectonics needs one finite, increasing time for every label frame"
        )
    return clock["hours"].to_numpy(float)

def _speed_values(cell_frame: pd.DataFrame, summary: str) -> pd.Series:
    """One median or mean speed per cell."""
    method = str(summary).strip().lower()
    if method not in {"median", "mean"}:
        raise ValueError("--map-summary must be median or mean for the speed map")
    required = {"identity", "frame_index", SPEED_METRIC}
    missing = sorted(required - set(cell_frame.columns))
    if missing:
        raise ValueError("cell_frame.csv is missing " + ", ".join(missing))
    work = cell_frame[["identity", "frame_index", SPEED_METRIC]].copy()
    work[SPEED_METRIC] = pd.to_numeric(work[SPEED_METRIC], errors="coerce")
    grouped = work.sort_values(["identity", "frame_index"]).groupby("identity")[SPEED_METRIC]
    values = grouped.median() if method == "median" else grouped.mean()
    values.index = values.index.astype(int)
    values.name = SPEED_METRIC
    if not np.isfinite(values).any():
        raise ValueError("cell_frame.csv has no finite per-cell speed values")
    return values

def _panel_luts(requested: list[str]) -> dict[str, str | None]:
    """One optional colour map per canonical panel, in canonical order."""
    if not requested:
        return dict.fromkeys(PANEL_ORDER)
    if len(requested) == 1:
        return dict.fromkeys(PANEL_ORDER, requested[0])
    if len(requested) != len(PANEL_ORDER):
        raise ValueError(
            f"--map-luts needs one colour map or {len(PANEL_ORDER)} maps in panel order"
        )
    return dict(zip(PANEL_ORDER, requested))

def _plotted_pixels(panel: str, table: pd.DataFrame, value_column: str) -> pd.DataFrame:
    """Normalise one panel's exact pixel encodings for figure_data.csv."""
    result = table.copy()
    if value_column in result and value_column != "value":
        result = result.rename(columns={value_column: "value"})
    result.insert(0, "panel", panel)
    result.insert(1, "panel_order", PANEL_ORDER.index(panel) + 1)
    result["measure"] = value_column
    return result

def _period_support_masks(labels: np.ndarray, fits: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Pixels ever occupied by supported and exploratory displayed periods."""
    shape = labels.shape[1:]

    def occupied(rows: pd.DataFrame) -> np.ndarray:
        identities = rows.identity.dropna().astype(int).unique()
        if not len(identities):
            return np.zeros(shape, dtype=bool)
        return np.any(np.isin(labels, identities), axis=0)

    shown = fits[fits.period_map_displayed.fillna(False).astype(bool)]
    supported = shown[shown.period_map_supported.fillna(False).astype(bool)]
    exploratory = shown[~shown.period_map_supported.fillna(False).astype(bool)]
    return occupied(supported), occupied(exploratory)
