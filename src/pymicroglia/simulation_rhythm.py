"""Optional optical rhythm test for recorded pixels and detected cell outlines.

Single-plane images only. Emission geometry is approximated by the supplied
detected outlines; no hidden anatomical truth or programmed rhythm is accepted.
Measurements are summed light with local background removed, never pixel means.
Numerical fitting lives in Circadian Workbench and is reached through workbench.
"""
from pathlib import Path
from collections.abc import Mapping
import hashlib
import numpy as np
from scipy import ndimage as ndi
from . import workbench

METHOD_VERSION='2026-10-15-shared-recording-optical-rhythm'
PATTERNS=('uniform','boundary','centre','thin')
BLURS=(1.5,2.5,3.5)


def parameter_choices():
    return dict(processing=['none','asls','mean3','asls_mean3'],selection=['pooled_recording','fixed'],pattern=list(PATTERNS))


def _patterns(mask,patterns):
    distance=ndi.distance_transform_edt(np.pad(mask,1))[1:-1,1:-1]
    edge=mask & ~ndi.binary_erosion(mask);fields=[]
    for name in patterns:
        if name=='uniform':field=mask.astype(float)
        elif name=='boundary':field=edge.astype(float)
        elif name=='centre':field=np.where(mask,distance**2,0.)
        elif name=='thin':field=np.divide(mask,distance**2,out=np.zeros(mask.shape),where=distance>0)
        else:raise ValueError('Unknown emission pattern: '+str(name))
        total=field.sum();fields.append(field/total if total else field)
    return np.array(fields)


def prepare_recording(raw,detected_outlines,*,patterns=PATTERNS,blur_grid_pixels=BLURS,annulus_pixels=(5,10)):
    """Prepare every detected identity and channel before shared optical selection.

    Returns numeric responses and measurements only, not a new tracked movie.
    Candidate order matches the accepted prototype (uniform, 2.5 px first).
    """
    raw=np.asarray(raw);labels=np.asarray(detected_outlines)
    if raw.ndim==3:raw=raw[:,None]
    if raw.ndim!=4 or labels.shape!=(raw.shape[0],*raw.shape[2:]):raise ValueError('Expected single-plane TCYX pixels and matched TYX detected outlines')
    if labels.dtype.kind not in 'biu' or np.any(labels<0):raise ValueError('Detected outlines must contain nonnegative integer cell identities')
    if raw.dtype.kind not in 'biuf' or (raw.dtype.kind=='f' and np.isinf(raw).any()):raise ValueError('Recorded pixels must be real numeric values without infinities')
    patterns=tuple(patterns);blurs=tuple(float(v) for v in blur_grid_pixels)
    if not patterns or len(set(patterns))!=len(patterns) or set(patterns)-set(PATTERNS):raise ValueError('Choose unique supported spatial patterns')
    if not blurs or len(set(blurs))!=len(blurs) or any(not np.isfinite(v) or v<=0 for v in blurs):raise ValueError('Choose unique positive Gaussian blur standard deviations')
    if len(annulus_pixels)!=2 or not np.isfinite(annulus_pixels).all() or not 0<float(annulus_pixels[0])<float(annulus_pixels[1]):raise ValueError('Background annulus radii must be finite, positive and increasing')
    order=[(p,b) for p in range(len(patterns)) for b in range(len(blurs))]
    if 'uniform' in patterns and 2.5 in blurs:
        first=(patterns.index('uniform'),blurs.index(2.5));order.remove(first);order.insert(0,first)
    candidates=[dict(pattern=patterns[p],blur_pixels=blurs[b]) for p,b in order]
    ids=sorted(int(v) for v in np.unique(labels) if v>0)
    if not ids:raise ValueError('No detected cell identities in outlines')
    entries=[];responses=[];areas=[];values=[];totals=[];backgrounds=[]
    for cell in ids:
        points=np.argwhere(labels==cell);lo=np.maximum(0,points[:,1:].min(0)-20);hi=np.minimum(labels.shape[1:],points[:,1:].max(0)+21)
        cropped=labels[:,lo[0]:hi[0],lo[1]:hi[1]];mask=cropped==cell;area=mask.sum((1,2)).astype(float)
        response=np.full((len(order),len(raw)),np.nan)
        measured=np.full((raw.shape[1],len(raw)),np.nan);summed=np.full_like(measured,np.nan);background=np.full_like(measured,np.nan)
        for f,outline in enumerate(mask):
            if not outline.any():continue
            distance=ndi.distance_transform_edt(~outline)
            ring=(distance>=annulus_pixels[0])&(distance<=annulus_pixels[1])&(cropped[f]==0)
            if not ring.any():continue
            fields=_patterns(outline,patterns)
            blurred=[ndi.gaussian_filter(fields,(0,sigma,sigma),mode='constant',truncate=4) for sigma in blurs]
            for j,(p,b) in enumerate(order):
                field=blurred[b][p];response[j,f]=field[outline].sum()-area[f]*np.median(field[ring])
            for channel in range(raw.shape[1]):
                image=raw[f,channel,lo[0]:hi[0],lo[1]:hi[1]]
                summed[channel,f]=image[outline].astype(float).sum();background[channel,f]=np.median(image[ring])
                measured[channel,f]=summed[channel,f]-area[f]*background[channel,f]
        for ch in range(raw.shape[1]):
            entries.append(dict(cell=cell,channel=ch,label=f'cell{cell}_ch{ch+1}'))
            values.append(measured[ch]);responses.append(response);areas.append(area)
            totals.append(summed[ch]);backgrounds.append(background[ch])
    return dict(entries=entries,values=np.asarray(values),responses=np.asarray(responses),outline_areas=np.asarray(areas),
                raw_totals=np.asarray(totals),background_median=np.asarray(backgrounds),candidates=candidates,
                measurement='Summed recorded light minus outline area times local-annulus median',
                annulus_pixels=list(annulus_pixels))


def _read_pixels(source):
    if isinstance(source,(str,Path)):
        import tifffile
        with tifffile.TiffFile(source) as t:
            series=t.series[0];axes=series.axes;shape=series.shape
            if 'Z' in axes and shape[axes.index('Z')]>1:raise ValueError('Simulation rhythm testing requires a single image plane, not a projection')
            raw=series.asarray()
            if 'Z' in axes:raw=np.squeeze(raw,axis=axes.index('Z'));axes=axes.replace('Z','')
            if axes=='CYX':raw=raw[None]
            elif axes=='YX':raw=raw[None,None]
            elif axes not in {'TYX','TCYX','QYX','QCYX'}:raise ValueError('TIFF axes must describe a single-plane time sequence')
        return raw
    return np.asarray(source)


def _fingerprint(array):
    array=np.ascontiguousarray(array)
    digest=hashlib.sha256();digest.update(str(array.shape).encode());digest.update(array.dtype.str.encode());digest.update(memoryview(array).cast('B'))
    return digest.hexdigest()


def test_simulation_rhythm(source,*,outlines,times_h=None,interval_hours=None,
                           processing='none',selection='pooled_recording',pattern=None,blur_pixels=None,
                           patterns=PATTERNS,blur_grid_pixels=BLURS,annulus_pixels=(5,10),
                           options=None,control=None,output_dir=None,progress=None):
    """Opt-in rhythm test using recorded pixels and saved detected outlines.

    source is a single-plane TIFF or TYX/TCYX array; outlines is a TYX identity
    TIFF or integer array from tracking. Provide times_h or interval_hours when
    the recording has no time metadata. Every cell/channel selects shared optics
    before preprocessing; emission periods remain independent. An instrumental
    control is retained using the same gate as the existing test_rhythm action.
    """
    from . import rhythm
    if control is None and isinstance(source,(str,Path)):control=rhythm._stored_control(source)
    if control is None:raise rhythm.ControlMissing('No instrumental control for simulation-based rhythm testing. Supply the measured control or run the existing recording controls first.')
    if not isinstance(control,Mapping):raise ValueError('control must be a measured instrumental-control mapping')
    raw=_read_pixels(source)
    if isinstance(outlines,(str,Path)):
        import tifffile
        labels=tifffile.imread(outlines)
    else:labels=np.asarray(outlines)
    if times_h is not None and interval_hours is not None:raise ValueError('Provide times_h or interval_hours, not both')
    if times_h is None and interval_hours is None and isinstance(source,(str,Path)):
        from .series import open_series
        with open_series(source) as opened:times_h=opened.meta.times_h
    if times_h is None:
        if interval_hours is None or isinstance(interval_hours,bool) or not np.isfinite(interval_hours) or interval_hours<=0:
            raise ValueError('A real recording clock is required: times_h or a positive interval_hours')
        times_h=np.arange(len(raw))*float(interval_hours)
    prepared=prepare_recording(raw,labels,patterns=patterns,blur_grid_pixels=blur_grid_pixels,annulus_pixels=annulus_pixels)
    names=[v['label'] for v in prepared['entries']]
    seed=(options or {}).get('seed',20261014)
    seeds=[seed+v['cell']*100+v['channel'] for v in prepared['entries']]
    result=workbench.test_simulation_recording(times_h,prepared['values'],prepared['responses'],prepared['outline_areas'],prepared['candidates'],
             labels=names,processing=processing,selection=selection,pattern=pattern,blur_pixels=blur_pixels,options=options,trace_seeds=seeds,progress=progress)
    lookup={v['label']:v for v in prepared['entries']}
    for row in result['rows']:row.update(cell=lookup[row['label']]['cell'],channel=lookup[row['label']]['channel'])
    result.update(instrumental_control=dict(control),measurement=prepared['measurement'],annulus_pixels=prepared['annulus_pixels'],
                  method_version=METHOD_VERSION,source_fingerprints=dict(recorded_pixels=_fingerprint(raw),detected_outlines=_fingerprint(labels),hours=_fingerprint(np.asarray(times_h,float))),
                  measured_values=prepared['values'],outline_areas=prepared['outline_areas'],raw_totals=prepared['raw_totals'],background_median=prepared['background_median'])
    if output_dir is not None:_save_result(result,output_dir)
    return result


def _clean(value):
    if isinstance(value,dict):return {k:_clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_clean(v) for v in value]
    if isinstance(value,np.ndarray):return _clean(value.tolist())
    if isinstance(value,np.generic):return _clean(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def _save_result(result,output_dir):
    import pandas as pd
    from auto_organotypic.io import write_json
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    summary=[];curves=[]
    indexes={name:i for i,name in enumerate(result['labels'])}
    for row in result['rows']:
        summary.append({k:v for k,v in row.items() if k not in {'significance','prediction','estimated_emission','processed_input'}})
        for f,t in enumerate(result['hours']):
            index=indexes[row['label']]
            curves.append(dict(label=row['label'],cell=row['cell'],channel=row['channel'],processing=row['processing'],hours=t,
                               recorded_total=result['raw_totals'][index,f],background_median=result['background_median'][index,f],
                               outline_area=result['outline_areas'][index,f],background_removed_total=result['measured_values'][index,f],
                               fit_input=row['processed_input'][f],predicted_recorded_light=row['prediction'][f],estimated_emission=row['estimated_emission'][f]))
    pd.DataFrame(summary).to_csv(out/'simulation_rhythm_results.csv',index=False)
    pd.DataFrame(curves).to_csv(out/'simulation_rhythm_traces.csv',index=False)
    # The document lives in the package's existing hidden metadata folder.
    metadata={k:v for k,v in result.items() if k not in {'rows','hours','measured_values','outline_areas','raw_totals','background_median'}}
    metadata['rows']=[{k:v for k,v in row.items() if k not in {'prediction','estimated_emission','processed_input'}} for row in result['rows']]
    metadata['trace_file']='simulation_rhythm_traces.csv';metadata['result_file']='simulation_rhythm_results.csv'
    write_json(out/'simulation_rhythm.json',_clean(metadata))
