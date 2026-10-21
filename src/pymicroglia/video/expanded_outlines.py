"""Render the three measured totals below original and expanded cell outlines."""
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
from ..measure.fixed_apertures import digest


def cell_video(measurement_dir, *, identity, output_dir, reporter="gray", fps=4., frame_width_px=1280,
               channel=None,prefix='',workspace=None):
    """Render source-linked cards; optional workspace keeps large stacks on disk."""
    if workspace is None:
        return _cell_video(measurement_dir,identity=identity,output_dir=output_dir,reporter=reporter,fps=fps,
            frame_width_px=frame_width_px,channel=channel,prefix=prefix)
    import tempfile
    with tempfile.TemporaryDirectory(prefix='cell-video-',dir=workspace) as scratch:
        maps=[]
        try:
            return _cell_video(measurement_dir,identity=identity,output_dir=output_dir,reporter=reporter,fps=fps,
                frame_width_px=frame_width_px,channel=channel,prefix=prefix,_workspace=Path(scratch),_maps=maps)
        finally:
            for array in maps:array._mmap.close()


def _cell_video(measurement_dir, *, identity, output_dir, reporter="gray", fps=4., frame_width_px=1280,
               channel=None,prefix='',_workspace=None,_maps=None):
    """Use the shared cell-card renderer and decode every delivered frame."""
    import imageio.v2 as imageio
    import tifffile
    from PIL import Image
    from ..figure_tables.cell_cards import analyse
    from ..measure.mask_geometry import crop_size
    from ..visualisation.cell_cards import render_frames
    from auto_organotypic.video.encode import write_video
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    folder = Path(measurement_dir)
    report = json.loads((folder/'manifest.json').read_text())
    paths = {row['role']:row for row in report['inputs']}
    for row in report['inputs'] + report['outputs']:
        if digest(row['path']) != row['sha256']:
            raise ValueError("Source fingerprint mismatch: " + row['path'])
    table = pd.read_csv(folder/'cell_frame.csv')
    raw_role='raw'
    if 'channel' in table.columns:
        available=list(table.channel.unique())
        if channel is None:
            if len(available)!=1:raise ValueError('Choose channel= for a multiple-channel comparison')
            channel=available[0]
        if channel not in available:raise ValueError('Requested channel is absent from the measurements')
        table=table[table.channel.eq(channel)].copy();raw_role=f'channel:{channel}'
    elif channel is not None:raise ValueError('This measurement has no named channels')
    selected = table[table.identity.eq(int(identity))].copy()
    if selected.empty:
        raise ValueError("identity is absent from measurement")
    out = Path(output_dir)
    name = f"cell_{int(identity)}_comparison"
    if prefix:
        import re
        if not re.fullmatch(r'[A-Za-z0-9_-]+',prefix):raise ValueError('Video prefix must be a safe recording name')
        name=f'{prefix}_{name}'
    if channel is not None:
        import re
        slug=re.sub(r'[^a-zA-Z0-9_-]','_',str(channel)).lower()
        name=f'{name}_{slug}'
    movie = out/(name+'.mp4')
    if movie.exists():
        raise FileExistsError(movie)
    ancestor=out.resolve()
    while not ancestor.exists(): ancestor=ancestor.parent
    required=int(frame_width_px*round(frame_width_px*.75)*3*len(selected))+256*1024**2
    if shutil.disk_usage(ancestor).free < required:
        raise OSError("Insufficient free space for movie and encoding buffer")
    def load(path):
        if _workspace is None:return tifffile.imread(path)
        with tifffile.TiffFile(path) as opened:array=opened.series[0].asarray(out=str(_workspace/f'{len(_maps)}.dat'))
        array.flush();_maps.append(array);return array
    raw=load(paths[raw_role]['path'])
    labels=load(paths['labels']['path'])
    expanded=load(folder/'expanded_labels.tif')
    settings=report['settings']; offset=settings['source_frame_offset']
    hours=selected.sort_values('source_frame_index').hours.to_numpy(float)
    if not np.array_equal(selected.sort_values('source_frame_index').source_frame_index,np.arange(len(raw))):
        raise ValueError('The saved trace must include every native frame, including absent masks')
    analysis=analyse(raw,labels,identities=[identity],source_frame_offset=offset,
                     frame_interval_h=settings['frame_interval_h'],frame_times_hours=hours,measurement_table=selected)
    analysis.update(expanded_labels=expanded,background_inner_px=settings['background_inner_px'],
                    background_outer_px=settings['background_outer_px'],table=selected)
    if prefix or channel:analysis['title_prefix']=' · '.join(str(v) for v in (prefix,channel) if v)
    if 'blocked' in paths:
        obstacle=load(paths['blocked']['path'])
        analysis['blocked']=obstacle.view(bool) if obstacle.dtype==np.uint8 and np.max(obstacle)<=1 else obstacle.astype(bool)
    cell=analysis['cells'][int(identity)]
    cell['table']=selected
    cell['crop_size_px']=crop_size(cell['observations'],margin=int(np.ceil(settings['margin_px']+settings['background_outer_px']))+2)
    # Re-read sums independently from the saved masks; compare every plotted point.
    checked=[]
    from ..measure.expanded_outlines import background_mask
    for row in selected.itertuples():
        f=int(row.source_frame_index); j=f-offset
        if not row.mask_present:
            assert np.isnan(row.integrated_density) and np.isnan(row.expanded_integrated_density)
            continue
        original=labels[j]==identity; grown=expanded[j]==identity
        assert np.all(expanded[j][labels[j]>0]==labels[j][labels[j]>0])
        assert np.isclose(np.sum(raw[f][original],dtype=float),row.integrated_density,rtol=1e-12,equal_nan=True)
        total=np.sum(raw[f][grown],dtype=float)
        assert np.isclose(total,row.expanded_integrated_density,rtol=1e-12,equal_nan=True)
        obstacle=analysis.get('blocked')
        ring=background_mask(expanded[j],identity,settings['background_inner_px'],settings['background_outer_px'],
                             obstacle[j] if obstacle is not None else None)
        # Float32 preserves native uint16 pixels and represents missing camera
        # support as NaN. Accumulate its background in float64, like measurement.
        values=np.asarray(raw[f][ring],dtype=float);values=values[np.isfinite(values)]
        if len(values)>=settings['min_background_px']:
            corrected=total-getattr(np,settings['background_statistic'])(values)*grown.sum()
            assert np.isclose(corrected,row.corrected_expanded_integrated_density,rtol=1e-12,atol=1e-7,equal_nan=True)
        else: assert np.isnan(row.corrected_expanded_integrated_density)
        checked.append(f)
    out.mkdir(parents=True,exist_ok=True)
    write_video(str(movie),render_frames(analysis,reporter=reporter,comparison=True,frame_width_px=frame_width_px),fps=fps)
    reader=imageio.get_reader(movie); count=0; shape=None
    previews=[]
    try:
        for frame in reader:
            if shape is None: shape=frame.shape
            assert shape==frame.shape
            if count in (0,len(raw)//2,len(raw)-1):
                path=out/f'preview_{name}_{count:03d}.png';Image.fromarray(frame).save(path);previews.append(str(path))
            count+=1
    finally: reader.close()
    assert count==len(raw)
    audit=dict(cell=int(identity),output_sha256=digest(movie),
               source_sha256=paths[raw_role]['sha256'],outline_sha256=digest(folder/'expanded_labels.tif'),
               source_frames=list(range(len(raw))),decoded_frames=count,all_outline_totals_match=True,
               checked_observed_frames=checked,frame_shape=shape,display_range=analysis['display_range'],
               raw_path=paths[raw_role]['path'],original_labels_path=paths['labels']['path'],channel=channel,
               original_labels_sha256=paths['labels']['sha256'],expanded_labels_path=str((folder/'expanded_labels.tif').resolve()),
               fps=fps,reporter=reporter,source_frame_offset=offset,frame_interval_h=settings['frame_interval_h'],
               source_hours=hours.tolist(),
               spatial_filter='none',resampling='nearest',previews=previews)
    (out/(name+'_audit.json')).write_text(json.dumps(audit,indent=2))
    return audit
