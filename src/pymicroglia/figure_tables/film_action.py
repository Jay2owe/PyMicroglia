"""Select saved tracks, prepare display pixels and record streamed cell films."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from .saved import Tables
from .follow import DEFAULTS, Film, _plan, _region, _presentation, _window, _captions


def plans(cell_frame, events, labels, identities, *, span, event_hours, interval_min, options):
    if span not in {'recording', 'lifespan', 'event'}:
        raise ValueError('span must be recording, lifespan or event')
    if not np.isfinite(event_hours) or event_hours < 0:
        raise ValueError('event_hours must be finite and nonnegative')
    if not np.isfinite(interval_min) or interval_min <= 0:
        raise ValueError('A positive recorded minutes_per_frame is required')
    output = []
    for identity in identities:
        frames, centres, named, window = _plan(cell_frame, identity, labels,
            {**options, 'span': 'recording' if span == 'event' else span})
        clips = [(None, None, frames[0], frames[-1] + 1)]
        if span == 'event':
            half = int(round(event_hours * 60 / interval_min))
            mine = events.loc[events.identity.eq(identity)].sort_values('frame_index') if not events.empty else events
            clips = [(r.event, int(r.frame_index), max(0, int(r.frame_index)-half),
                      min(len(labels), int(r.frame_index)+half+1)) for r in mine.itertuples()]
        for event, event_frame, first, stop in clips:
            select = np.array([(first <= f < stop) for f in frames])
            chosen = np.array(frames)[select].tolist()
            if not chosen:
                continue
            name = str(identity) + (f'_{event}_{event_frame}' if event is not None else '')
            output.append(dict(identity=int(identity), name=name, event=event,
                event_frame=event_frame, frames=chosen, centres=centres[select], named=named[select],
                window_px=window, hours=np.array(chosen)*interval_min/60))
    return output


def _stacks(tables):
    import tifffile
    movie = next((m for m in tables.manifest.get('movies', []) if m['stem']==tables.stem), None)
    if movie is None:
        raise ValueError('Choose stem to identify one recorded movie')
    provenance = movie.get('provenance', {})
    records = provenance.get('inputs', {})
    if 'labels' not in records:
        raise ValueError('The run records no accepted label stack')
    stacks = {}
    for name in ('labels', 'raw'):
        record = records.get(name)
        if not record:
            continue
        path = Path(record['path'])
        if not path.is_absolute():
            path = tables.run/path
        digest = None
        if record.get('sha256'):
            with path.open('rb') as handle:
                digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        if digest is not None and digest != record['sha256']:
            raise ValueError(f'{name} stack differs from the recorded input: {path}')
        stacks[name] = tifffile.imread(path)
        tables.sources.append(path)
    if stacks['labels'].ndim != 3:
        raise ValueError('Label stack must have frame, row and column axes')
    interval = provenance.get('scale', {}).get('minutes_per_frame', movie.get('summary', {}).get('minutes_per_frame'))
    return stacks['labels'], stacks.get('raw'), interval


def prepared_frames(film, display, labels, options):
    from skimage.segmentation import find_boundaries
    captions = _captions(film.events, film.identity, options['event_hold_frames'])
    top, _, left, _ = film.region
    for position, frame in enumerate(film.frames):
        centre = film.centres[position] - (top, left)
        values = _window(labels[frame], centre, film.window_px).astype(int)
        picture = _window(display[frame], centre, film.window_px) if display is not None else (values > 0).astype(float)
        own = find_boundaries(values == film.identity, mode='outer')
        other = find_boundaries((values > 0) & (values != film.identity), mode='outer') if options['neighbours'] else np.zeros_like(own)
        caption = captions.get(frame, '')
        if not caption and not film.named[position]:
            caption = 'name not on screen'
        yield dict(picture=picture, own=own, other=other, centre=film.centres[position],
                   heading=f'Cell {film.identity} · {film.hours[position]:.1f} h · frame {frame}',
                   caption=caption, missing=not film.named[position])


def follow(run, *, stem=None, identities=(), events=(), cell_count=6, span='recording',
           event_hours=6.0, fps=6, crop_px=None, locator=True, cell_lut=None,
           display_options=None, output_dir=None, dry_run=False, keep_frames=False):
    from ..measure.modules.lifecycle import EVENTS
    wanted = tuple(events.split(',')) if isinstance(events,str) else tuple(events)
    unknown = set(wanted)-set(EVENTS)
    if unknown:
        raise ValueError(f'Unknown lifecycle events: {sorted(unknown)}; choose from {EVENTS}')
    options = {**DEFAULTS, **(display_options or {}), 'span':span, 'fps':fps,
               'crop_px':crop_px or 0, 'locator':locator, 'cell_lut':cell_lut or DEFAULTS['cell_lut']}
    unknown = set(options)-set(DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown display options: {sorted(unknown)}')
    if not np.isfinite(fps) or fps <= 0 or int(cell_count) < 1 or (crop_px is not None and crop_px < 1):
        raise ValueError('fps, cell_count and an explicit crop_px must be positive')
    tables = Tables(run, stem)
    frame = tables.table('cell_frame')
    all_events = tables.table('lifecycle_events', optional=True)
    all_events = pd.DataFrame() if all_events is None else all_events
    selected = all_events.loc[all_events.event.isin(wanted)] if wanted and not all_events.empty else all_events
    if identities:
        chosen = list(dict.fromkeys(map(int, identities)))
    elif not selected.empty:
        eligible = selected if wanted else selected.loc[~selected.event_status.eq('censored')]
        chosen = eligible.sort_values(['event_status','identity']).identity.drop_duplicates().astype(int).tolist()[:cell_count]
    elif wanted or span == 'event':
        chosen = []
    else:
        chosen = frame.groupby('identity').frame_index.count().sort_values(ascending=False).index[:cell_count].tolist()
    labels, raw, interval = _stacks(tables)
    if interval is None and len(frame):
        clock = frame[['frame_index','hours']].drop_duplicates().sort_values('frame_index')
        steps = clock.frame_index.diff()
        interval = (clock.hours.diff()/steps).dropna().median()*60
    clips = plans(frame, selected, labels, chosen, span=span, event_hours=event_hours,
                  interval_min=float(interval), options=options)
    target = Path(output_dir) if output_dir else tables.run/'films'/tables.stem
    report = [{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in clip.items()}
              | {'path':str(target/(clip['name']+'.mp4')),
                 'captions':_captions(all_events,clip['identity'],options['event_hold_frames'])} for clip in clips]
    if dry_run:
        return {'clips':report, 'dry_run':True}
    from ..visualisation.follow import render_frames
    from auto_organotypic.video.encode import write_video
    from .. import store, __version__
    # Validate every destination before encoding the first clip.
    for clip in report:
        if Path(clip['path']).exists():
            raise FileExistsError(clip['path'])
    source = store.collection(tables.sources)
    field = labels.max(axis=0)>0 if locator else None
    cache = {}
    for clip, entry in zip(clips, report):
        identity = clip['identity']
        if identity not in cache:
            full = plans(frame,all_events,labels,[identity],span='recording',event_hours=event_hours,
                         interval_min=float(interval),options=options)[0]
            region = _region(full['centres'],full['window_px'],labels.shape[1:])
            display, settings = _presentation(raw,labels,frame,options,region)
            cache[identity] = region,display,settings
        region, display, settings = cache[identity]
        film = Film(identity,tables.stem,clip['frames'],clip['hours'],clip['named'],clip['window_px'],
                    clip['centres'],all_events,region,settings)
        cropped = labels[:,region[0]:region[1],region[2]:region[3]]
        frames = render_frames(prepared_frames(film,display,cropped,options),field_view=field,
                               window_px=film.window_px,options=options)
        if keep_frames:
            frames = _keep(frames,target/(clip['name']+'_frames'))
        output = write_video(entry['path'],frames,fps=fps)
        store.claim('figure',source,{**options,'event_hours':event_hours,'stem':tables.stem},
                    path=output,method_version=__version__,display_only=True,
                    extra={'frame_plan':entry,'display':settings,'claim':
                           'Accepted identities and recorded events followed without reinterpretation.'})
    return {'clips':report,'dry_run':False}


def _keep(frames, folder):
    import imageio.v3 as imageio
    folder.mkdir(parents=True,exist_ok=False)
    for number,pixels in enumerate(frames):
        imageio.imwrite(folder/f'frame_{number:04d}.png',pixels)
        yield pixels
