"""Expose exact native measurement masks, channels and comparison movies in the workbench."""
from pathlib import Path
import json


def write_workspace(recordings, output, *, preferred_metric):
    import pandas as pd
    from auto_organotypic.native_channel_registration import digest
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    movies = []
    for recording in recordings:
        source = Path(recording['measurement_dir'])
        table = pd.read_csv(source/'cell_frame.csv')
        for channel, raw in recording['channels'].items():
            stem = Path(recording['folder']).name+'_'+channel.lower()
            rows = table[table.channel.eq(channel) & table.frame_index.ge(0)].copy()
            rows['stem'] = stem
            rows['signal_mean'] = rows.integrated_density/rows.area_px
            target = output/'measure'/stem/'cell_frame.csv'
            target.parent.mkdir(parents=True, exist_ok=True)
            rows.to_csv(target, index=False)
            masks = source/'expanded_labels.tif'
            movies.append(dict(stem=stem, subject=Path(recording['folder']).name,
                spec=dict(raw=str(Path(raw).resolve()), labels=str(masks.resolve()),
                          source_frame_offset=recording['offset'],
                          sha256={'raw':digest(raw), 'labels':digest(masks)}),
                tables={'cell_frame':{'path':str(target.relative_to(output)), 'sha256':digest(target)}},
                display=dict(base_labels=recording['labels'], blocked=recording['blocked'],
                    outline_settings=recording['outline_options'], preferred_metric=preferred_metric,
                    comparison_metrics=['integrated_density', 'expanded_integrated_density', 'corrected_expanded_integrated_density'],
                    measurement_labels=str(masks.resolve()), original_channel=channel),
                summary={'frames':recording['frames']-recording['offset']}))
    manifest = dict(movies=movies, run={'folder':str(output)}, settings={'measurement_space':'original camera pixels'},
                    measurement_option='expanded outlines with exclusive ownership and local-background comparison')
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
    return output


def render_comparisons(recordings, output, *, identities=(), options=None):
    from ..video.expanded_outlines import cell_video
    import tempfile
    allowed = {'fps', 'frame_width_px'}
    options = dict(options or {})
    if set(options)-allowed:
        raise ValueError('Comparison video options are fps and frame_width_px')
    reports = []
    for recording in recordings:
        chosen = tuple(dict.fromkeys(map(int, identities))) or recording['report']['identities']
        if set(chosen)-set(recording['report']['identities']):
            raise ValueError('Requested video identities must exist in each selected recording')
        for identity in chosen:
            for channel in recording['channels']:
                report = cell_video(recording['measurement_dir'], identity=identity,
                    channel=channel, reporter='cd68' if channel=='Red' else 'green', output_dir=output,
                    prefix=Path(recording['folder']).name, workspace=tempfile.gettempdir(), **options)
                reports.append(report)
    return reports
