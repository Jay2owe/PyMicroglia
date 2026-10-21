"""Deliver full-camera outline movies for the existing recording batch."""
from pathlib import Path
import argparse
import json
from pymicroglia.video.full_recording import render


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('well')
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--all-videos', action='store_true')  # legacy queue spelling
    args = parser.parse_args()
    home = args.home.resolve()
    complete = json.loads((home/'rigid_analysis'/args.well/'measurement_complete.json').read_text())
    handoff = json.loads((home/'rigid_batch/.j'/args.well/'motion_inputs.json').read_text())
    prepared = handoff['prepared']['95_full_'+args.well]
    crop_manifest = Path(prepared['frame_clock']).parent/'manifest.json'
    tracking = json.loads((home/'rigid_analysis'/args.well/'tracking_action.json').read_text())['result']
    folder = home/'review/full_recordings'
    def progress(done, total):
        (folder/(args.well+'_progress.json')).write_text(json.dumps(dict(rendered_frames=done, frames=total)))
    result = render(complete['measurement_dir'], folder, prefix=args.well, progress=progress,
                    registered_manifest=crop_manifest, tracking_labels=tracking['labels'],
                    tracking_offset=tracking['source_frame_offset'])
    marker = home/'review'/f'{args.well}_delivery.json'
    if marker.is_file():
        previous = json.loads(marker.read_text())
        if previous.get('delivery_mode') != 'full_recording':
            old = marker.with_name(args.well+'_individual_cell_delivery_history.json')
            if not old.exists(): old.write_text(marker.read_text())
    marker.write_text(json.dumps(dict(recording=args.well, delivery_mode='full_recording',
        all_videos_complete=True, movies=1, measured_cells=len(complete['identities']), receipt=str(folder/f'{args.well}_full_recording.json')), indent=2))
    records = [json.loads(p.read_text()) for p in folder.glob('*_full_recording.json')]
    # Existing queue consumes these summary keys at its final handoff.
    summary = dict(recordings=[r['recording'] for r in records], physical_cells=0,
        movie_count=len(records), intensity_trace_count=0, grid_count=0, delivery_mode='full_recording')
    (home/'review/figures/der_comparison_workspace.json').write_text(json.dumps(summary, indent=2))
    blocks = []
    for r in records:
        movie = next(n for n in r['files'] if n.endswith('.mp4'))
        links = ' | '.join(f'<a href="{n}">{n}</a>' for n in r['files'] if n.endswith('.tif'))
        blocks.append(f'<h2>{r["recording"]}</h2><video controls preload="none" style="width:100%" src="{movie}"></video><p>{links}</p>')
    (folder/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>Full recordings with outlines</title>'+''.join(blocks))
    (home/'review/figures/review.html').write_text('<!doctype html><meta http-equiv="refresh" content="0;url=../full_recordings/review.html"><a href="../full_recordings/review.html">Full recordings with outlines</a>')
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
