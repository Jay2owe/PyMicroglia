"""Compare episode-consistent fusion absorption events with the indexed-frame audit.

The frozen episode module is loaded from the packaged correction engine and its
_fast_absorption_events is run on real label movies (the correction sequence's
input and the final labels, whole or as frame windows) at the discovery settings
and at looser ones, then through the indexed frames. The event tables must be
identical, dtypes and row order included; the full _discover result is compared
too.
"""
import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

import snapshot_engines

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'development/cd68_full_red_green/rigid_analysis/A1/tracking/engine'
HISTORY = ENGINE / 'm22_accepted_history/pymicroglia_95_full_A1/mid/accepted_history'
NATIVE = ENGINE / 'm22_accepted_history/pymicroglia_95_full_A1/out/95_full_A1.tif'
FINAL = ENGINE / 'm23_completed_feedback/pymicroglia_95_full_A1/out/95_full_A1.tif'


def frames_equal(x, y):
    try:
        pd.testing.assert_frame_equal(x, y, check_exact=True)
        return True
    except AssertionError:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--adapters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=0, help='window length (0: whole movie)')
    a = parser.parse_args()
    snapshot_engines.unpack(a.snapshot, snapshot_engines.COMPLETED_FEEDBACK, a.scratch / 'engine')
    sys.path[:0] = [str(a.scratch / 'engine/code'), str(a.adapters.resolve())]
    episode = importlib.import_module('episode_consistent_fusion')
    import episode_fusion_events
    points = pd.read_csv(HISTORY / '26_latent_body_tracks/out/latent_track_points.csv', low_memory=False)
    loose = dict(episode.DISCOVERY)
    for key, value in list(loose.items()):
        if key == 'minimum_pre_event_frames':
            loose[key] = max(1, int(value) // 2)
        elif key == 'merge_reach_px':
            loose[key] = float(value) * 1.5
    failures, cases, events_total = [], {}, 0
    for movie_name, path in (('native', NATIVE), ('final', FINAL)):
        with tifffile.TiffFile(path) as handle:
            count = len(handle.pages)
            starts = [0] if not a.frames else [0, max(0, count // 2 - a.frames // 2), max(0, count - a.frames)]
            windows = [(s, count if not a.frames else s + a.frames) for s in starts]
            for first, last in windows:
                labels = handle.asarray(key=range(first, last))
                window_points = points[(points.frame.astype(int) >= first) & (points.frame.astype(int) < last)].copy()
                window_points['frame'] = window_points.frame.astype(int) - first
                for setting, params in (('discovery', episode.DISCOVERY), ('loose', loose)):
                    label = f'{movie_name}:{first}-{last}:{setting}'
                    t0 = time.perf_counter()
                    expected = episode._fast_absorption_events(labels, params)
                    t1 = time.perf_counter()
                    with episode_fusion_events.install() as stats:
                        assert stats['supported'] == 1, stats
                        actual = episode._fast_absorption_events(labels, params)
                    t2 = time.perf_counter()
                    events_total += len(expected)
                    cases[label] = dict(events=len(expected), indexed=stats['indexed'],
                                        original_seconds=round(t1 - t0, 2), indexed_seconds=round(t2 - t1, 2))
                    if not frames_equal(expected, actual) or stats['indexed'] != 1:
                        failures.append(label)
                label = f'{movie_name}:{first}-{last}:discover'
                expected = episode._discover(labels, window_points)
                with episode_fusion_events.install():
                    actual = episode._discover(labels, window_points)
                if len(expected) != len(actual) or not all(frames_equal(x, y) for x, y in zip(expected, actual)):
                    failures.append(label)
                cases[label] = dict(compared=len(expected))
    report = dict(status='pass' if not failures and events_total else 'fail', events=events_total,
                  cases=cases, failures=failures)
    a.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(report))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
