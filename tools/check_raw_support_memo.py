"""Compare raw_supported_points with its remembered per-point decisions, exactly.

Each frozen recovery_qualified_soma version runs on the recording's registered
raw movie and tracker points (first frames only), whole and as a registration
crop with shifted points, for every flag combination the version accepts. One
memo is shared across all of them, as across correction stages. The movie is
then changed at some point centres and every call is repeated: changed patches
must be measured again, never reused. The unchanged movie is then run again
(whole calls answered from memory), and with one point moved (never reused).
Kept rows (index, values, order) must equal the original's.

With --labels, tracker_support_frames also runs with ownership labels (the
accepted label movie, whole and cropped, and a copy with some frames relabelled
in the changed round) for every ownership option of each version; frame masks
must equal the original's.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

VERSIONS = {'recovery/r185_a002': [{}],
            'recovery/r190_a002': [{}, dict(centre_connected=True)],
            'current/r195_a002': [{}, dict(centre_connected=True), dict(internal_peak=True),
                                  dict(internal_peak=True, peak_quantisation=True),
                                  dict(centre_connected=True, internal_peak=True, peak_quantisation=True)]}


FRAMES = {'recovery/r191_a003': [{}],
          'current/r195_a002': [{}, dict(internal_peak=True), dict(internal_peak=True, peak_quantisation=True)],
          'current/r197_a002': [{}, dict(internal_peak=True, peak_quantisation=True)],
          'current/r200_a004': [{}, dict(internal_peak=True, retain_named_centre=True)],
          'current/r201_a002': [{}, dict(internal_peak=True, peak_quantisation=True, prefer_centred_owner=True)]}
OWNED = {'current/r197_a002': [{}, dict(internal_peak=True, peak_quantisation=True)],
         'current/r200_a004': [dict(require_foreground_majority=True),
                               dict(internal_peak=True, retain_named_centre=True, require_foreground_majority=True)],
         'current/r201_a002': [dict(retain_named_centre=True, prefer_centred_owner=True),
                               dict(internal_peak=True, peak_quantisation=True, prefer_centred_owner=True,
                                    require_foreground_majority=True)]}
STAGE_MODULES = ('spatial_soma_paths', 'recovery_temporal_soma_support', 'recovery_qualified_soma',
                 'recovery_destination_soma', 'tracker_core_ownership')


def load(directory, name='recovery_qualified_soma'):
    for stage_module in STAGE_MODULES:
        sys.modules.pop(stage_module, None)
    sys.path.insert(0, str(directory))
    try:
        return importlib.import_module(name)
    finally:
        sys.path.remove(str(directory))


def same_frames(left, right):
    return (list(left) == list(right) and all(
        type(left[t]) is type(right[t]) and left[t].dtype == right[t].dtype
        and left[t].shape == right[t].shape and np.array_equal(left[t], right[t]) for t in left))


def main():
    p = argparse.ArgumentParser()
    for name in ('engine', 'adapter', 'frozen', 'config', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--stem', default='95_full_A1')
    p.add_argument('--run', default='pymicroglia_95_full_A1')
    p.add_argument('--frames', type=int, default=240)
    p.add_argument('--labels', type=Path, help='label movie for the ownership-label cases')
    a = p.parse_args()
    global np
    import numpy as np
    import pandas as pd
    import tifffile
    sys.path[:0] = [str((a.engine / 'code').resolve())]
    import common
    import pipeline
    sys.path.insert(0, str(a.adapter.resolve()))
    import raw_support_memo as adapter
    cfg = common.Config.load(a.config)
    raw = tifffile.imread(pipeline._pinned_paths(cfg, a.stem)['registered_raw'])[:a.frames]
    points = pd.read_csv(a.engine / 'm22_accepted_history' / a.run /
                         'mid/accepted_history/26_latent_body_tracks/out/latent_track_points.csv', low_memory=False)
    points = points[points.frame < a.frames].reset_index(drop=True)
    points.index = points.index * 3 + 7  # a non-default index must be preserved
    crop = (slice(None), slice(9, raw.shape[1] - 6), slice(4, raw.shape[2] - 11))
    shifted = points.copy()
    shifted['y'] -= 9
    shifted['x'] -= 4
    changed = raw.copy()
    rng = np.random.default_rng(23)
    strong = points[points.strong.astype(bool) & points.state.eq('observed')]
    for _, q in strong.iloc[rng.choice(len(strong), min(400, len(strong)), replace=False)].iterrows():
        t, y, x = int(q.frame), int(np.floor(q.y + .5)), int(np.floor(q.x + .5))
        if 0 <= y < raw.shape[1] and 0 <= x < raw.shape[2]:
            changed[t, y, x] = changed[t, y, x] + 1 if changed[t, y, x] < np.iinfo(changed.dtype).max else 0
    owned = changed_owned = None
    if a.labels is not None:
        labels = tifffile.imread(a.labels)[:a.frames]
        owned = np.zeros(raw.shape, labels.dtype)
        f, h, w = (min(u, v) for u, v in zip(raw.shape, labels.shape))
        owned[:f, :h, :w] = labels[:f, :h, :w]
        changed_owned = owned.copy()
        names = np.unique(owned[owned > 0])
        for t in range(0, len(owned), 7):
            if len(names) > 1:
                swap = owned[t] == names[t % len(names)]
                changed_owned[t][swap] = names[(t + 1) % len(names)]
    memo = adapter.Memo()
    report = dict(cases=[])
    timing = dict(original=0.0, adapted=0.0)
    moved = points.copy()
    moved.loc[strong.index[len(strong) // 2], 'x'] += 0.5
    for movie_name, movie in (('raw', raw), ('changed', changed), ('raw-again', raw), ('moved-point', raw)):
        for relative, combos in VERSIONS.items():
            module = load(a.frozen / relative)
            digest = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            assert digest in adapter.VERSIONS, relative
            for view, table in ((('whole', moved),) if movie_name == 'moved-point' else (('whole', points), ('crop', shifted))):
                stack = movie if view == 'whole' else movie[crop]
                for kwargs in combos:
                    tick = time.perf_counter()
                    expected = module.raw_supported_points(stack, table, **kwargs)
                    middle = time.perf_counter()
                    with adapter.install(memo) as stats:
                        assert module.raw_supported_points is not None and module.raw_supported_points.__wrapped__
                        actual = module.raw_supported_points(stack, table, **kwargs)
                    timing['original'] += middle - tick
                    timing['adapted'] += time.perf_counter() - middle
                    assert getattr(module.raw_supported_points, '__wrapped__', None) is None
                    assert expected.index.equals(actual.index) and expected.equals(actual), (movie_name, relative, view, kwargs)
                    report['cases'].append(dict(movie=movie_name, version=relative, view=view, flags=kwargs,
                                                kept=len(actual), rejected=len(table) - len(actual)))
        for relative, combos in FRAMES.items():
            module = load(a.frozen / relative, 'spatial_soma_paths')
            for view, table in ((('whole', moved),) if movie_name == 'moved-point' else (('whole', points), ('crop', shifted))):
                stack = movie if view == 'whole' else movie[crop]
                for kwargs in combos:
                    tick = time.perf_counter()
                    expected = module.tracker_support_frames(stack, table, **kwargs)
                    middle = time.perf_counter()
                    with adapter.install(memo):
                        assert module.tracker_support_frames.__wrapped__
                        actual = module.tracker_support_frames(stack, table, **kwargs)
                    timing['original'] += middle - tick
                    timing['adapted'] += time.perf_counter() - middle
                    assert getattr(module.tracker_support_frames, '__wrapped__', None) is None
                    assert same_frames(expected, actual), (movie_name, relative, view, kwargs)
                    report['cases'].append(dict(movie=movie_name, frames_version=relative, view=view, flags=kwargs,
                                                frames=len(actual)))
        for relative, combos in (OWNED.items() if owned is not None else ()):
            module = load(a.frozen / relative, 'spatial_soma_paths')
            owners = changed_owned if movie_name == 'changed' else owned
            for view, table in ((('whole', moved),) if movie_name == 'moved-point' else (('whole', points), ('crop', shifted))):
                stack = movie if view == 'whole' else movie[crop]
                labels_view = owners if view == 'whole' else owners[crop]
                for kwargs in combos:
                    tick = time.perf_counter()
                    expected = module.tracker_support_frames(stack, table, ownership_labels=labels_view, **kwargs)
                    middle = time.perf_counter()
                    with adapter.install(memo):
                        assert module.tracker_support_frames.__wrapped__
                        actual = module.tracker_support_frames(stack, table, ownership_labels=labels_view, **kwargs)
                    timing['original'] += middle - tick
                    timing['adapted'] += time.perf_counter() - middle
                    assert getattr(module.tracker_support_frames, '__wrapped__', None) is None
                    assert same_frames(expected, actual), ('owned', movie_name, relative, view, kwargs)
                    report['cases'].append(dict(movie=movie_name, owned_version=relative, view=view, flags=kwargs,
                                                frames=len(actual)))
    report.update(status='pass', points=len(points), frames=a.frames, seconds=timing, stats=dict(memo.stats),
                  retained_bytes=memo.bytes)
    assert memo.stats['reused'] > 0 and memo.stats['fallback'] == 0 and memo.stats['calls_reused'] > 0
    assert memo.stats['frames_reused'] > 0 and memo.stats['frames_fallback'] == 0
    assert memo.stats['cores_reused'] > 0 and memo.stats['cores_measured'] > 0
    if owned is not None:
        assert memo.stats['frames_ownership'] > 0 and memo.stats['owners_reused'] > 0
    report['sources'] = {str(f): hashlib.sha256(Path(f).read_bytes()).hexdigest()
                         for f in (Path(__file__), Path(adapter.__file__))}
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'cases'}))


if __name__ == '__main__':
    main()
