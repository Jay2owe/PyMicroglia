"""Compare Issue 011 median summaries from column arrays with the original, exactly.

python check_merge_split_medians.py --engine <engine> --adapter <identity_candidate dir> --output <report.json>

The saved A1 frame-feature table (as read, and with integer areas, float32
intensities, NaN values, an object column and a missing column), asked for
every identity's pre, post and gap windows of every saved merge-split event,
random windows, absent identities and frames that take the frame filter, and
infinite values, which leave the bottleneck route: each
summary must be None in both or hold the same keys with bit-equal floats, and
errors must match.
"""
import argparse
import json
from pathlib import Path
import struct
import sys
import time


def main():
    p = argparse.ArgumentParser()
    for name in ('engine', 'adapter', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--run', default='pymicroglia_95_full_A1')
    a = p.parse_args()
    import numpy as np
    import pandas as pd
    sys.path[:0] = [str(a.adapter.resolve()), str((a.engine / 'code/corrections/merge_splits').resolve()),
                    str((a.engine / 'code').resolve())]
    import reconcile_merge_splits as module
    import merge_split_feature_index as adapter
    original = module.median_summary
    original_position = module.predicted_position
    folder = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history/15_merge_split_profile/out'
    features = pd.read_csv(folder / 'frame_features.csv')
    events = pd.read_csv(folder / 'merge_split_events.csv')
    timing = [0.0, 0.0, 0.0, 0.0]
    positions = 0
    compared = errors = nones = 0

    def bits(value):
        if isinstance(value, np.ndarray):
            return (value.dtype.str, value.shape, value.tobytes())
        return value if value is None else {k: struct.pack('<d', v) for k, v in value.items()}

    def outcome(function, *args):
        try:
            tick = time.perf_counter()
            value = function(*args)
            return 'ok', bits(value), time.perf_counter() - tick
        except Exception as error:
            return 'error', (type(error).__name__, str(error)), 0.0

    def questions(table):
        rng = np.random.default_rng(3)
        identities = sorted(set(int(v) for v in table.identity.dropna()))
        last = int(table.frame_index.max()) + 1
        for event in events.itertuples(index=False):
            pre, post = int(event.pre_frame_index), int(event.post_frame_index)
            people = [int(event.identity)] + [int(v) for v in str(event.candidate_identities).split('|')
                                                if v and v != 'nan']
            windows = [list(range(max(0, pre - w + 1), pre + 1)) for w in (3, 5, 8)]
            windows += [list(range(post, min(last, post + w))) for w in (3, 5, 8)]
            if not pd.isna(getattr(event, 'gap_start_frame_index', np.nan)):
                windows.append(list(range(int(event.gap_start_frame_index), int(event.gap_end_frame_index) + 1)))
            for person in people:
                for window in windows:
                    yield person, window
        for _ in range(3000):
            start = int(rng.integers(-3, last))
            yield int(rng.choice(identities + [10 ** 7])), list(range(start, start + int(rng.integers(0, 12))))
        yield identities[0], [1.0, 2.0]
        yield identities[0], np.arange(5)
        yield np.int64(identities[0]), [0, 1, 2]
        yield identities[0], [2 ** 70]

    def run(table, label):
        nonlocal compared, errors, nones, positions
        with adapter.install(module) as stats:
            assert stats['supported'] and stats['direct_medians']
            for identity, frames in questions(table):
                old = outcome(original, table, identity, frames)
                new = outcome(module.median_summary, table, identity, frames)
                timing[0] += old[2]; timing[1] += new[2]
                compared += 1
                assert old[:2] == new[:2], (label, identity, frames, old[:2], new[:2])
                errors += old[0] == 'error'
                nones += old[1] is None
                target = float(frames[-1]) + 1.5 if len(frames) and not isinstance(frames[-1], (np.ndarray,)) else 2.0
                old = outcome(original_position, table, identity, frames, target)
                new = outcome(module.predicted_position, table, identity, frames, target)
                timing[2] += old[2]; timing[3] += new[2]
                positions += 1
                assert old[:2] == new[:2], (label, 'position', identity, frames, old[:2], new[:2])
        assert module.median_summary is original and module.predicted_position is original_position
        return dict(stats)

    variants = {'saved': features}
    changed = features.copy(); changed['area'] = changed['area'].round().astype(np.int64)
    changed['mean_intensity'] = changed['mean_intensity'].astype(np.float32); variants['typed'] = changed
    holes = features.copy(); rng = np.random.default_rng(9)
    for column in ('centroid_x', 'solidity', 'area'):
        holes.loc[rng.random(len(holes)) < .1, column] = np.nan
    variants['nan'] = holes
    odd = features.copy(); odd['eccentricity'] = odd['eccentricity'].astype(object); variants['object'] = odd
    variants['missing'] = features.drop(columns=['normalized_perimeter'])
    variants['shuffled'] = features.sample(frac=1, random_state=2)
    variants['repeated'] = pd.concat([features, features.sample(300, random_state=3)], ignore_index=True)
    narrow = features.copy(); narrow['centroid_x'] = narrow['centroid_x'].astype(np.float32); variants['float32'] = narrow
    variants['labelled'] = features.assign(note='x')
    floated = features.copy(); floated['frame_index'] = floated['frame_index'].astype(float); variants['float_frames'] = floated
    infinite = features.copy(); rng = np.random.default_rng(11)
    for column in ('centroid_y', 'mean_intensity'):
        infinite.loc[rng.random(len(infinite)) < .05, column] = np.inf
    infinite.loc[rng.random(len(infinite)) < .05, 'solidity'] = -np.inf
    variants['infinite'] = infinite
    counters = {label: run(table, label) for label, table in variants.items()}
    report = dict(status='pass', compared=compared, original_errors=errors, none_results=nones,
                  original_seconds=round(timing[0], 2), adapter_seconds=round(timing[1], 2),
                  positions=positions, original_position_seconds=round(timing[2], 2),
                  adapter_position_seconds=round(timing[3], 2), counters=counters)
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(report))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
