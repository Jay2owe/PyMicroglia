"""Compare mixed-owner flash owner runs from column arrays with the original, exactly.

python check_flash_owner_runs.py --engine <engine> --adapter <identity_candidate dir> --output <report.json>

Every track of A1's owner-attached latent points (as discover groups them, and
shuffled), then synthetic tracks: sorted, unsorted and repeated frames, float,
NaN, infinite and object frames, NaN owners, string visibility, nullable
integers, missing columns and empty tracks. Runs must be equal lists of equal
dicts (Python ints), and errors must match.
"""
import argparse
import json
from pathlib import Path
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
    sys.path[:0] = [str(a.adapter.resolve()), str((a.engine / 'code').resolve())]
    import bracketed_mixed_owner_flash as module
    import flash_owner_runs as adapter
    from common import load_stack
    original = module._owner_runs
    timing = [0.0, 0.0]
    compared = errors = 0

    def outcome(function, group):
        try:
            tick = time.perf_counter()
            value = function(group)
            elapsed = time.perf_counter() - tick
            assert all(type(v) is int for run in value for v in (run['owner'], run['first'], run['last'], *run['frames']))
            return 'ok', repr(value), elapsed
        except AssertionError:
            raise
        except Exception as error:
            return 'error', (type(error).__name__, str(error)), 0.0

    def compare(group):
        nonlocal compared, errors
        old = outcome(original, group)
        new = outcome(module._owner_runs, group)
        timing[0] += old[2]; timing[1] += new[2]
        compared += 1
        assert old[:2] == new[:2], (old[:2], new[:2])
        errors += old[0] == 'error'

    history = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
    points = pd.read_csv(history / '26_latent_body_tracks/out/latent_track_points.csv')
    labels = load_stack(history / '60_well_b2_issue004_terminal_two_seat_assimilation.tif')
    attached = module.owner_consensus.attach_owners(points, labels).sort_values(
        ['track_id', 'frame']).reset_index(drop=True)
    rng = np.random.default_rng(4)
    with adapter.install(module) as stats:
        assert stats['supported'], 'pinned sources differ'
        for _, group in attached.groupby('track_id', sort=True):
            compare(group.sort_values('frame'))
            compare(group.sample(frac=1, random_state=int(rng.integers(1 << 30))))
        real = compared
        base = attached[attached.track_id == attached.track_id.iloc[0]].sort_values('frame')
        for case in range(400):
            n = int(rng.integers(0, 30))
            frames = np.sort(rng.choice(60, n, replace=bool(case % 3 == 0)))
            table = pd.DataFrame(dict(frame=frames, track_id=1, candidate_owner=rng.integers(0, 4, n),
                                      physically_visible=rng.random(n) > .2, x=rng.random(n)))
            if case % 7 == 1:
                table = table.iloc[::-1]
            if case % 5 == 2:
                table['frame'] = table.frame.astype(float)
            compare(table)
        holes = base.copy(); holes['frame'] = holes.frame.astype(float); holes.iloc[3, holes.columns.get_loc('frame')] = np.nan
        compare(holes)
        last = base.copy(); last['frame'] = last.frame.astype(float); last.iloc[-1, last.columns.get_loc('frame')] = np.inf
        compare(last)
        compare(base.assign(frame=base.frame.astype(object)))
        compare(base.assign(candidate_owner=base.candidate_owner.astype(float).where(base.index != base.index[2])))
        compare(base.assign(physically_visible=base.physically_visible.astype(str)))
        compare(base.assign(candidate_owner=base.candidate_owner.astype('Int64')))
        compare(base.assign(frame=base.frame.astype('Int64')))
        compare(base.drop(columns=['frame']))
        compare(base.drop(columns=['candidate_owner']))
        compare(base.iloc[:0])
        compare(base.iloc[:1])
        compare(base.set_index(np.zeros(len(base), int)))
    assert module._owner_runs is original
    report = dict(status='pass', compared=compared, real_tracks=real, original_errors=errors,
                  original_seconds=round(timing[0], 2), adapter_seconds=round(timing[1], 2), counters=stats)
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(report))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
