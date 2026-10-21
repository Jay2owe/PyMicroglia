"""Compare indexed Issue 004 two-body fusion discovery with the frozen module, exactly.

Synthetic movies of drifting, vanishing and absorbed bodies (including bodies
on the frame edge) check the absorption events and identity histories against
the frozen helpers; the real stage-44 movie checks the whole discover result.
With --episode-code, the correction engine's episode-consistent fusion
discovery (which substitutes its own precomputed absorption events) is also
compared on the real movie.
"""
import argparse
import json
from pathlib import Path
import sys
import time


def synthetic(rng, frames, size, owners):
    import numpy as np
    yy, xx = np.mgrid[:size, :size]
    stack = np.zeros((frames, size, size), np.uint16)
    centre = rng.uniform(0, size, (owners, 2))
    radius = rng.uniform(1.5, 6, owners)
    alive = np.ones(owners, bool)
    for t in range(frames):
        centre += rng.normal(0, 1.5, centre.shape)
        centre = np.clip(centre, -2, size + 1)
        alive ^= rng.random(owners) < 0.12
        host = rng.integers(0, owners, owners)
        for k in rng.permutation(owners):
            painted = k + 1 if alive[k] else (host[k] + 1 if alive[host[k]] else 0)
            if not painted:
                continue
            disk = (yy - centre[k, 0]) ** 2 + (xx - centre[k, 1]) ** 2 <= radius[k] ** 2
            if rng.random() < 0.2:
                disk &= rng.random(disk.shape) < 0.7  # several components
            stack[t][disk] = painted
    return stack


def frames_equal(expected, actual):
    import pandas as pd
    pd.testing.assert_frame_equal(expected, actual, check_exact=True)
    assert list(expected.dtypes) == list(actual.dtypes)


def main():
    p = argparse.ArgumentParser()
    for name in ('engine', 'adapter', 'config', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--run', default='pymicroglia_95_full_A1')
    p.add_argument('--stack', default='44_well_issue001_isolated_pair_encounter_recovery.tif')
    p.add_argument('--movies', type=int, default=300)
    p.add_argument('--skip-real', action='store_true')
    p.add_argument('--episode-code', type=Path)
    a = p.parse_args()
    import numpy as np
    import pandas as pd
    import tifffile
    sys.path[:0] = [str(a.adapter.resolve()), str((a.engine / 'code').resolve())]
    import recurrent_two_body_fusion as module
    import two_body_fusion_index as index
    import common
    configured = common.Config.load(a.config).values["accepted_postprocessing"][
        "field_wide_recurrent_two_body_fusion_recovery"]
    params = dict(configured.get("parameters", {}), targeting_mode="field_wide_discovery")
    report = dict(status='fail', synthetic_movies=0, synthetic_events=0, synthetic_histories=0)

    rng = np.random.default_rng(4)
    loose = dict(params, merge_reach_px=3.0, minimum_absorbed_fraction=0.05,
                 minimum_host_growth_fraction=-5.0, maximum_host_growth_fraction=5.0,
                 minimum_pre_event_frames=1)
    for number in range(a.movies):
        stack = synthetic(rng, int(rng.integers(4, 25)), int(rng.choice([24, 48, 64])),
                          int(rng.integers(2, 12)))
        chosen = params if number % 3 == 0 else dict(
            loose, merge_reach_px=float(rng.choice([0.5, 3.0, 12.0, 1e9])),
            minimum_pre_event_frames=int(rng.integers(0, 4)))
        frames = index._Frames(module, stack)
        expected = module._absorption_events(stack, chosen)
        actual = frames.absorption_events(chosen)
        frames_equal(expected, actual)
        report['synthetic_events'] += len(expected)
        for owner in range(1, 14):
            e, f = module._history_stats(stack, owner), frames.history_stats(owner)
            assert e == f and all(type(e[k]) is type(f[k]) for k in e), (number, owner, e, f)
            report['synthetic_histories'] += 1
        report['synthetic_movies'] += 1
    print({k: report[k] for k in ('synthetic_movies', 'synthetic_events', 'synthetic_histories')}, flush=True)

    if not a.skip_real:
        accepted = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
        labels = tifffile.imread(accepted / a.stack)
        points = pd.read_csv(accepted / '26_latent_body_tracks/out/latent_track_points.csv')
        tick = time.perf_counter()
        expected = module.discover(labels, points, params)
        report['before_seconds'] = time.perf_counter() - tick
        with index.install(module) as stats:
            assert stats['supported']
            tick = time.perf_counter()
            actual = module.discover(labels, points, params)
            report['after_seconds'] = time.perf_counter() - tick
        for e, f in zip(expected, actual):
            frames_equal(e, f)
        report.update(real_events=len(expected[1]), real_pairs=len(expected[0]), counters=stats)
        if a.episode_code:
            sys.path.append(str(a.episode_code.resolve()))
            import episode_consistent_fusion as episode
            assert episode.fusion is module
            tick = time.perf_counter()
            expected = episode._discover(labels, points)
            report['episode_before_seconds'] = time.perf_counter() - tick
            with index.install(module) as stats:
                tick = time.perf_counter()
                actual = episode._discover(labels, points)
                report['episode_after_seconds'] = time.perf_counter() - tick
            assert stats.get('substituted_events') == 1
            for e, f in zip(expected, actual):
                frames_equal(e, f)
            report.update(episode_events=len(expected[1]), episode_pairs=len(expected[0]), episode_counters=stats)
    report['status'] = 'pass'
    print(report)
    a.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
