"""Compare Issue 012 ownership reconcile with its dormancy adapters, exactly.

python tools/check_dormancy_prefilter.py --engine <engine folder> --adapter <identity_candidate> --labels <labels.tif> --output <report.json>

reconcile runs with the original functions and with ownership_region_rows
(observations, source metrics, dormancy and overlap source prefilters) and
ownership_observation_reuse (apply_dormancy) installed in the history's order,
on small synthetic merge movies and on real frames in which touching cells are
merged for a few frames and then separated again, so dormancy starts, blocked
bids and split releases occur, with the recorded thresholds and with looser
and stricter overlap and merge thresholds. Labels, unclaimed frames and both
tables must be identical.
"""
import argparse
import json
from pathlib import Path
import platform
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--engine', type=Path, required=True)
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    platform.platform = lambda *a, **k: 'Windows-dormancy-prefilter-checks'
    sys.path[:0] = [str(a.engine.resolve() / 'code'), str(a.adapter.resolve())]
    import time
    from contextlib import ExitStack
    import numpy as np
    import pandas as pd
    import tifffile
    from scipy import ndimage as ndi
    import accepted_history as ah
    import ownership_observation_reuse
    import ownership_region_rows
    module = ah._issue_module(ah.DOMAIN_OWNERSHIP_OPS, 'reconcile_domain_ownership')
    params = ah._params('issue012', 'domain_ownership')
    cfg = params['ownership']
    seconds = dict(before=0., after=0.)
    totals = {}

    def run(labels, unclaimed, lag, methods, protection=None, cfg=cfg):
        tick = time.perf_counter()
        old = module.reconcile(labels, unclaimed, lag, methods, cfg, 0, protection)
        seconds['before'] += time.perf_counter() - tick
        originals = (module.observations, module.source_metrics, module.detect_new_dormancy, module.apply_dormancy,
                     module.overlap_proposals)
        with ExitStack() as stack:
            rows = stack.enter_context(ownership_region_rows.install(module))
            reuse = stack.enter_context(ownership_observation_reuse.install(module))
            assert rows['supported'] and 'apply_dormancy' in reuse['supported']
            assert '_dormancy_sources' in vars(module) and '_overlap_sources' in vars(module)
            tick = time.perf_counter()
            new = module.reconcile(labels, unclaimed, lag, methods, cfg, 0, protection)
            seconds['after'] += time.perf_counter() - tick
        assert (module.observations, module.source_metrics, module.detect_new_dormancy, module.apply_dormancy,
                module.overlap_proposals) == originals
        assert '_dormancy_sources' not in vars(module) and '_overlap_sources' not in vars(module)
        for key in ('dormancy_items', 'dormancy_measured', 'dormancy_fallback',
                    'overlap_items', 'overlap_skipped', 'overlap_fallback'):
            totals[key] = totals.get(key, 0) + rows[key]
        assert np.array_equal(old[0], new[0]) and old[0].dtype == new[0].dtype
        assert np.array_equal(old[1], new[1]) and old[1].dtype == new[1].dtype
        for left, right in zip(old[2:], new[2:]):
            pd.testing.assert_frame_equal(left, right, check_exact=True)
            assert left.to_csv(index=False) == right.to_csv(index=False)
        actions = new[2]
        return {} if actions.empty else actions['action'].value_counts().to_dict()

    cases = []
    rng = np.random.default_rng(8521)
    for case in range(12):
        labels = np.zeros((9, 41, 47), dtype=np.uint16)
        for i in range(1, 9):
            y, x = rng.integers(0, 33), rng.integers(0, 39)
            labels[:, y:y + 7, x:x + 7] = i
        labels[3:6, labels[0] == 1] = 2
        lag = rng.normal(0, 3, (8, 41, 47)).astype('float32')
        cases.append(dict(case=f'synthetic {case}', actions=run(labels, np.zeros_like(labels), lag, {'O', 'R', 'M', 'U'})))

    movie = tifffile.imread(a.labels)
    for start in (0, 40):
        labels = np.array(movie[start:start + 30])
        # Touching names at frame 8: the second takes the first's name for
        # frames 8-13, then separates again.
        frame = labels[8]
        pairs = []
        used = set()
        for name in [int(v) for v in np.unique(frame) if v]:
            if len(pairs) >= 12 or name in used:
                continue
            ring = ndi.binary_dilation(frame == name, np.ones((3, 3), bool)) & (frame != name) & (frame > 0)
            others = [int(v) for v in np.unique(frame[ring]) if int(v) not in used]
            if others:
                pairs.append((name, others[0]))
                used.update((name, others[0]))
        for host, merged in pairs:
            for t in range(8, 14):
                labels[t][labels[t] == merged] = host
        lag = rng.normal(0, 2, (len(labels) - 1,) + labels.shape[1:]).astype('float32')
        unclaimed = np.zeros_like(labels)
        cases.append(dict(case=f'real frames {start}-{start + 29}, {len(pairs)} merged pairs',
                          actions=run(labels, unclaimed, lag, set(params['methods']))))
        for changes in (dict(minimum_overlap_px=0), dict(minimum_overlap_px=1, merge_source_fraction=0.1),
                        dict(minimum_overlap_px=40, merge_source_fraction=0.9)):
            varied = {**cfg, **changes}
            cases.append(dict(case=f'real frames {start}-{start + 29}, thresholds {changes}',
                              actions=run(labels, unclaimed, lag, set(params['methods']), cfg=varied)))

    seen = {}
    for item in cases:
        for key, value in item['actions'].items():
            seen[key] = seen.get(key, 0) + value
    passed = (totals['dormancy_items'] > 0 and totals['dormancy_measured'] > 0
              and totals['overlap_items'] > 0 and totals['overlap_skipped'] > 0
              and seen.get('dormancy_started', 0) > 0
              and seen.get('external_bid_blocked', 0) + seen.get('supported_split_release', 0) > 0)
    report = dict(status='pass' if passed else 'failed', cases=cases, counters=totals, actions=seen,
                  seconds={k: round(v, 3) for k, v in seconds.items()})
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(status=report['status'], counters=totals, actions=seen, seconds=report['seconds'])))
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()
