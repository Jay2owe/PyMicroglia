"""Run every correction whose per-track sort track_sort_skips replaces, original against adapted, exactly.

python tools/check_track_sort_corrections.py --engine <A1 engine> --code <engine code> --adapter <identity_candidate> --output <report.json>

On 40-frame windows of the accepted A1 movie (labels, registered raw, tracker
points), with no and with scattered protected pixels, and with the points as
recorded and shuffled (so tracks arrive out of frame order and take the
sorts): every module track_sort_skips rewrites from dormant name basins on,
through its correct() (short bookended somata through discover). Flanked contacts also run
with the reference-area memo wrapped around the sort-skipped correct(). Results
(arrays, tables, values, dtypes) or raised errors must be identical, the
adapted function must be the one that ran, and the originals are restored.
"""
import argparse
import importlib
import json
from pathlib import Path
import sys
import time



def same(left, right):
    import numpy as np
    import pandas as pd
    if type(left) is not type(right):
        return False
    if isinstance(left, pd.DataFrame):
        try:
            pd.testing.assert_frame_equal(left, right, check_exact=True)
        except AssertionError:
            return False
        return type(left.index) is type(right.index)
    if isinstance(left, pd.Series):
        try:
            pd.testing.assert_series_equal(left, right, check_exact=True)
        except AssertionError:
            return False
        return True
    if isinstance(left, np.ndarray):
        return left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(map(same, left, right))
    if isinstance(left, dict):
        return list(left) == list(right) and all(same(left[k], right[k]) for k in left)
    if isinstance(left, float) and left != left:
        return right != right
    return bool(left == right)

CALLS = {
    'recover_dormant_name_soma_basins': ('correct', lambda l, u, r, p, q: (l, r, p, q)),
    'recover_multistage_reciprocal_handoff': ('correct', lambda l, u, r, p, q: (l, p)),
    'complete_owner_handoff_episodes_r10_a002': ('correct', lambda l, u, r, p, q: (l, r, p, q)),
    'recover_short_bookended_soma_beside_transient_holders': ('discover', lambda l, u, r, p, q: (l, p)),
    'recover_reciprocal_retired_name_a002': ('correct', lambda l, u, r, p, q: (l, u, r, p, q)),
    'bookended_multi_owner_soma_a002': ('correct', lambda l, u, r, p, q: (l, r, p, q)),
    'coobserved_bookended_soma_memory_a001': ('correct', lambda l, u, r, p, q: (l, r, p, q)),
    'recover_flanked_large_contact_interruptions': ('correct', lambda l, u, r, p, q: (l, r, p, q)),
    'recover_emergent_reciprocal_late_handoff': ('correct', lambda l, u, r, p, q: (l, p)),
    'bridge_dormant_raw_soma_gaps': ('correct', lambda l, u, r, p, q: (l, r, p, q)),
    'recover_repeated_soma_renaming': ('correct', lambda l, u, r, p, q: (l, p, q)),
    'remove_optical_core_adjacent_pseudosomata': ('correct', lambda l, u, r, p, q: (l, r, p, q)),
    'restore_terminal_exclusive_soma_from_shared_name': ('correct', lambda l, u, r, p, q: (l, r, p, q)),
    'shared_parent_soma_ownership_a003': ('correct', lambda l, u, r, p, q: (l, r, p, q)),
    'recover_displaced_soma_from_dim_name_reuse': ('correct', lambda l, u, r, p, q: (l, r, p, q)),
    'restore_brief_name_in_raw_separable_contact': ('correct', lambda l, u, r, p, q: (l, r, p, u, q)),
    'recover_dim_displaced_name_paths': ('correct', lambda l, u, r, p, q: (l, r, p)),
    'overlapping_detector_soma_memory_a003': ('correct', lambda l, u, r, p, q: (l, r, p, q)),
}


def main():
    p = argparse.ArgumentParser()
    for name in ('engine', 'adapter', 'code', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--stem', default='95_full_A1')
    p.add_argument('--run', default='pymicroglia_95_full_A1')
    p.add_argument('--starts', default='0,200,436')
    p.add_argument('--frames', type=int, default=40)
    a = p.parse_args()
    import numpy as np
    import pandas as pd
    import tifffile
    folder = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
    raw_movie = tifffile.imread(folder / '07_trimmed_evidence/out' / (a.stem + '_registered_raw.tif'))
    label_movie = tifffile.imread(folder / '99_well_b3_issue013_anchored_projection_owner_relay.tif')
    all_points = pd.read_csv(folder / '26_latent_body_tracks/out/latent_track_points.csv', low_memory=False)
    sys.path[:0] = [str(a.code.resolve()), str(a.adapter.resolve())]
    modules = {name: importlib.import_module(name) for name in CALLS}
    import track_sort_skips as adapter
    import reference_area_boxes
    dual = importlib.import_module('recover_stable_dual_soma_contacts')
    flanked = modules['recover_flanked_large_contact_interruptions']
    specs = {}
    for spec in adapter.SPECS:
        specs.setdefault(spec['module'], []).append(spec['function'])
    rng = np.random.default_rng(23)
    report = dict(status='pass', calls=0, values=0, errors=0, stacked=0)
    timings = dict(original=0.0, adapted=0.0)
    install_stats, reference_stats = {}, dict(calls=0, labelled=0, reused=0, fallback=0)
    per_module = {name: 0 for name in CALLS}

    def run(function, args):
        try:
            return ('ok', function(*args))
        except Exception as error:
            return ('error', type(error).__name__, str(error))
    for start in (int(s) for s in a.starts.split(',')):
        start = max(0, min(start, len(label_movie) - a.frames))
        window = slice(start, start + a.frames)
        labels = np.ascontiguousarray(label_movie[window])
        raw = np.ascontiguousarray(raw_movie[window])
        points = all_points[(all_points.frame >= start) & (all_points.frame < start + a.frames)].copy()
        points['frame'] = points.frame - start
        ledger = labels.copy()
        for protected_name, protected in (('none', np.zeros(labels.shape, bool)),
                                          ('scattered', rng.random(labels.shape) < .002)):
            for order, table in (('recorded', points), ('shuffled', points.sample(frac=1, random_state=start))):
                for name, (entry, arguments) in CALLS.items():
                    module = modules[name]
                    args = arguments(labels, ledger, raw, table, protected)
                    targets = specs[name]
                    originals = [getattr(module, target) for target in targets]
                    tick = time.perf_counter()
                    expected = run(getattr(module, entry), args)
                    timings['original'] += time.perf_counter() - tick
                    stacks = [False] + ([True] if module is flanked else [])
                    for stacked in stacks:
                        with adapter.install(install_stats):
                            for target, original in zip(targets, originals):
                                assert getattr(module, target).__wrapped__ is original, name
                                assert getattr(module, target).__code__.co_filename == '<track-sort-skips>'
                            if stacked:
                                with reference_area_boxes.install(dual, reference_stats, caller=module):
                                    assert module.correct.__wrapped__.__wrapped__ is originals[0]
                                    calls = reference_stats['calls']
                                    actual = run(getattr(module, entry), args)
                                    assert reference_stats['calls'] == calls + 1
                                report['stacked'] += 1
                            else:
                                tick = time.perf_counter()
                                actual = run(getattr(module, entry), args)
                                timings['adapted'] += time.perf_counter() - tick
                        assert all(getattr(module, t) is o for t, o in zip(targets, originals))
                        assert expected[0] == actual[0], (name, start, order, expected[:3], actual[:3])
                        if expected[0] == 'ok':
                            assert same(list(expected[1]) if isinstance(expected[1], tuple) else expected[1],
                                        list(actual[1]) if isinstance(actual[1], tuple) else actual[1]), \
                                (name, start, protected_name, order)
                            report['values'] += 1
                        else:
                            assert expected == actual
                            report['errors'] += 1
                        report['calls'] += 1
                        per_module[name] += 1
        print(start, report, install_stats, flush=True)
    # A correct() wrapped by anything other than the sort skips is left alone by the memo.
    other = lambda *args: None
    other.__wrapped__ = flanked.correct
    saved = flanked.correct
    flanked.correct = other
    try:
        with reference_area_boxes.install(dual, reference_stats, caller=flanked):
            assert flanked.correct is other
    finally:
        flanked.correct = saved
    assert install_stats['in_order'] > 0 and install_stats['sorted'] > 0, install_stats
    assert install_stats['groups_in_order'] > 0 and install_stats['groups_sorted'] > 0, install_stats
    report.update(seconds=timings, stats=install_stats, reference=reference_stats, per_module=per_module)
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
