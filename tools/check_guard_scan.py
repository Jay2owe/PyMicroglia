"""Compare the block-wise composition guards with the original whole-movie expressions, exactly.

Real accepted label frames with synthetic edits: renamed pixels, gained and lost
foreground, new and vanished names, protected masks that do and do not meet a
change, ledgers that do and do not change, several dtypes, negative names and
mismatched shapes (fallback). Every helper must give the same value as its
original expression, for plain arrays and for disk-backed working arrays. Every
rewritten _guard function must raise the same error (or return the same value)
as the original on the same inputs.
"""
import argparse
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser()
    for name in ('engine', 'adapter', 'code', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--run', default='pymicroglia_95_full_A1')
    p.add_argument('--frames', type=int, default=120)
    p.add_argument('--temporary', type=Path, required=True)
    a = p.parse_args()
    import numpy as np
    import pandas as pd
    import tifffile
    sys.path[:0] = [str(a.adapter.resolve()), str(a.code.resolve())]
    import guard_scan as gs
    from benchmark_working_storage import load
    storage = load(a.adapter / 'array_storage.py')
    movie = tifffile.imread(a.engine / 'm22_accepted_history' / a.run
                            / 'mid/accepted_history/99_well_b3_issue013_anchored_projection_owner_relay.tif')[:a.frames]
    rng = np.random.default_rng(11)
    names = [int(v) for v in np.unique(movie)[1:]]

    def variants(base):
        """One edited copy at a time, so memory holds two movies."""
        def edit(change):
            x = base.copy(); change(x); return x
        yield 'identical', base.copy()
        yield 'renamed', edit(lambda x: x[10:14].__setitem__(x[10:14] == names[3], names[4]))
        yield 'gained', edit(lambda x: x.__setitem__((20, slice(0, 5), slice(0, 5)), names[2]))
        yield 'lost', edit(lambda x: x[30].__setitem__(x[30] == names[5], 0))
        yield 'new_name', edit(lambda x: x[40].__setitem__(x[40] == names[6], 200 if x.dtype == np.uint8 else 9000))
        yield 'vanished_name', edit(lambda x: x.__setitem__(x == names[7], names[8]))
        yield 'vanished_to_background', edit(lambda x: x.__setitem__(x == names[9], 0))
        yield 'last_pixel', edit(lambda x: x.__setitem__((-1, -1, -1), names[1] if x[-1, -1, -1] == 0 else 0))
        yield 'first_pixel_large', edit(lambda x: x.__setitem__((0, 0, 0), 255 if x.dtype == np.uint8 else 60000))

    report = dict(status='pass', frames=a.frames, helpers={}, guards={})
    pool = storage.Pool(a.temporary, minimum_bytes=1024**2, temporary_ram_bytes=32 * 1024**2)
    checked = 0
    seconds = dict(original=0.0, adapted=0.0)

    def compare(label, original, adapted):
        nonlocal checked
        t0 = time.perf_counter(); expected = original(); t1 = time.perf_counter(); actual = adapted(); t2 = time.perf_counter()
        seconds['original'] += t1 - t0; seconds['adapted'] += t2 - t1
        if isinstance(expected, set) or isinstance(actual, set):
            assert expected == actual, (label, expected, actual)
        elif isinstance(expected, (int, np.integer)) and not isinstance(expected, (bool, np.bool_)):
            assert int(expected) == int(actual) and isinstance(actual, int), (label, expected, actual)
        else:
            assert bool(expected) == bool(actual), (label, expected, actual)
        checked += 1

    with storage.install(pool):
        for dtype in (np.uint16, np.int32, np.int64, np.uint8):
            base = movie.astype(dtype) if dtype != np.uint8 else (movie % 200).astype(np.uint8)
            for disk in (False, True):
                def held(x):
                    if not disk:
                        return x
                    y = pool.allocate(x.shape, x.dtype, force_disk=True); y[...] = x
                    return y
                L = held(base)
                masks = {'none': np.zeros(base.shape, bool), 'foreground': base > 0,
                         'random': rng.random(base.shape) < .01, 'int': (base > 0).astype(np.uint8)}
                masks = {k: held(v) for k, v in masks.items()}
                for name, value in variants(base):
                    C = held(value)
                    tag = '%s/%s/%s' % (np.dtype(dtype).name, 'disk' if disk else 'ram', name)
                    for mname, m in masks.items():
                        compare(tag + '/where/' + mname, lambda: np.any((C != L) & m), lambda: gs.changed_where(C, L, m))
                        compare(tag + '/equal_where/' + mname, lambda: np.any((C != L) & (m == L)), lambda: gs.changed_where_equal(C, L, m, L))
                        compare(tag + '/union/' + mname, lambda: np.array_equal((C > 0) | (m > 0), (L > 0) | (m > 0)),
                                lambda: gs.same_union_foreground(C, m, L, m))
                    compare(tag + '/positive', lambda: np.any((C != L) & (L > 0)), lambda: gs.changed_where_positive(C, L, L))
                    compare(tag + '/nonzero', lambda: np.any((C != L) & (C != 0)), lambda: gs.changed_where_nonzero(C, L, C))
                    compare(tag + '/any', lambda: bool(np.any(C != L)), lambda: gs.any_changed(C, L))
                    compare(tag + '/gained', lambda: np.any((L == 0) & (C > 0)), lambda: gs.gained_foreground(L, C))
                    compare(tag + '/lost', lambda: np.any((L > 0) & (C == 0)), lambda: gs.lost_foreground(L, C))
                    compare(tag + '/foreground', lambda: np.array_equal(C > 0, L > 0), lambda: gs.same_foreground(C, L))
                    compare(tag + '/equal', lambda: np.array_equal(C, L), lambda: gs.equal(C, L))
                    def survivor_original():
                        survivor = C > 0
                        return np.any(survivor & (L == 0)) or np.any(C[survivor] != L[survivor])
                    compare(tag + '/survivor', survivor_original, lambda: gs.survivor_changed(C, L))
                    compare(tag + '/count', lambda: int((C != L).sum()), lambda: gs.count_changed(C, L))
                    compare(tag + '/count_nonzero', lambda: np.count_nonzero(C != L), lambda: gs.count_changed(C, L))
                    compare(tag + '/names', lambda: np.array_equal(np.unique(C), np.unique(L)), lambda: gs.same_names(C, L))
                    compare(tag + '/names_reversed', lambda: np.array_equal(np.unique(L), np.unique(C)), lambda: gs.same_names(L, C))
                    compare(tag + '/added', lambda: set(map(int, np.unique(C))) - set(map(int, np.unique(L))), lambda: gs.added_names(C, L))
                    compare(tag + '/added_reversed', lambda: set(map(int, np.unique(L))) - set(map(int, np.unique(C))), lambda: gs.added_names(L, C))
                    compare(tag + '/setdiff', lambda: len(np.setdiff1d(np.unique(C), np.unique(L))), lambda: len(gs.added_names(C, L)))
                    del C
                del L, masks
        # Fallback inputs: negative names, mismatched shapes, float names, lists.
        L = movie[:20].astype(np.int32); C = L.copy(); C[3, 0, 0] = -5
        fallback_before = gs.stats['fallback']
        compare('negative/names', lambda: np.array_equal(np.unique(C), np.unique(L)), lambda: gs.same_names(C, L))
        compare('negative/added', lambda: set(map(int, np.unique(C))) - set(map(int, np.unique(L))), lambda: gs.added_names(C, L))
        S = L[:1]
        compare('broadcast/where', lambda: np.any((C != S) & (S > 0)), lambda: gs.changed_where_positive(C, S, S))
        compare('broadcast/foreground', lambda: np.array_equal(C > 0, S > 0), lambda: gs.same_foreground(C, S))
        compare('broadcast/count', lambda: int((C != S).sum()), lambda: gs.count_changed(C, S))
        F = L.astype(float); F[2, 0, 0] = np.nan
        compare('float/equal', lambda: np.array_equal(F, F), lambda: gs.equal(F, F))
        compare('float/names', lambda: np.array_equal(np.unique(F), np.unique(L)), lambda: gs.same_names(F, L))
        assert gs.stats['fallback'] - fallback_before >= 5, gs.stats
    report['helpers'] = dict(comparisons=checked, seconds=seconds, stats=dict(gs.stats))

    # Whole guard functions: original against rewritten, same raise or result.
    base = movie.astype(np.int32)
    guards = 0
    for path in sorted(a.code.glob('*_composition*.py')):
        name = gs.VERSIONS.get(hashlib.sha256(path.read_bytes()).hexdigest()[:16])
        if name is None:
            continue
        module = importlib.import_module(name)
        found = gs.adapt(module)
        assert found, name
        module._guard_scan = gs
        for function, (original, rewritten) in found.items():
            if not function.startswith('_guard'):
                continue
            parameters = list(inspect.signature(original).parameters)
            for case, value in variants(base):
                changed = int(np.count_nonzero(value != base))
                for protected in (np.zeros(base.shape, bool), base > 0):
                    for audited in (changed, changed + 1):
                        for flag in (True, False):
                            arguments = []
                            for parameter in parameters:
                                arguments.append({
                                    'labels': base, 'parent': base, 'candidate': value, 'result': value,
                                    'protected': protected, 'audited': audited, 'foreground': flag, 'identities': flag,
                                    'applications': pd.DataFrame(dict(changed_pixels=[audited])) if flag else pd.DataFrame(),
                                }[parameter])
                            outcome = []
                            for call in (original, rewritten):
                                try:
                                    outcome.append(('value', call(*arguments)))
                                except Exception as error:
                                    outcome.append((type(error).__name__, str(error)))
                            assert outcome[0] == outcome[1], (name, function, case, outcome)
                            guards += 1
            report['guards'][name + '.' + function] = 'pass'
        del module._guard_scan
    report['guard_calls'] = guards

    # Sequence compositions: whole apply generators with a stand-in stage method
    # returning prepared proposals; every yield must be identical.
    from contextlib import contextmanager
    frames = min(40, len(movie))
    labels0 = movie[:frames].astype(np.int32)
    saved = labels0.copy(); saved[2:6][saved[2:6] == names[3]] = 0
    restored = labels0.copy(); restored[8:12][restored[8:12] == names[4]] = names[5]
    ledger0 = np.where((labels0 == 0) & (rng.random(labels0.shape) < .001), names[6], 0).astype(np.int32)
    raw = rng.random(labels0.shape).astype(np.float32)
    proposals = []
    for k in range(6):
        after = saved.copy()
        after[k:k + 9][after[k:k + 9] == names[k + 1]] = names[k + 2]
        after[k + 3, 10:30, 10:30] = names[k + 7]
        after[k + 5][after[k + 5] == names[k + 8]] = 0
        proposals.append(after)
    scopes = ['other', 'unrevised_original', 'rule_validated_complete_paths']
    sequence_checks = applied = 0
    for disk in (False, True):
        def held(x):
            if not disk:
                return x.copy()
            with storage.install(pool):
                y = pool.allocate(x.shape, x.dtype, force_disk=True); y[...] = x
            return y
        for name in ('recovery_sequence_composition', 'current_path_sequence_composition', 'contact_sequence_composition'):
            module = importlib.import_module(name)
            found = gs.adapt(module)
            assert found and 'apply' in found, name
            original, rewritten = found['apply']
            module._guard_scan = gs
            operations = [dict(stage='s%d' % k, uses_history=False, use_current_labels=False, use_original_labels=False,
                               projection_scope=scopes[k % 3]) for k in range(6)]
            outputs = []
            for function in (original, rewritten):
                calls = iter(range(6))

                @contextmanager
                def stand_in(spec):
                    index = next(calls)
                    yield lambda *args, **kwargs: (proposals[index].copy(), pd.DataFrame(dict(stage=[spec['stage']])))
                held_method = module.frozen_method
                module.frozen_method = stand_in
                try:
                    with storage.install(pool):
                        out = [(stage, np.asarray(labels).copy(), np.asarray(ledger).copy(), tables, extra)
                               for stage, labels, ledger, tables, extra in function(
                                   held(labels0), held(ledger0), held(raw), pd.DataFrame(dict(frame=[0])), held(restored),
                                   held(labels0), held(saved), {}, pd.DataFrame(), operations)]
                finally:
                    module.frozen_method = held_method
                outputs.append(out)
            for (s0, l0, d0, t0, e0), (s1, l1, d1, t1, e1) in zip(*outputs):
                assert s0 == s1 and l0.dtype == l1.dtype and np.array_equal(l0, l1) and d0.dtype == d1.dtype and np.array_equal(d0, d1), (name, s0)
                assert set(t0) == set(t1), name
                for key in t0:
                    pd.testing.assert_frame_equal(t0[key], t1[key], check_exact=True)
                sequence_checks += 1
                applied += len(t0['birth_applications'])
            assert len(outputs[0]) == len(outputs[1]) == 6
            del module._guard_scan
    assert applied > 0
    report['sequence_yields'] = sequence_checks
    report['sequence_applications'] = applied
    report['settle'] = {key: gs.stats.get(key, 0) for key in ('settle_calls', 'settle_fallback')}
    assert report['settle']['settle_calls'] > 0 and report['settle']['settle_fallback'] == 0, report['settle']
    pool.close()
    a.output.write_text(json.dumps(report, indent=1, default=str) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=report['status'], comparisons=checked, guard_calls=guards, seconds=seconds, stats=gs.stats)))


if __name__ == '__main__':
    main()
