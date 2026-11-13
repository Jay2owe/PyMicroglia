"""Check guard_scan's lent method inputs against the compositions' copies, exactly.

python tools/check_lent_inputs.py --adapter <identity_candidate> --engine <completed_feedback_engine folder, or its zip> --output <report.json>

Synthetic frozen methods (reading, writing, failing, returning an input,
keeping keyword order) run once as the compositions wrote them (copies) and
once through lent_call; values, types, dtypes, raised errors and the caller's
arrays must match, a writing method must be repeated in a fresh import, and
inputs that cannot be lent (lists, object or non-contiguous arrays) must be
copied. The three pinned compositions are then loaded from the engine
and their rewritten apply functions are run against the originals on small
movies with stand-in frozen methods, including file-backed (memmap) movies.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
import tempfile

import snapshot_engines


def same(left, right):
    import numpy as np
    import pandas as pd
    if isinstance(left, pd.DataFrame) or isinstance(right, pd.DataFrame):
        if type(left) is not type(right):
            return False
        try:
            pd.testing.assert_frame_equal(left, right, check_exact=True)
        except AssertionError:
            return False
        return True
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (type(left) is type(right) and left.dtype == right.dtype and left.shape == right.shape
                and np.array_equal(left, right))
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(map(same, left, right))
    if isinstance(left, dict):
        return type(left) is type(right) and list(left) == list(right) and all(same(left[k], right[k]) for k in left)
    return type(left) is type(right) and bool(left == right)


def run(function, *args, **kwargs):
    try:
        return ('ok', function(*args, **kwargs))
    except Exception as error:
        return ('error', type(error).__name__, str(error))


def main():
    p = argparse.ArgumentParser()
    for name in ('adapter', 'engine', 'output'):
        p.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    a = p.parse_args()
    import numpy as np
    import pandas as pd
    sys.path.insert(0, str(a.adapter.resolve()))
    import guard_scan
    stats = guard_scan.stats
    report = dict(status='pass', calls=0, values=0, errors=0)
    temporary = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)

    def compare(expected, actual, label):
        assert expected[0] == actual[0], (label, expected[:3], actual[:3])
        if expected[0] == 'ok':
            assert same(expected[1], actual[1]), label
            report['values'] += 1
        else:
            assert expected == actual, (label, expected, actual)
            report['errors'] += 1
        report['calls'] += 1

    imports = []

    def frozen_for(method):
        @contextmanager
        def frozen(spec):
            imports.append(spec['stage'])
            yield method
        return frozen

    sys.path.insert(0, str(a.adapter.resolve()))
    from array_storage import Pool
    pool = Pool(Path(temporary.name) / 'pool', minimum_bytes=1024, temporary_ram_bytes=32 * 1024**2)
    rng = np.random.default_rng(5)
    movie = rng.integers(0, 40, (6, 32, 48), dtype=np.uint16)
    raw = rng.random((6, 32, 48), dtype=np.float32)
    store = np.memmap(Path(temporary.name) / 'movie.memmap', dtype=movie.dtype, mode='w+', shape=movie.shape)
    store[:] = movie
    working = pool.array(movie, copy=True)
    working_raw = pool.array(raw, copy=True)
    assert type(working).__name__ == 'WorkingArray' and type(working.copy()) is type(working)

    def reads(observed, saved, raw, points, params, **kw):
        total = observed.astype(np.int64) + saved
        out = np.where(raw > params['cut'], total, 0)
        return out, pd.DataFrame(dict(n=[int(out.sum())], keys=[','.join(kw)], kind=[type(observed).__name__]))

    def writes(observed, saved, raw, points, params, **kw):
        observed[observed > 30] = 0
        saved += 1
        return reads(observed, saved, raw, points, params, **kw)

    def writes_keyword(observed, saved, raw, points, params, current_labels=None, **kw):
        current_labels[0] = 7
        return current_labels.copy(), pd.DataFrame(dict(n=[int(current_labels.sum())]))

    def fails(observed, saved, raw, points, params, **kw):
        raise ValueError('bad specification %d' % params['cut'])

    def returns_input(observed, saved, raw, points, params, **kw):
        return observed, saved[1:3], pd.DataFrame()

    def mutates_points(observed, saved, raw, points, params, **kw):
        points.loc[0, 'x'] = -1
        params['cut'] = 0
        observed[0] = 0
        return reads(observed, saved, raw, points, params, **kw)

    points = pd.DataFrame(dict(x=[1.0, 2.0], y=[3.0, 4.0]))
    history = pd.DataFrame(dict(track=[1, 2]))
    params = dict(cut=0.5)
    methods = dict(reads=reads, writes=writes, writes_keyword=writes_keyword, fails=fails,
                   returns_input=returns_input, mutates_points=mutates_points)
    inputs = dict(plain=(movie, movie[::-1].copy(), raw), memmap=(store, store, raw),
                  working=(working, working, working_raw), frames=(working[0], movie[0], raw[0]),
                  noncontiguous=(movie[:, ::2], movie[:, ::2].copy(), raw[:, ::2]),
                  objects=(movie.astype(object), movie, raw))
    for method_name, method in methods.items():
        for input_name, (observed, saved, rawish) in inputs.items():
            for lent_names in ((), ('current_labels',), ('current_labels', 'original_labels')):
                keep = [x.copy() for x in (observed, saved, rawish)]
                spec = dict(stage=method_name + '/' + input_name)
                lent = {name: (saved if name == 'current_labels' else observed) for name in lent_names}
                if method_name == 'writes_keyword' and 'current_labels' not in lent:
                    continue

                def original():
                    kwargs = {'proposal_history': history.copy(deep=True)}
                    for name, value in lent.items():
                        kwargs[name] = value.copy()
                    return method(observed.copy(), saved.copy(), rawish.copy(),
                                  points.copy(deep=True), deepcopy(params), **kwargs)
                expected = run(original)
                before = dict(stats)
                imports.clear()
                actual = run(guard_scan.lent_call, spec, frozen_for(method), method, (observed, saved, rawish), lent,
                             lambda: ((points.copy(deep=True), deepcopy(params)),
                                      {'proposal_history': history.copy(deep=True)}))
                compare(expected, actual, spec['stage'] + str(lent_names))
                for kept, now in zip(keep, (observed, saved, rawish)):
                    assert np.array_equal(kept, now) and kept.dtype == now.dtype, ('caller array changed', spec)
                assert points.loc[0, 'x'] == 1.0 and params['cut'] == 0.5
                lendable = input_name in ('plain', 'working')
                retried = stats.get('lent_retried', 0) - before.get('lent_retried', 0)
                if lendable and method_name in ('writes', 'mutates_points', 'writes_keyword'):
                    assert retried == 1 and imports == [spec['stage']], (spec, retried, imports)
                else:
                    assert retried == 0 and not imports, (spec, retried, imports)
                if not lendable:
                    assert stats.get('lent_copied', 0) > before.get('lent_copied', 0), spec
                if actual[0] == 'ok':
                    for value in actual[1] if isinstance(actual[1], tuple) else (actual[1],):
                        if isinstance(value, np.ndarray):
                            assert value.flags.writeable and not any(
                                np.may_share_memory(value, x) for x in (observed, saved, rawish)), spec
    view = guard_scan.lend(working)
    assert type(view) is type(working) and not view.flags.writeable and np.array_equal(view, working)
    assert guard_scan.lend(store).flags.writeable and guard_scan.lend(working[0]).flags.writeable
    assert guard_scan.lend(movie[:, ::2]).flags.writeable

    # The pinned compositions: original apply against the rewritten one.
    code = Path(temporary.name) / 'code'
    code.mkdir()
    names = ('recovery_sequence_composition', 'current_path_sequence_composition', 'reviewed_ownership_composition')
    for name in names:
        (code / (name + '.py')).write_bytes(snapshot_engines.shipped_engines.read(a.engine, 'code/' + name + '.py'))
    sys.path.insert(0, str(code))
    modules = [importlib.import_module(name) for name in names]
    originals = [m.apply for m in modules]

    def stand_in(method):
        @contextmanager
        def frozen(spec):
            imports.append(spec['stage'])
            yield method
        return frozen

    def sequence_specs(method_names, current):
        specs = []
        for index, method_name in enumerate(method_names):
            spec = dict(stage='s%d_%s' % (index, method_name), uses_history=index % 2 == 0)
            if current:
                spec.update(use_current_labels=index % 3 != 1, use_original_labels=index % 2 == 1,
                            projection_scope='unrevised_original' if index % 2 else 'other')
            specs.append(spec)
        return specs

    def sequence_method(observed, saved, raw, points, params, **kw):
        changed = np.where(raw > params['cut'], (observed.astype(np.int64) + 1) % 40, observed).astype(observed.dtype)
        return changed, pd.DataFrame(dict(keys=[','.join(kw)], n=[int(changed.sum())]))

    def sequence_writer(observed, saved, raw, points, params, **kw):
        observed[:, 0] = 3
        return sequence_method(observed, saved, raw, points, params, **kw)

    def ownership(observed, saved, raw, points, params, proposal_history=None, current_labels=None, original_labels=None):
        candidate = current_labels.copy()
        foreground = candidate > 0
        candidate[foreground] = np.where(raw[foreground] > params['cut'], original_labels[foreground],
                                         candidate[foreground])
        candidate[foreground & (candidate == 0)] = 1
        return candidate, pd.DataFrame(dict(n=[int(candidate.sum())], h=[len(proposal_history)]))

    def ownership_identity(observed, saved, raw, points, params, proposal_history=None, current_labels=None,
                           original_labels=None):
        return current_labels, pd.DataFrame()

    labels = rng.integers(1, 40, movie.shape, dtype=np.uint16)
    labels[:, :4] = 0
    ledger = np.zeros_like(labels)
    stores = {}
    for key, value in dict(labels=labels, saved=labels, restored=labels, observed=movie).items():
        path = Path(temporary.name) / (key + '.memmap')
        stores[key] = pool.array(value, copy=True)
    with guard_scan.install() as counts:
        assert counts['versions'] == 3, counts
        for module, original in zip(modules, originals):
            assert module.apply.__wrapped__ is original
            for movies in ('arrays', 'working'):
                L, S, R, O = ((labels, labels.copy(), labels.copy(), movie) if movies == 'arrays'
                              else (stores['labels'], stores['saved'], stores['restored'], stores['observed']))
                for writer in (False, True):
                    if module.__name__ == 'reviewed_ownership_composition':
                        chosen = [ownership, ownership_identity, ownership]
                        specs = [dict(stage='o%d' % i, kind='ownership') for i in range(3)]
                        args = (L, ledger, raw, points, R, O, S, params, history, None, None, specs)
                    else:
                        chosen = [sequence_method, sequence_writer if writer else sequence_method, sequence_method]
                        specs = sequence_specs([m.__name__ for m in chosen], 'current' in module.__name__)
                        args = (L, ledger, raw, points, R, O, S, params, history, specs)
                    by_stage = {spec['stage']: method for spec, method in zip(specs, chosen)}

                    @contextmanager
                    def frozen(spec):
                        imports.append(spec['stage'])
                        yield by_stage[spec['stage']]
                    keep = [x.copy() for x in (L, S, R, O)]
                    module.frozen_method = frozen
                    imports.clear()
                    expected = run(lambda: list(original(*args)))
                    imports.clear()
                    actual = run(lambda: list(module.apply(*args)))
                    compare(expected, actual, (module.__name__, movies, writer))
                    for kept, now in zip(keep, (L, S, R, O)):
                        assert np.array_equal(kept, now), (module.__name__, movies, 'caller array changed')
                    assert len(imports) == len(specs) + (writer and module.__name__ != 'reviewed_ownership_composition'), imports
    for module, original in zip(modules, originals):
        assert module.apply is original and '_guard_scan' not in vars(module)
    del store, stores, working, working_raw, view
    pool.close()
    report['stats'] = {k: v for k, v in stats.items() if k.startswith('lent')}
    assert report['stats'].get('lent_retried') and report['stats'].get('lent_returned') and report['stats'].get('lent_copied')
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
