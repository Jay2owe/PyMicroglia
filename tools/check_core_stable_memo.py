"""Compare the overbright-source corrections with their shared stable-core measurement, exactly.

python tools/check_core_stable_memo.py --engine <A1 engine> --code <engine code> --adapter <identity_candidate> --output <report.json>

On windows of the accepted A1 movie (labels, registered raw, tracker points),
with no and with scattered protected pixels, the three corrections, their
_eligible_cores and the optical-core union run original and adapted, in the
order the run uses them, so the adapted calls reuse the held measurement.
Results (arrays, tables, dtypes) or raised errors must be identical. A raw
movie changed in one pixel, a float copy and a 2-D raw must be measured again
or run directly; callers writing into a returned array must not change a
later answer; the originals are restored afterwards.
"""
import argparse
import importlib
import json
from pathlib import Path
import sys


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
        return True
    if isinstance(left, np.ndarray):
        return left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(map(same, left, right))
    return bool(left == right)


def main():
    p = argparse.ArgumentParser()
    for name in ('engine', 'adapter', 'code', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--stem', default='95_full_A1')
    p.add_argument('--run', default='pymicroglia_95_full_A1')
    p.add_argument('--starts', default='0,200,436')
    p.add_argument('--frames', type=int, default=60)
    a = p.parse_args()
    import numpy as np
    import pandas as pd
    import tifffile
    folder = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
    raw_movie = tifffile.imread(folder / '07_trimmed_evidence/out' / (a.stem + '_registered_raw.tif'))
    label_movie = tifffile.imread(folder / '99_well_b3_issue013_anchored_projection_owner_relay.tif')
    all_points = pd.read_csv(folder / '26_latent_body_tracks/out/latent_track_points.csv', low_memory=False)
    sys.path[:0] = [str(a.code.resolve()), str(a.adapter.resolve())]
    import core_stable_memo as adapter
    names = [spec['module'] for spec in adapter.SPECS]
    modules = [importlib.import_module(name) for name in names]
    optical = importlib.import_module('remove_optical_core_adjacent_pseudosomata')
    held = [m._eligible_cores for m in modules]
    rng = np.random.default_rng(31)
    report = dict(status='pass', calls=0, values=0, errors=0)

    def run(function, *args):
        try:
            return ('ok', function(*args))
        except Exception as error:
            return ('error', type(error).__name__, str(error))

    def compare(expected, actual, label):
        assert expected[0] == actual[0], (label, expected[:3], actual[:3])
        if expected[0] == 'ok':
            assert same(expected[1], actual[1]), label
            report['values'] += 1
        else:
            assert expected == actual, label
            report['errors'] += 1
        report['calls'] += 1
    stats = {}
    for start in (int(s) for s in a.starts.split(',')):
        start = max(0, min(start, len(label_movie) - a.frames))
        window = slice(start, start + a.frames)
        labels = np.ascontiguousarray(label_movie[window])
        raw = np.ascontiguousarray(raw_movie[window])
        points = all_points[(all_points.frame >= start) & (all_points.frame < start + a.frames)].copy()
        points['frame'] = points.frame - start
        changed = raw.copy()
        changed[a.frames // 2, 7, 9] += 1
        for protected_name, protected in (('none', np.zeros(labels.shape, bool)),
                                          ('scattered', rng.random(labels.shape) < .002)):
            # The run's order: each correction, then the optical-core union.
            sequence = [(f'{name} correct', module.correct, (labels, raw, points, protected))
                        for name, module in zip(names, modules)]
            sequence += [('optical union', optical._optical_core_union, (raw,))]
            sequence += [(f'{name} changed raw', module._eligible_cores, (changed,))
                         for name, module in zip(names, modules)]
            sequence += [('float raw', modules[0]._eligible_cores, (raw.astype(np.float32),)),
                         ('one frame', modules[1]._eligible_cores, (raw[0],)),
                         ('repeat', modules[2]._eligible_cores, (raw,))]
            expected = [run(function, *args) for _, function, args in sequence]
            with adapter.install(stats):
                for module, original in zip(modules, held):
                    assert module._eligible_cores.__wrapped__ is original
                    assert module._eligible_cores.__code__.co_filename == '<core-stable-memo>'
                for index, ((label, _, args), wanted) in enumerate(zip(sequence, expected)):
                    function = sequence[index][1]
                    function = getattr(sys.modules[function.__module__], function.__name__)
                    actual = run(function, *args)
                    compare(wanted, actual, (start, protected_name, label))
                    # A caller writing into the answer must not reach a later call.
                    if actual[0] == 'ok' and label == 'repeat':
                        actual[1][0][:] = -1
                        again = run(function, *args)
                        compare(wanted, again, (start, protected_name, 'after write'))
            for module, original in zip(modules, held):
                assert module._eligible_cores is original and '_core_stable' not in vars(module)
        print(start, report, stats, flush=True)
    assert stats['versions'] > 0 and stats['reused'] > 0 and stats['measured'] > 0 and stats['direct'] > 0, stats
    report['stats'] = stats
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
