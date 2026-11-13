"""Compare boxed component coordinates with np.nonzero and np.where, exactly, and check the rewrites install.

python tools/check_sparse_nonzero.py --engine <A1 engine> --code <engine code> --adapter <identity_candidate> --output <report.json>

Every owner's whole-frame mask (labels == owner) and every foreground mask of
sampled frames of the accepted A1 movie, the same masks as strided views, and
random masks of every density (empty and full included) go through the helper
with np.nonzero and with np.where as the original call: the coordinate arrays
must have the same values, dtype and count. Masks the helper does not box
(integer, 3-D, small) must come from the original call itself. Then all
rewritten functions must install from their verified sources, run the helper
in place of the call, and be restored with the helper removed.
"""
import argparse
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
    p.add_argument('--frames', default='0,60,120,200,300,436')
    a = p.parse_args()
    import numpy as np
    import tifffile
    folder = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
    movie = tifffile.imread(folder / '99_well_b3_issue013_anchored_projection_owner_relay.tif')
    frozen = a.code.parent / 'frozen'
    sys.path[:0] = [str(a.code.resolve()), str(a.adapter.resolve())] + [
        str((frozen / part).resolve()) for part in ('recovery/r180_a002', 'recovery/r181_a001',
                                                    'recovery/r178_a020', 'recovery/r179_a002')]
    import sparse_nonzero as adapter
    stats = dict(boxed=0, whole=0)
    helper = adapter.sparse_nonzero(stats)
    report = dict(status='pass', masks=0, real_masks=0)

    def same(expected, actual):
        return (type(expected) is type(actual) is tuple and len(expected) == len(actual)
                and all(x.dtype == y.dtype and x.shape == y.shape and np.array_equal(x, y)
                        for x, y in zip(expected, actual)))

    def compare(mask):
        for original in (np.nonzero, np.where):
            assert same(original(mask), helper(mask, original)), (mask.shape, mask.dtype, int(mask.sum()))
            report['masks'] += 1
    tick = time.perf_counter()
    for t in (int(s) for s in a.frames.split(',')):
        frame = np.ascontiguousarray(movie[min(t, len(movie) - 1)])
        for mask in [frame > 0] + [frame == owner for owner in np.unique(frame[frame > 0])]:
            compare(mask)
            compare(mask[1::2, ::3])
            report['real_masks'] += 1
    rng = np.random.default_rng(41)
    for trial in range(4000):
        h, w = (int(v) for v in rng.integers(1, 300, 2))
        mask = rng.random((h, w)) < float(rng.choice([0, 1e-4, 2e-3, .05, .5, 1]))
        compare(mask if trial % 5 else np.asfortranarray(mask))
    # Not boxed: the original call's own result is returned.
    calls = []
    for value in (np.ones((80, 80), np.int32), np.zeros((4, 80, 80), bool), np.ones((10, 10), bool),
                  np.zeros((90, 90), bool)):
        whole = stats['whole']
        sentinel = object()
        assert helper(value, lambda m: (calls.append(m), sentinel)[1]) is sentinel
        assert stats['whole'] == whole + 1
    assert stats['boxed'] > 0 and len(calls) == 4, stats
    report['helper_seconds'] = round(time.perf_counter() - tick, 1)
    # Install: every rewritten function from its verified source, restored afterwards.
    modules = [importlib.import_module(spec['module']) for spec in adapter.SPECS]
    originals = [getattr(m, s['function']) for m, s in zip(modules, adapter.SPECS)]
    install_stats = {}
    with adapter.install(install_stats):
        for module, spec, original in zip(modules, adapter.SPECS, originals):
            current = getattr(module, spec['function'])
            assert current.__wrapped__ is original and current.__code__.co_filename == '<sparse-nonzero>', spec
            source = inspect.getsource(original)
            for old, new in spec['edits']:
                assert source.count(old) == 1 and '_sparse_nonzero' in new
            assert '_sparse_nonzero' in current.__code__.co_names or any(
                '_sparse_nonzero' in getattr(c, 'co_names', ()) for c in current.__code__.co_consts), spec
    assert install_stats['versions'] == len(adapter.SPECS), install_stats
    for module, spec, original in zip(modules, adapter.SPECS, originals):
        assert getattr(module, spec['function']) is original and '_sparse_nonzero' not in vars(module)
    report.update(versions=install_stats['versions'], helper=stats)
    # The domain ownership module, loaded through a stand-in accepted history.
    folder = (a.engine / 'code/corrections/domain_ownership').resolve()
    sys.path.insert(0, str(folder))
    issue = importlib.import_module('reconcile_domain_ownership')
    sys.path.remove(str(folder))
    held = dict(centroid=issue.centroid, unclaimed=issue.unclaimed_observations, points=issue.identity_points)

    class History:
        def _issue_module(self, *args, **kwargs):
            return issue
    history = History()
    loader_stats = {}
    observations = 0
    identity_frames = 0

    def fields(row):
        return [getattr(row, name) for name in getattr(row, '_fields', None) or vars(row)]
    with adapter.install_loader(history, loader_stats):
        assert history._issue_module() is issue
        assert issue.centroid.__wrapped__ is held['centroid']
        assert issue.unclaimed_observations.__wrapped__ is held['unclaimed']
        assert issue.identity_points.__wrapped__ is held['points']
        for t in (int(s) for s in a.frames.split(',')):
            frame = np.ascontiguousarray(movie[min(t, len(movie) - 1)])
            # identity_points: label frames of several integer types, a memmap-like
            # subclass view, an empty frame, and frames that keep np.nonzero.
            for variant in (frame, frame.astype(np.int32), frame.astype(np.int64), frame.view(np.memmap),
                            np.zeros_like(frame), frame.astype(float), frame[None], frame.tolist()):
                try:
                    expected = ('ok', held['points'](variant))
                except Exception as error:
                    expected = ('error', type(error).__name__, str(error))
                try:
                    actual = ('ok', issue.identity_points(variant))
                except Exception as error:
                    actual = ('error', type(error).__name__, str(error))
                assert expected[0] == actual[0] and (expected[0] == 'error' and expected == actual
                    or list(expected[1]) == list(actual[1]) and all(
                        expected[1][k].dtype == actual[1][k].dtype and np.array_equal(expected[1][k], actual[1][k])
                        for k in expected[1])), (t, type(variant))
                identity_frames += 1
            expected, actual = held['unclaimed'](frame), issue.unclaimed_observations(frame)
            assert len(expected) == len(actual)
            for left, right in zip(expected, actual):
                assert type(left) is type(right)
                for x, y in zip(fields(left), fields(right)):
                    if isinstance(x, np.ndarray):
                        assert x.dtype == y.dtype and x.shape == y.shape and np.array_equal(x, y, equal_nan=x.dtype.kind == 'f')
                    else:
                        assert x == y
                observations += 1
                for mask in (left.mask, np.zeros_like(left.mask)):
                    x, y = held['centroid'](mask), issue.centroid(mask)
                    assert x.dtype == y.dtype and np.array_equal(x, y, equal_nan=True)
    assert issue.centroid is held['centroid'] and issue.identity_points is held['points']
    assert '_sparse_nonzero' not in vars(issue)
    assert loader_stats['versions'] == 3 and loader_stats['boxed'] > 0, loader_stats
    report.update(loader=loader_stats, observations=observations, identity_frames=identity_frames)
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
