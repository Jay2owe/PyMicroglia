"""Compare boxed copies of frozen observation masks with mask.copy(), exactly.

Part 1: frames of accepted-history label movies become read-only snapshots as
the frozen observation reader makes them (plain, and through the memory pool
with its default and a tiny disk threshold). For every identity in the frame,
zero, a negative, absent and out-of-range identities, the boxed copy must have
the same type, dtype, shape, layout flags and values as the LabelMask's own
copy; writable snapshots and plain Boolean masks are copied as before.

Part 2: presence completion (forward and bracketed, as check_presence_completion
builds them) runs with the frozen observation reader on both sides, plain and
through the memory pool; the completed movie, recovered pixels and attempt rows
must be equal with and without the presence adapters.
"""
import argparse
import importlib.util
import json
from contextlib import ExitStack
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import tifffile

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'development/cd68_full_red_green/rigid_analysis/A1/tracking/engine'
HISTORY = ENGINE / 'm22_accepted_history/pymicroglia_95_full_A1/mid/accepted_history'
FROZEN = ROOT / 'validation/completed_base_20261012/memory/snapshots/combined_speed166_candidate/pymicroglia/tracking/frozen_io.py'


def describe(value):
    if not isinstance(value, np.ndarray):
        return (type(value).__name__,)
    flags = value.flags
    return (type(value).__name__, value.dtype.str, value.shape, flags.c_contiguous, flags.f_contiguous,
            flags.writeable, flags.owndata)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=70)
    a = parser.parse_args()
    sys.path[:0] = [str(ENGINE / 'code'), str(a.adapters.resolve())]
    import accepted_history as history
    import presence
    import array_storage
    import presence_area_table, presence_contrast_crop, presence_mask_boxes, presence_settle_boxes
    import presence_descriptor_medians
    spec = importlib.util.spec_from_file_location('frozen_io_under_test', FROZEN)
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    # The adapter recognises masks by class name and module, as installed.
    frozen.LabelMask.__module__ = 'pymicroglia.tracking.frozen_io'
    a.scratch.mkdir(parents=True, exist_ok=True)
    report = dict(failures=[], unit_cases=0, unit_stats={}, completion={})

    def fail(**item):
        if len(report['failures']) < 30:
            report['failures'].append(item)

    # Part 1.
    stats = dict(boxed=0, whole=0, held_tables=0, held_table=0, held_whole=0, areas_reused=0,
                 copy_boxed=0, copy_whole=0, copy_frames=0)
    found = presence_mask_boxes.helpers(stats)
    mask_copy = found['_mask_copy']
    rng = np.random.default_rng(20261024)
    stacks = {name: tifffile.imread(HISTORY / name) for name in
              ('11_issue017_rebase.tif', '10_blob_residency.tif', '57_well_b2_issue001_terminal_projection_diversion.tif')}
    pools = {}
    for name, minimum in (('pool', 32 * 1024 ** 2), ('pool_tiny', 1 << 10)):
        pools[name] = array_storage.Pool(a.scratch / name, minimum_bytes=minimum, temporary_ram_bytes=64 * 1024 ** 2)
    timing = dict(original=0.0, adapted=0.0)
    for stack_name, stack in stacks.items():
        for t in rng.choice(len(stack), 25, replace=False):
            frame = stack[int(t)]
            values = np.unique(frame)
            identities = [int(v) for v in values] + [-1, int(values.max()) + 1, int(values.max()) + 40, 70000]
            for form in ('plain', 'writable', 'pool', 'pool_tiny', 'int32', 'fortran'):
                with ExitStack() as scope:
                    source = frame
                    if form == 'int32':
                        source = frame.astype(np.int32)
                    if form == 'fortran':
                        source = np.asfortranarray(frame)
                    if form.startswith('pool'):
                        scope.enter_context(array_storage.install(pools[form]))
                        snapshot = pools[form].snapshot(source)
                    else:
                        snapshot = np.array(source, copy=True)
                    if form != 'writable':
                        snapshot.flags.writeable = False
                    for identity in identities:
                        mask = frozen.LabelMask(snapshot, identity)
                        t0 = time.perf_counter()
                        try:
                            expected = ('ok', mask.copy())
                        except Exception as error:
                            expected = ('error', type(error).__name__, str(error))
                        t1 = time.perf_counter()
                        try:
                            actual = ('ok', mask_copy(mask))
                        except Exception as error:
                            actual = ('error', type(error).__name__, str(error))
                        t2 = time.perf_counter()
                        timing['original'] += t1 - t0
                        timing['adapted'] += t2 - t1
                        report['unit_cases'] += 1
                        if expected[0] != actual[0] or (expected[0] == 'error' and expected != actual):
                            fail(stack=stack_name, t=int(t), form=form, identity=identity, problem='outcome')
                        elif expected[0] == 'ok' and (describe(expected[1]) != describe(actual[1])
                                                      or not np.array_equal(expected[1], actual[1])):
                            fail(stack=stack_name, t=int(t), form=form, identity=identity,
                                 problem=f'{describe(expected[1])} vs {describe(actual[1])}')
            plain_mask = frame == int(values[-1])
            report['unit_cases'] += 1
            if describe(plain_mask.copy()) != describe(mask_copy(plain_mask)) or not np.array_equal(
                    plain_mask.copy(), mask_copy(plain_mask)):
                fail(stack=stack_name, t=int(t), form='boolean', problem='plain Boolean mask')
    report['unit_stats'] = {k: v for k, v in stats.items() if k.startswith('copy')}
    report['unit_seconds'] = {k: round(v, 3) for k, v in timing.items()}

    # Part 2.
    repair = history._issue_module(history.CONTINUITY_OPS, 'continuity_repair')
    labels_all = tifffile.imread(HISTORY / '10_blob_residency.tif')
    raw_all = tifffile.imread(HISTORY / '20_continuity_evidence_inputs/out/raw.tif')
    lag_all = tifffile.imread(HISTORY / '20_continuity_evidence_inputs/out/lag.tif')
    original_observations = presence.frame_observations
    counters = {}
    rows = cases = 0
    for storage in ('plain', 'pool'):
        for start in (40, 360):
            stop = min(len(labels_all), start + a.frames)
            labels = labels_all[start:stop]
            raw, lag = raw_all[start:stop], lag_all[start:stop]
            present = {int(v): np.flatnonzero(np.any(labels == v, axis=(1, 2))) for v in np.unique(labels)[1:]}
            endings = [dict(identity=i, start_t=int(f[-1]) + 1, end_t=len(labels) - 1)
                       for i, f in present.items() if 5 <= f[-1] < len(labels) - 2]
            gaps = []
            for i, f in present.items():
                for h in np.flatnonzero(np.diff(f) > 1)[:2]:
                    gaps.append(dict(identity=i, start_t=int(f[h]) + 1, end_t=int(f[h + 1]) - 1))
            for kind, intervals in (('ending', endings), ('bracketed', gaps)):
                if not intervals:
                    continue
                params = repair._presence_params(intervals, bracketed=kind == 'bracketed')
                function = 'complete_presence' if kind == 'ending' else 'complete_presence_bidirectional'
                results = []
                for adapted in (False, True):
                    with ExitStack() as scope:
                        copy_labels = None
                        if storage == 'pool':
                            pool = array_storage.Pool(a.scratch / f'completion_{cases}_{adapted}',
                                                      temporary_ram_bytes=64 * 1024 ** 2)
                            scope.enter_context(array_storage.install(pool))
                            copy_labels = pool.snapshot
                        presence.frame_observations = frozen.observation_reader(original_observations, copy_labels)
                        scope.callback(setattr, presence, 'frame_observations', original_observations)
                        installed = []
                        if adapted:
                            for adapter in (presence_contrast_crop, presence_area_table, presence_mask_boxes,
                                            presence_settle_boxes, presence_descriptor_medians):
                                counter = scope.enter_context(adapter.install(presence))
                                assert counter['supported'], adapter.__name__
                                installed.append((adapter.__name__, counter))
                        t0 = time.perf_counter()
                        output = getattr(presence, function)(labels, raw, lag, params)
                        seconds = time.perf_counter() - t0
                        results.append((tuple(np.array(x) for x in output[:2]), output[2].copy(), seconds))
                        for name, counter in installed:
                            total = counters.setdefault(name, {})
                            for key, value in counter.items():
                                if type(value) is int:
                                    total[key] = total.get(key, 0) + value
                cases += 1
                (left, left_rows, left_s), (right, right_rows, right_s) = results
                try:
                    for x, y in zip(left, right):
                        assert x.dtype == y.dtype and np.array_equal(x, y)
                    pd.testing.assert_frame_equal(left_rows, right_rows, check_exact=True)
                    rows += len(left_rows)
                except AssertionError as error:
                    fail(storage=storage, start=start, kind=kind, problem=str(error)[:300])
                report['completion'][f'{storage}:{start}:{kind}'] = dict(rows=len(left_rows), original_seconds=round(left_s, 2),
                                                                        adapted_seconds=round(right_s, 2))
    report['completion_counters'] = counters
    copied = counters.get('presence_mask_boxes', {}).get('copy_boxed', 0)
    report['status'] = 'pass' if (not report['failures'] and rows and copied
                                  and report['unit_stats']['copy_boxed']) else 'fail'
    a.output.write_text(json.dumps(report, indent=1, default=str), encoding='utf-8')
    print(json.dumps(report, indent=1, default=str))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
