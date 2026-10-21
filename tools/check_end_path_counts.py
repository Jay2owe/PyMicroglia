"""Compare count-table Issue 017 end-path truncation with the original, exactly.

The recorded continuity input and output stacks with the recorded termination
evidence (no locks, and random locked identities), plus synthetic movies with
identities invented after their ending, gaps, border exits, int32 labels and
an identity absent at its ending: output labels, every action and flag row
must match, or the same error must be raised; also through the history loader.
"""
import argparse
import hashlib
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
    import tifffile
    sys.path[:0] = [str(a.adapter.resolve()), str((a.engine / 'code').resolve())]
    import accepted_history as history
    import end_path_counts as counts
    folder = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
    before = tifffile.imread(folder / '10_blob_residency.tif')
    after = tifffile.imread(folder / '11_issue017_rebase.tif')
    terminations = pd.read_csv(folder / '21_continuity_evidence/out/termination_evidence.csv')
    rng = np.random.default_rng(5)
    # Only identities still present in the output can have a last frame.
    kept = terminations[terminations.identity.astype(int).isin(np.unique(after))]
    # A real movie with invented continuations: each ended identity's last
    # outline is copied into later frames, where truncation must remove it.
    invented = before.copy()
    for row in kept.itertuples(index=False):
        last = int(row.last_source_frame) - 1
        if 0 <= last < len(before) - 1:
            mask = before[last] == int(row.identity)
            for t in range(last + 1 + int(rng.integers(0, 3)), min(len(before), last + 40)):
                invented[t][mask] = int(row.identity)
    cases = [(before, after, kept, 0, set()),
             (before, after, kept, 0, set(int(v) for v in rng.choice(kept.identity, 5))),
             (before, invented, kept, 0, set()),
             (before, invented, kept, 0, set(int(v) for v in rng.choice(kept.identity, 5)))]
    for number in range(150):
        shape = (int(rng.integers(3, 14)), int(rng.integers(4, 24)), int(rng.integers(4, 24)))
        start = rng.integers(0, 6, shape).astype([np.uint16, np.int32][number % 2])
        start[rng.random(shape) < 0.4] = 0
        candidate = start.copy()
        rows = []
        for identity in range(1, 6):
            last = int(rng.integers(0, shape[0]))
            for t in range(last + 1 + int(rng.integers(0, 3)), shape[0]):
                if rng.random() < 0.6:
                    candidate[t][rng.random(shape[1:]) < 0.15] = identity
            rows.append(dict(mechanism=str(rng.choice(['silent_no_foreground', 'other'])),
                             identity=identity, last_source_frame=last + 1 + number % 3))
        cases.append((start, candidate, pd.DataFrame(rows), number % 3,
                      {int(rng.integers(1, 6))} if number % 4 == 0 else set()))

    def outcome(function, case):
        try:
            labels, actions, flags = function(*case)
        except Exception as error:
            return ('error', type(error).__name__, str(error))
        data = np.asarray(labels)
        return ('ok', data.dtype.str, data.shape, hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest(),
                json.dumps(actions, sort_keys=False), json.dumps(flags, sort_keys=False))
    with counts.install_loader(history) as loaded:
        module = history._issue_module(history.I017, 'm3_repair')
        assert loaded and loaded[0]['supported'], loaded
        original = module._end_path_original
        replaced = module._truncate_to_contiguous_end_path
        tick = time.perf_counter(); expected = [outcome(original, c) for c in cases]; before_seconds = time.perf_counter() - tick
        tick = time.perf_counter(); actual = [outcome(replaced, c) for c in cases]; after_seconds = time.perf_counter() - tick
    assert module._truncate_to_contiguous_end_path is original
    for number, (e, f) in enumerate(zip(expected, actual)):
        assert e == f, (number, e[:2], f[:2])

    # _ending_search: the presence completion is replaced by a stand-in that
    # returns the case's candidate movie, so only the report rows differ in code.
    def ending_outcome(function, case):
        labels, candidate, terms, offset, locked = case
        module.complete_presence = lambda *args, **kwargs: (candidate, None, ['attempts'])
        try:
            result, attempts, actions = function(labels, labels, labels, terms, offset, locked)
        except Exception as error:
            return ('error', type(error).__name__, str(error))
        finally:
            del module.complete_presence
        return ('ok', result is candidate or result.tobytes() == candidate.tobytes(), repr(attempts), repr(actions))
    presence = vars(module).get('complete_presence')
    original_ending = module._ending_search
    ending_cases = cases + [(c[0], c[1], c[2], -2, c[4]) for c in cases[4:20]]
    tick = time.perf_counter(); ending_expected = []
    for c in ending_cases:
        ending_expected.append(ending_outcome(original_ending, c)); module.complete_presence = presence
    ending_before = time.perf_counter() - tick
    with counts.install_loader(history) as loaded_again:
        module = history._issue_module(history.I017, 'm3_repair')
        replaced_ending = module._ending_search
        assert replaced_ending is not original_ending
        tick = time.perf_counter(); ending_actual = []
        for c in ending_cases:
            ending_actual.append(ending_outcome(replaced_ending, c)); module.complete_presence = presence
        ending_after = time.perf_counter() - tick
    assert module._ending_search is not replaced_ending
    for number, (e, f) in enumerate(zip(ending_expected, ending_actual)):
        assert e == f, ('ending', number, e[:3], f[:3])
    report = dict(status='pass', cases=len(cases), real_cases=4,
                  ending_cases=len(ending_cases), ending_errors=sum(e[0] == 'error' for e in ending_expected),
                  ending_before_seconds=ending_before, ending_after_seconds=ending_after,
                  ending_counters=loaded_again[0],
                  real_outcomes=[e[:2] if e[0] == 'error' else len(json.loads(e[4])) for e in expected[:4]],
                  errors=sum(e[0] == 'error' for e in expected),
                  before_seconds=before_seconds, after_seconds=after_seconds, counters=loaded[0])
    print(report)
    a.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
