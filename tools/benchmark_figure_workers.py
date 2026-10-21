"""Test the existing process fitter before exposing it to figure export callers.

Growth inputs repeat the frozen synthetic cell traces under distinct identities;
they are a workload fixture, not additional biological observations.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback
from contextlib import contextmanager
from threading import Event, Thread

import pandas as pd
from ai_research_assistant.tracking import RunTracker


@contextmanager
def sampled_memory():
    import psutil
    parent = psutil.Process()
    stop = Event()
    result = {'sampled_peak_sum_rss_bytes': 0, 'interval_seconds': .1,
              'note': 'Sum of parent and child working sets; shared pages may be counted more than once.'}

    def sample():
        while not stop.is_set():
            total = 0
            for process in [parent, *parent.children(recursive=True)]:
                try:
                    total += process.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            result['sampled_peak_sum_rss_bytes'] = max(result['sampled_peak_sum_rss_bytes'], total)
            stop.wait(.1)

    thread = Thread(target=sample, daemon=True)
    thread.start()
    try:
        yield result
    finally:
        stop.set(); thread.join()


def main():
    import circadian_workbench
    from benchmark_figure_exports import source_identity
    from pymicroglia import workbench
    from pymicroglia.figure_tables import all_cell_traces
    from pymicroglia.figure_tables.saved import Tables
    from pymicroglia.visualisation.figures import load
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--copies', type=int, nargs='+', default=[1, 4])
    args = parser.parse_args()
    source = Tables(Path(__file__).resolve().parents[1]/'tests/fixtures/motion_parity/run', 'parity_A1')
    original = source.table('cell_frame')
    spec = load()['all_cell_trace_grid']
    options = {option.name: option.default for option in spec.options}
    options['metrics'] = ['corrected_mean']
    roots = {'pymicroglia': Path(workbench.__file__).parent,
             'circadian_workbench': Path(circadian_workbench.__file__).parent}
    identities = {name: source_identity(root) for name, root in roots.items()}
    report = {'recipe': __doc__, 'options': options, 'cases': [], 'sources': identities}
    with RunTracker('Test serial and parallel cell figure preparation', project='PyMicroglia') as tracker:
        with tracker.stage('Compare existing worker execution', total=2*len(args.copies)):
            for copies in args.copies:
                frame = pd.concat([original.assign(identity=original.identity+i*int(original.identity.max()))
                                   for i in range(copies)], ignore_index=True)

                class Input:
                    stem = source.stem

                    def table(self, name):
                        assert name == 'cell_frame.csv'
                        return frame.copy()

                    def module_params(self, name):
                        return source.module_params(name)

                entry = {'cells': int(frame.identity.nunique()), 'rows': len(frame),
                         'input_sha256': hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest(),
                         'samples': [], 'exact': False}
                reference = None
                report['cases'].append(entry)
                for workers in [1, 4]:
                    with tracker.item(f'{entry["cells"]} cells, {workers} workers'):
                        with sampled_memory() as memory:
                            started = time.perf_counter()
                            with workbench.analysis_session():
                                prepared, evidence = all_cell_traces.prepare(Input(), options, workers=workers)
                            seconds = time.perf_counter()-started
                        entry['samples'].append({'workers': workers, 'seconds': seconds, **memory})
                        actual = (prepared['traces']['table'], evidence)
                        if reference is None:
                            reference = actual
                        else:
                            try:
                                for before, after in zip(reference, actual):
                                    pd.testing.assert_frame_equal(before, after, check_exact=True)
                                entry['exact'] = True
                            except AssertionError:
                                entry['parity_error'] = traceback.format_exc()
                        print(entry['cells'], workers, seconds, flush=True)
                        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    report['source_unchanged'] = identities == {name: source_identity(root) for name, root in roots.items()}
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
