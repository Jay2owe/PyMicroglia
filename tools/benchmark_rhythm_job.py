"""Measure a fresh UI/package rhythm run, progress, memory and exact statistics."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from threading import Event, Thread

import pandas as pd
import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', help='UI server; omit for a fresh direct package run.')
    parser.add_argument('--workers', type=int)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--stem', required=True)
    parser.add_argument('--metric', action='append', required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--process', type=int)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    observer = None
    if args.process:
        import psutil
        observer = psutil.Process(args.process)
    source = args.run / 'measure' / args.stem / 'cell_frame.csv'
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    submitted = time.perf_counter()
    progress, latencies, memory = [], [], []
    if args.url:
        if args.workers is not None:
            parser.error('--workers applies to direct package runs only.')
        response = requests.post(args.url + '/api/exploration/rhythm-jobs', json={
            'run': str(args.run.resolve()), 'stem': args.stem,
            'options': {'metrics': args.metric}}, timeout=15)
        response.raise_for_status()
        job = response.json()
    else:
        import os
        import psutil
        from pymicroglia.ui.exploration_rhythms import run_rhythm_analysis
        observer = psutil.Process(os.getpid())
        stopped = Event()

        def sample_memory():
            while not stopped.is_set():
                processes = [observer, *observer.children(recursive=True)]
                resident = 0
                for process in processes:
                    try:
                        resident += process.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                memory.append(resident)
                stopped.wait(.5)

        monitor = Thread(target=sample_memory, daemon=True)
        monitor.start()
        start = datetime.now(timezone.utc).isoformat()
        try:
            result = run_rhythm_analysis(args.run, args.stem, {'metrics': args.metric},
                                         workers=args.workers, progress=progress.append)
        finally:
            stopped.set()
            monitor.join()
        job = {'state': 'completed', 'started_at': start,
               'finished_at': datetime.now(timezone.utc).isoformat(),
               'rhythm_result': result['source'], 'progress': progress[-1],
               'cells': len({row['identity'] for row in result['rows']})}
    previous = 0
    while args.url:
        sample = time.perf_counter()
        response = requests.get(args.url + '/api/jobs/' + job['id'], timeout=10)
        response.raise_for_status()
        job = response.json()
        latencies.append(time.perf_counter() - sample)
        progress.append(job['progress'])
        if observer:
            processes = [observer, *observer.children(recursive=True)]
            resident = 0
            for process in processes:
                try:
                    resident += process.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            memory.append(resident)
        if job['state'] in {'completed', 'failed'}:
            break
        time.sleep(.5)
    assert job['state'] == 'completed', job
    assert not job.get('reused'), 'Benchmark requires a fresh calculation, not result reuse.'
    result_path = Path(job['rhythm_result'])
    result = json.loads(result_path.read_text(encoding='utf-8'))
    for current in progress:
        assert previous <= current['completed'] <= current['total'], current
        previous = current['completed']
    exact = None
    if args.reference:
        reference = json.loads(args.reference.read_text(encoding='utf-8'))
        key = lambda row: (row['identity'], row['metric'])
        expected = {key(row): row for row in reference['rows']}
        actual = {key(row): row for row in result['rows']}
        assert expected.keys() <= actual.keys()
        assert all(actual[identity] == row for identity, row in expected.items())
        old = pd.read_csv(args.reference.with_name('statistics.csv'))
        new = pd.read_csv(result_path.with_name('statistics.csv'))
        new = new[new.metric.isin(reference['metrics'])].reset_index(drop=True)
        columns = [column for column in old if 'run_record' not in column]
        pd.testing.assert_frame_equal(old[columns], new[columns], check_exact=True, check_dtype=False)
        exact = True
    after = hashlib.sha256(source.read_bytes()).hexdigest()
    assert before == after == result['source_sha256']
    seconds = (datetime.fromisoformat(job['finished_at']) - datetime.fromisoformat(job['started_at'])).total_seconds()
    receipt = {'job': job, 'duration_seconds': seconds,
               'request_to_ready_seconds': time.perf_counter() - submitted,
               'execution': result['execution'], 'cells': job['cells'],
               'metrics': result['metrics'], 'result_rows': len(result['rows']),
               'source_csv_sha256': after, 'exact_reference_statistics': exact,
               'reference': str(args.reference) if args.reference else None,
               'peak_sampled_process_tree_rss_bytes': max(memory) if memory else None,
               'memory_note': 'Sampled summed resident memory of server and children; shared pages can be counted more than once.',
               'mode': 'ui' if args.url else 'package',
               'max_progress_latency_seconds': max(latencies) if latencies else None,
               'progress': progress}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in receipt.items() if key not in {'job', 'progress'}}, indent=2), flush=True)


if __name__ == '__main__':
    main()
