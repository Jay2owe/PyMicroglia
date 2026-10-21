"""Write the whole-group exactness queue for one composed snapshot.

Checks run one at a time (tools/run_tracking_performance_checks.py): the full
476-frame identity history, 99-frame evidence plus all accepted corrections,
the full 476-frame correction evidence, and the twelve earlier stages.
"""
import argparse
import json
from pathlib import Path

MEMORY = Path('validation/completed_base_20261012/memory')
ENGINE = 'development/cd68_full_red_green/rigid_analysis/A1/tracking/engine'
CONFIG = 'development/cd68_full_red_green/rigid_batch/.j/A1/inputs/motion_config.json'


def guarded(receipt, limit, command):
    return ['{python}', '-m', 'auto_organotypic.resource_guard', '--receipt', str(receipt),
            '--reserve-gib', '1', '--limit-gib', str(limit), '--start-headroom-gib', '1',
            '--stable-seconds', '0', '--', '{python}', *command]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('snapshot', help='folder name under memory/snapshots')
    p.add_argument('--tag', required=True, help='suffix for run folders')
    p.add_argument('--wait-pid', type=int)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    snap = MEMORY / 'snapshots' / a.snapshot
    runs = MEMORY / 'runs'
    checks = [
        dict(name=f'Full 476-frame identity history ({a.snapshot})',
             report=str(runs / f'history_{a.tag}' / 'report.json'), log=str(runs / f'history_{a.tag}.log'),
             command=guarded(runs / f'history_{a.tag}_resources.json', 6, [
                 'tools/benchmark_identity_reservation.py', '--engine', ENGINE, '--config', CONFIG,
                 '--snapshot', str(snap), '--adapter', str(snap / 'pymicroglia/tracking'),
                 '--output', str(runs / f'history_{a.tag}'), '--run', 'pymicroglia_95_full_A1',
                 '--stem', '95_full_A1', '--native-history', '--stage', 'history'])),
        dict(name=f'99-frame evidence and all 114 accepted corrections ({a.snapshot})',
             report=str(runs / f'corrections_{a.tag}' / 'report.json'), log=str(runs / f'corrections_{a.tag}.log'),
             command=guarded(runs / f'corrections_{a.tag}_resources.json', 4, [
                 'tools/benchmark_completed_evidence.py',
                 '--engine', str(MEMORY / 'diagnostic-data/baseline/engine'), '--snapshot', str(snap),
                 '--config', str(MEMORY / 'peak_recording.json'), '--output', str(runs / f'corrections_{a.tag}'),
                 '--run', 'profile', '--stem', '95_A1',
                 '--reference', str(MEMORY / 'diagnostic-data/baseline/engine/m23_completed_feedback/profile'),
                 '--reference-evidence', str(runs / 'evidence_fragment_candidate/evidence_hashes.json')])),
        dict(name=f'Full 476-frame correction evidence preparation ({a.snapshot})',
             report=str(runs / f'evidence_{a.tag}' / 'report.json'), log=str(runs / f'evidence_{a.tag}.log'),
             command=guarded(runs / f'evidence_{a.tag}_resources.json', 4, [
                 'tools/benchmark_completed_evidence.py', '--engine', ENGINE, '--snapshot', str(snap),
                 '--config', CONFIG, '--output', str(runs / f'evidence_{a.tag}'),
                 '--run', 'pymicroglia_95_full_A1', '--stem', '95_full_A1', '--evidence-only'])),
        dict(name=f'All twelve earlier native tracking stages ({a.snapshot})',
             report=str(runs / f'earlier_{a.tag}' / 'summary.json'), log=str(runs / f'earlier_{a.tag}.log'),
             command=guarded(runs / f'earlier_{a.tag}_resources.json', 4, [
                 'tools/profile_tracking_memory.py', '--source', str(snap), '--config', CONFIG,
                 '--movie', '95_full_A1', '--output', str(MEMORY / 'diagnostic-data' / f'earlier_{a.tag}'),
                 '--report', str(runs / f'earlier_{a.tag}'), '--bounded', '--before-history-only',
                 '--working-array-min-bytes', '1048576', '--storage-source', str(snap / 'array_storage.py')])),
    ]
    queue = dict(root=str(Path.cwd()), checks=checks)
    if a.wait_pid:
        import psutil
        process = psutil.Process(a.wait_pid)
        queue['wait_for'] = dict(pid=a.wait_pid, created=process.create_time(),
                                 command_token=Path(process.cmdline()[1]).name)
    a.output.write_text(json.dumps(queue, indent=2) + '\n')
    print(a.output)


if __name__ == '__main__':
    main()
