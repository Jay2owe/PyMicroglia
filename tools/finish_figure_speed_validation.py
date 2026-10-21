"""Collect final correctness and residual-cost receipts after serial timings."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from ai_research_assistant.tracking import RunTracker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', required=True, type=Path)
    parser.add_argument('--scratch', required=True, type=Path)
    parser.add_argument('--wait-pid', type=int)
    parser.add_argument('--revision', type=int, choices=[4, 5], default=5)
    args = parser.parse_args()
    root, scratch = args.evidence, args.scratch
    python = shutil.which('python')
    suffix = '' if args.revision == 4 else f'-v{args.revision}'
    direct_prefix = 'direct-' if args.revision == 4 else f'direct-v{args.revision}-'
    full_after = f'full-after{args.revision}'
    full_before = 'full-before3' if args.revision == 4 else 'full-before4'
    candidate = scratch/f'candidate{args.revision}'
    jobs = []
    if args.revision == 5:
        # A fresh whole-set baseline controls the host-load shift seen in the
        # final renderer trial; do not attribute faster unchanged fits to it.
        jobs.append(('full-current-baseline', ['tools/benchmark_figure_exports.py',
                     '--source', str(scratch/'baseline'), '--output', str(root/full_before),
                     '--exports', str(scratch/full_before), '--cases', str(root/'prior-cases.json'),
                     '--options', str(root/'fixture-options.json'), '--all']))
    for name, before, after, actions in [
        ('final', full_before, full_after, []),
        ('direct-pair1', direct_prefix+'before1', direct_prefix+'after1', []),
        ('direct-pair2', direct_prefix+'before2', direct_prefix+'after2', []),
        ('parallel', full_after, 'parallel-exports'+suffix, ['all_cell_trace_grid', 'cell_grid']),
    ]:
        for kind, script in [('data', 'check_figure_export_parity.py'), ('svg', 'check_figure_svg_parity.py')]:
            jobs.append((name+'-'+kind, ['tools/'+script, str(root/before), str(root/after),
                         '--output', str(root/(name+'-'+kind+'-parity.json')),
                         *(['--actions', *actions] if actions else [])]))
    jobs.append(('baseline-raster-rounding', ['tools/check_figure_svg_parity.py',
                 str(root/'full-before1'), str(root/'full-before3'), '--actions', 'upheaval_events',
                 '--output', str(root/'baseline-raster-rounding.json')]))
    focused = ['tests/test_figure_export_speed.py', 'tests/test_figure_export_checks.py',
               'tests/test_figure_map_speed.py', 'tests/test_pipeline_table_speed.py',
               'tests/test_trace_workers.py', '-k', 'not native_components', '-q']
    bootstrap = ("import sys,shutil,types; from pathlib import Path; "
                 "sys.executable=shutil.which('python'); "
                 "localtests=types.ModuleType('tests'); localtests.__path__=[str(Path('tests').resolve())]; "
                 "sys.modules['tests']=localtests; import pytest; "
                 f"raise SystemExit(pytest.main({focused!r}))")
    jobs.append(('final-focused', ['-c', bootstrap]))
    jobs.append(('final-profile-extra'+suffix, ['tools/benchmark_figure_exports.py', '--source', str(candidate),
                 '--output', str(root/('final-profile-extra'+suffix)), '--exports', str(scratch/('final-profile-extra'+suffix)),
                 '--cases', str(root/'prior-cases.json'), '--options', str(root/'fixture-options.json'),
                 '--profile', '--actions', 'audit_performance', 'audit_periods', 'audit_decisions',
                 'audit_disagreement', 'audit_focused', 'neighbour_coordination', 'state_histories']))
    for label, source in [('regression-profile-before'+suffix, 'baseline')]:
        jobs.append((label, ['tools/benchmark_figure_exports.py', '--source', str(scratch/source),
                     '--output', str(root/label), '--exports', str(scratch/label),
                     '--cases', str(root/'prior-cases.json'), '--options', str(root/'fixture-options.json'),
                     '--profile', '--actions', 'cell_lifecycle_summary', 'pixel_fate_flow',
                     'measurement_relationship_populations', 'state_support_summary']))
    jobs.append(('worker-resources', ['tools/benchmark_figure_workers.py', '--copies', '4',
                 '--output', str(root/'worker-resources.json')]))
    jobs.append(('residual-measurements', ['tools/benchmark_figure_residuals.py', '--evidence', str(root)]))
    jobs.append(('final-project-contract', ['-c',
                 "import sys,shutil,runpy; sys.executable=shutil.which('python'); "
                 "sys.path.insert(0,'scripts'); sys.argv=['scripts/verify.py','--json']; "
                 "runpy.run_path('scripts/verify.py',run_name='__main__')"]))
    receipts = []
    with RunTracker('Validate complete figure export speed pass', project='PyMicroglia') as tracker:
        if args.wait_pid:
            import psutil
            with tracker.stage('Await serial measurements'):
                try:
                    psutil.Process(args.wait_pid).wait()
                except psutil.NoSuchProcess:
                    pass
        sequence = json.loads((root/'final-sequence.json').read_text())
        if not sequence or sequence[-1]['trial'] != 'final-profile'+suffix or not all(r['successful'] for r in sequence):
            raise RuntimeError('Measurement sequence is incomplete; preserve its failure before validation')
        with tracker.stage('Check outputs and remaining costs', total=len(jobs)):
            for name, arguments in jobs:
                with tracker.item(name):
                    with RunTracker('Figure validation: '+name, project='PyMicroglia',
                                    root=tracker.runs.root) as child:
                        env = {**os.environ, **child.environment(), 'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1'}
                        started = time.perf_counter()
                        with (root/(name+'.log')).open('w', encoding='utf-8') as stream:
                            completed = subprocess.run([python, *arguments], env=env, stdout=stream,
                                                       stderr=subprocess.STDOUT)
                        receipts.append({'name': name, 'command': [python, *arguments],
                                         'exit_code': completed.returncode,
                                         'seconds': time.perf_counter()-started,
                                         'log': name+'.log'})
                    (root/'final-validation.json').write_text(json.dumps(receipts, indent=2))
                    print(name, completed.returncode, flush=True)
    return int(any(row['exit_code'] for row in receipts))


if __name__ == '__main__':
    raise SystemExit(main())
