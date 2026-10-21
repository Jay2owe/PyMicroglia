"""Run the final serial measurement sequence without competing benchmark jobs."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from ai_research_assistant.tracking import RunTracker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scratch', required=True, type=Path)
    parser.add_argument('--evidence', required=True, type=Path)
    parser.add_argument('--wait-pid', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--revision', type=int, choices=[4, 5], default=5)
    args = parser.parse_args()
    with RunTracker('Finish complete figure export speed measurements', project='PyMicroglia') as tracker:
        if args.wait_pid:
            import psutil
            with tracker.stage('Await focused checks'):
                try:
                    psutil.Process(args.wait_pid).wait()
                except psutil.NoSuchProcess:
                    pass
        verification = json.loads((args.evidence/'verification/verification.json').read_text())
        if not verification['tests_executed'] or verification['new_test_failures']:
            raise RuntimeError('Focused verification has new failures; inspect verification.json')
        with tracker.stage('Verify final rendering guards'):
            with (args.evidence/'final-rendering-tests.log').open('w', encoding='utf-8') as stream:
                subprocess.run([sys.executable, '-m', 'pytest', 'tests/test_figure_export_speed.py',
                                'tests/test_figure_export_checks.py', '-q'],
                               env={**os.environ, **tracker.environment()}, stdout=stream,
                               stderr=subprocess.STDOUT, check=True)
        candidate = args.scratch/f'candidate{args.revision}'
        full_after = f'full-after{args.revision}'
        direct_prefix = 'direct-' if args.revision == 4 else f'direct-v{args.revision}-'
        parallel_name = 'parallel-exports' if args.revision == 4 else f'parallel-exports-v{args.revision}'
        profile_name = 'final-profile' if args.revision == 4 else f'final-profile-v{args.revision}'
        from benchmark_figure_exports import source_identity, digest
        if args.resume and candidate.exists():
            if source_identity(Path('src/pymicroglia')) != source_identity(candidate/'pymicroglia'):
                raise RuntimeError('Live candidate differs from the measured frozen candidate')
        else:
            shutil.copytree('src/pymicroglia', candidate/'pymicroglia',
                            ignore=shutil.ignore_patterns('__pycache__'))
        options = json.loads((args.evidence/'fixture-options.json').read_text())
        for name in ['all_cell_trace_grid', 'cell_grid']:
            options.setdefault(name, {})['workers'] = 4
        (args.evidence/'worker-options.json').write_text(json.dumps(options, indent=2))
        # Repeat the actual slower cases and the principal changed paths.
        repeated = ['own_clock_composite', 'cell_analysis_grid', 'metric_rhythm_matrix',
                    'radial_occupancy_rhythms', 'territory_anchoring', 'regime_reporter_bridge',
                    'cell_lifecycle_summary', 'accepted_state_card', 'recurrence_wall',
                    'recording_barcode', 'tissue_coverage_gaps', 'tissue_tectonics',
                    'pixel_fate_flow', 'measurement_relationship_populations', 'state_support_summary']
        if args.resume and args.revision == 4:
            a, b = [json.loads((args.evidence/('full-'+trial)/'summary.json').read_text())
                    for trial in ['before3', 'after3']]
            previous = {row['action']: row for row in a['cases']}
            for row in b['cases']:
                old = previous[row['action']]['seconds']
                if row['seconds']-old > max(.1, .05*old) and row['action'] not in repeated:
                    repeated.append(row['action'])
        jobs = [(full_after, candidate, ['--all']), ('full-before3', args.scratch/'baseline', ['--all'])]
        for trial, source in [('before1', args.scratch/'baseline'), ('after1', candidate),
                              ('after2', candidate), ('before2', args.scratch/'baseline')]:
            jobs.append((direct_prefix+trial, source, ['--direct', '--actions', 'all_cell_trace_grid',
                         '--options', str(args.evidence/'direct-options.json')]))
        first_repeat = 2 * (args.revision-2)
        for trial, source in [(f'before{first_repeat}', args.scratch/'baseline'), (f'after{first_repeat}', candidate),
                              (f'after{first_repeat+1}', candidate), (f'before{first_repeat+1}', args.scratch/'baseline')]:
            jobs.append(('repeat-'+trial, source, ['--actions', *repeated]))
        jobs.append((parallel_name, candidate, ['--actions', 'all_cell_trace_grid', 'cell_grid',
                     '--options', str(args.evidence/'worker-options.json')]))
        jobs.append((profile_name, candidate, ['--profile', '--actions', 'recurrence_wall',
                     'recording_barcode', 'accepted_state_card', 'tissue_coverage_gaps', 'tissue_tectonics',
                     'pixel_fate_flow', 'cell_lifecycle_summary', 'measurement_relationship_populations',
                     'state_support_summary']))
        receipts = []
        if args.resume and (args.evidence/'final-sequence.json').exists():
            history = args.evidence/'initial-sequence-failure.json'
            if not history.exists():
                shutil.copyfile(args.evidence/'final-sequence.json', history)
            log = args.evidence/'repeat-before4.log'
            if log.exists() and not (args.evidence/'initial-stage-count-failure.log').exists():
                shutil.copyfile(log, args.evidence/'initial-stage-count-failure.log')
        with tracker.stage('Execute paired export trials', total=len(jobs)):
            for name, source, extra in jobs:
                with tracker.item(name):
                    if name.startswith('repeat-'):
                        old_rows = json.loads((args.evidence/'full-before3/summary.json').read_text())['cases']
                        new_rows = json.loads((args.evidence/full_after/'summary.json').read_text())['cases']
                        old_by_name = {row['action']: row['seconds'] for row in old_rows}
                        for row in new_rows:
                            old = old_by_name[row['action']]
                            if row['seconds']-old > max(.1, .05*old) and row['action'] not in repeated:
                                repeated.append(row['action'])
                        extra = ['--actions', *repeated]
                    summary_path = args.evidence/name/'summary.json'
                    if args.resume and summary_path.exists():
                        saved = json.loads(summary_path.read_text())
                        stable = (saved.get('source_unchanged') and saved.get('dependencies_unchanged') and
                                  saved.get('harness_unchanged') and
                                  saved['harness_sha256'] == digest(Path('tools/benchmark_figure_exports.py')) and
                                  saved['sources'] == source_identity(source/'pymicroglia') and
                                  all(value['sources'] == source_identity(Path(value['path']))
                                      for value in saved['dependencies'].values()))
                        if not stable or not all(row['status'] == 'measured' for row in saved['cases']):
                            raise RuntimeError('Cannot reuse changed or incomplete measurement: '+name)
                        receipts.append({'trial': name, 'successful': True, 'reused_verified_summary': str(summary_path)})
                        (args.evidence/'final-sequence.json').write_text(json.dumps(receipts, indent=2))
                        print(name, 'verified existing trial', flush=True)
                        continue
                    command = [sys.executable, 'tools/benchmark_figure_exports.py',
                               '--source', str(source), '--output', str(args.evidence/name),
                               '--exports', str(args.scratch/name), '--cases', str(args.evidence/'prior-cases.json')]
                    # argparse's last options value deliberately wins for the
                    # direct-call and worker-specific selections.
                    command += ['--options', str(args.evidence/'fixture-options.json'), *extra]
                    # Each independent trial has its own item inventory. An
                    # explicit root creates its coordinator; the subprocess
                    # attaches to that coordinator using the supported API.
                    with RunTracker('Figure export trial: '+name, project='PyMicroglia',
                                    root=tracker.runs.root) as trial:
                        with (args.evidence/(name+'.log')).open('w', encoding='utf-8') as stream:
                            completed = subprocess.run(command, env=trial.environment(),
                                                       stdout=stream, stderr=subprocess.STDOUT)
                        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
                        okay = completed.returncode == 0 and summary.get('cases') and all(
                            row['status'] == 'measured' for row in summary['cases'])
                        if not okay:
                            raise RuntimeError('Incomplete export trial: '+name)
                    receipts.append({'trial': name, 'command': command, 'successful': bool(okay),
                                     'exit_code': completed.returncode, 'summary': str(summary_path)})
                    (args.evidence/'final-sequence.json').write_text(json.dumps(receipts, indent=2))
                    print(name, 'completed' if okay else 'FAILED', flush=True)
                    if not okay:
                        raise RuntimeError('Incomplete export trial: '+name)


if __name__ == '__main__':
    main()
