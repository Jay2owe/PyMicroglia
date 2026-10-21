"""Verify the final follow-ups and file an honest figure-speed completion receipt."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

from ai_research_assistant.tracking import RunTracker
from check_figure_speed_completion import check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', required=True, type=Path)
    args = parser.parse_args()
    root = args.evidence
    commands = []
    pairs = [(f'regression-before{i}', f'regression-after{i}', f'followup-{i}')
             for i in [8, 9]]
    pairs += [(f'causal-v2-{a}-control', f'causal-v2-{b}-current', f'causal-{a}-{b}')
              for a, b in [(1, 2), (4, 3), (5, 6), (8, 7)]]
    for before, after, label in pairs:
        for kind, tool in [('data', 'check_figure_export_parity.py'),
                           ('svg', 'check_figure_svg_parity.py')]:
            name = f'{label}-{kind}-parity'
            commands.append((name, [sys.executable, 'tools/'+tool, str(root/before),
                                   str(root/after), '--output', str(root/(name+'.json'))]))
    commands += [
        ('completion-check-tests', [sys.executable, '-m', 'pytest',
                                    'tests/test_figure_export_checks.py', '-q']),
        ('final-report', [sys.executable, 'tools/report_figure_speed.py',
                          '--evidence', str(root), '--final', '--revision', '5']),
    ]
    receipts = []
    with RunTracker('Close figure speed verification and retain evidence', project='PyMicroglia') as tracker:
        with tracker.stage('Verify follow-up outputs and report', total=len(commands)):
            for name, command in commands:
                with tracker.item(name):
                    with (root/(name+'.log')).open('w', encoding='utf-8') as stream:
                        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                                env=tracker.environment())
                    receipts.append({'name': name, 'command': command, 'exit_code': result.returncode})
                    (root/'closure-checks.json').write_text(json.dumps(receipts, indent=2))
                    if result.returncode:
                        raise RuntimeError('Closure verification failed: '+name)
    timings = json.loads((root/'final-timings.json').read_text())
    diagnostic = json.loads((root/'causal-regression-diagnostic.json').read_text())
    reviews = []
    for row in diagnostic['summary']:
        before = statistics.median(row['control_seconds'])
        delta = row['median_change_seconds']
        resolved = delta <= max(.1, .05*before)
        reviews.append({
            'action': row['action'], 'status': 'resolved' if resolved else 'open',
            'reason': 'Two ABBA sequences in one interpreter restore all changed functions reached by this action. '
                      'The controlled median difference is below the declared material threshold. '
                      'Separate-process repeats remain slower and are retained; attribution to one host mechanism is not established.',
            'controlled_change_seconds': delta,
            'evidence': ['causal-regression-diagnostic.json', 'regression-followups.json',
                         'closure-checks.json', 'final-timings.json'],
            'limit': 'This resolves a reproducible code-cost regression hypothesis for the tested sequence; '
                     'it does not promise identical cold-process latency or identify the cause of process-to-process variation.',
        })
    (root/'regression-review.json').write_text(json.dumps(reviews, indent=2))
    residuals = [
        {'candidate': 'SVG image expansion', 'status': 'closed', 'evidence': 'final-svg-parity.json',
         'decision': 'Keep native nearest-neighbour pixels with conservative fallbacks; all 91 exported families pass.'},
        {'candidate': 'Full cumulative coverage history', 'status': 'closed', 'evidence': 'map-timings.json',
         'decision': 'Keep final-only accumulation; exact values, repeated growth measurements and much lower allocation.'},
        {'candidate': 'Repeated saved scalar decoding', 'status': 'closed', 'evidence': 'residual-measurements.json',
         'decision': 'Keep bounded per-read reuse; repeated 57,912-cell decoder measurements are about three times faster.'},
        {'candidate': 'Repeated engine identity capture', 'status': 'closed', 'evidence': 'direct-pair1-data-parity.json',
         'decision': 'Keep existing bounded analysis session around direct refits and recurrence preparation; '
                     'fresh boundary verification and every scientific run record remain.'},
        {'candidate': 'Independent cell fitting', 'status': 'closed', 'evidence': 'worker-resources.json',
         'decision': 'Expose optional workers for supported trace grids; 12-cell preparation 78.0 to 24.9 seconds. '
                     'Retain serial default because startup and fourfold sampled working-set cost can lose on small grids.'},
        {'candidate': 'Settings conversion', 'status': 'closed', 'evidence': 'residual-measurements.json',
         'decision': 'Retain validation and detached settings. Empirical all-call upper bounds are 5.26 and 9.87 seconds '
                     'on this catalogue (nested costs, not additive), each below 3.3% of the final complete export. '
                     'This is a workload-specific bound, not a universal growth claim.'},
        {'candidate': 'Native analysis-grid replacement', 'status': 'closed', 'evidence': 'residual-measurements.json',
         'decision': 'Tested existing renderer; rejected replacement because sizing and required title differ. '
                     'Complete current grid is 15.43 seconds, so even deleting it would save only 5.1% of the full set.'},
        {'candidate': 'Required model and surrogate fits', 'status': 'closed', 'evidence': 'fit-reuse-audit.json',
         'decision': 'Retain exact requested models, seeds, surrogate counts and evidence. Exported records reveal 37 '
                     'distinct fit signatures; apparently similar grids use different scientific settings. '
                     'Hidden fits are outside this inventory. General fitting-engine changes are outside this figure-export pass.'},
        {'candidate': 'Artist discovery and four small regressions', 'status': 'closed'
         if all(r['status'] == 'resolved' for r in reviews) else 'open',
         'evidence': 'regression-review.json',
         'decision': 'Replace artist-tree walking with direct axes images; measured helper cost below 1 ms. '
                     'Final same-process restoration experiment has no material median increase.'},
    ]
    fits = json.loads((root/'fit-reuse-audit.json').read_text())
    names = {name for group in fits['repeated_across_figures'] for name in group['actions']}
    bound = sum(row['full_batch_after_seconds'] for row in timings['cases'] if row['action'] in names)
    residuals.append({
        'candidate': 'Cross-figure reuse of visible repeated fits', 'status': 'closed'
        if bound < .1*timings['full_batch']['after_seconds'] else 'open',
        'evidence': 'fit-reuse-audit.json',
        'decision': 'Conservative bound removes the entire exports containing repeated visible fits, not just fitting. '
                    'No persistent cache introduced; freshness and exact settings retained.',
        'actions': sorted(names), 'upper_bound_seconds_on_final_workload': bound,
        'limit': 'Only exported estimate_period records; no inference about hidden surrogate fits or larger recordings.',
    })
    (root/'residual-decisions.json').write_text(json.dumps(residuals, indent=2))
    for row in timings['cases']:
        if row['material_regression']:
            row['reviewed_evidence'] = 'regression-review.json'
    verification = json.loads((root/'verification/verification.json').read_text())
    final_validation = json.loads((root/'final-validation.json').read_text())
    evidence = {
        'required_actions': [r['action'] for r in timings['cases']], 'cases': timings['cases'],
        'source_stable': timings['source_stable'],
        'functions': json.loads((root/'function-coverage.json').read_text()),
        'residuals': residuals,
        'receipts': ['final-data-parity.json', 'final-svg-parity.json', 'final-focused.log',
                     'final-validation.json', 'verification/verification.json', 'closure-checks.json',
                     'source-recovery.json', 'causal-regression-diagnostic.json'],
        'focused_no_new_failures': not verification['new_test_failures'] and
                                  '35 passed' in (root/'final-focused.log').read_text(),
        'project_gate_passed': next(r for r in final_validation
                                   if r['name'] == 'final-project-contract')['exit_code'] == 0,
    }
    (root/'acceptance-evidence.json').write_text(json.dumps(evidence, indent=2))
    completion = check(evidence, root)
    (root/'completion.json').write_text(json.dumps(completion, indent=2))
    print(json.dumps(completion, indent=2))
    return 0 if completion['performance_evidence_verified'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
