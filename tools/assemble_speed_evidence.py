"""Compare available explicit before/after trials and assemble honest coverage.

Only successful comparison receipts qualify. Existing comparisons are reused
after verified output cleanup, provided both original receipt hashes still match.
Failures and slower trials remain in the inventory; no timing threshold hides them.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from benchmark_pipeline_execution import ROOT, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compare', action='store_true')
    args = parser.parse_args()
    root = ROOT / 'development/speed-pass-20261011'
    base = root / 'complete'
    comparisons = base / 'comparisons'
    comparisons.mkdir(exist_ok=True)
    if args.compare:
        for after in sorted(base.glob('*-after/**/receipt.json')):
            label = after.relative_to(base).parts[0]
            before = base / label.replace('-after', '-before') / after.relative_to(base / label)
            if not before.is_file():
                continue
            current = json.loads(after.read_text())
            action = current.get('id')
            if not action and current.get('operations'):
                action = current['operations'][0]['id']
            if not action:
                continue
            target = comparisons / (label + '-' + action.split(':', 1)[-1] + '.json')
            if target.exists():
                old = json.loads(target.read_text())
                if (old.get('parity') is True and old.get('before_receipt_sha256') == digest(before)
                        and old.get('after_receipt_sha256') == digest(after)):
                    continue
            script = ('compare_pipeline_execution.py' if 'operations' in current
                      else 'compare_action_execution.py')
            completed = subprocess.run([shutil.which('python'), 'tools/' + script,
                str(before), str(after), str(target)], cwd=ROOT, capture_output=True, text=True)
            if completed.returncode not in (0, 1) or not target.exists():
                target.with_suffix('.error.txt').write_text(completed.stderr, encoding='utf-8')
    scope = json.loads((root / 'scope.json').read_text())['required']
    candidates = {key: [] for key in scope}
    all_trials = []
    for path in sorted(comparisons.glob('*.json')):
        if path.name.endswith('.cleanup.json'):
            continue
        row = json.loads(path.read_text())
        if row.get('id') not in candidates:
            continue
        row['receipt'] = path.relative_to(root).as_posix()
        all_trials.append(row)
        if row.get('parity') is True and row.get('status') == 'measured':
            candidates[row['id']].append(row)
    # Preserve separately reviewed large-index measurement evidence.
    previous = json.loads((root / 'coverage-evidence.json').read_text())
    for row in previous['operations']:
        if (row.get('parity') is True and row.get('kind') == 'execution'
                and row.get('id') in candidates and (root / row.get('receipt', '')).is_file()
                and not row.get('receipt', '').startswith('complete/comparisons/')):
            candidates[row['id']].append(row)
    chosen = []
    for key, rows in candidates.items():
        if rows:
            # Prefer final repaired executions and the complete primary pipeline
            # over its optional narrower examples. Never choose by speed.
            rows.sort(key=lambda r: ('extra-final-' in r['receipt'],
                'pipeline-final-' in r['receipt'],
                'dataset-final-' in r['receipt'] or 'dataset-repaired-' in r['receipt'],
                '/pipeline-' in r['receipt'],
                (root / r['receipt']).stat().st_mtime))
            chosen.append(rows[-1])
        else:
            chosen.append({'id': key, 'kind': 'execution', 'status': 'unverified', 'parity': False})
    (root / 'coverage-evidence.json').write_text(json.dumps({'operations': chosen}, indent=2), encoding='utf-8')
    (base / 'comparison-inventory.json').write_text(json.dumps({'trials': all_trials}, indent=2), encoding='utf-8')
    print(json.dumps({'required': len(scope), 'verified': sum(bool(r.get('parity')) for r in chosen),
        'unverified': [r['id'] for r in chosen if not r.get('parity')],
        'failed_comparisons': [r['receipt'] for r in all_trials
                               if not r.get('parity') or r.get('status') != 'measured']}, indent=2))


if __name__ == '__main__':
    main()
