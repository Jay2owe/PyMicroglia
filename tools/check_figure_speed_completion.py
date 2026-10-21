"""Fail closed on incomplete figure speed evidence; preserve existing project failures."""
import argparse
import json
import math
from pathlib import Path


def check(document, root):
    problems = []
    required = document.get('required_actions', [])
    if not required or len(set(required)) != len(required):
        problems.append('missing or duplicate required actions')
    cases = document.get('cases', [])
    if len(cases) != len(required) or {row.get('action') for row in cases} != set(required):
        problems.append('missing required execution')
    for row in cases:
        if not row.get('executed') or not row.get('before_seconds') or not row.get('after_seconds'):
            problems.append('incomplete measurements: '+str(row.get('action')))
        for field in ['before_seconds', 'after_seconds']:
            if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                   for value in row.get(field, [])):
                problems.append('invalid timing: '+str(row.get('action')))
        if row.get('parity') is not True:
            problems.append('output mismatch: '+str(row.get('action')))
        if row.get('material_regression'):
            try:
                reviews = json.loads((root/row['reviewed_evidence']).read_text())
                review = next(r for r in reviews if r['action'] == row['action'])
                supported = (review['status'] == 'resolved' and review.get('reason') and
                             review.get('evidence') and
                             all((root/path).is_file() for path in review['evidence']))
            except (KeyError, TypeError, ValueError, OSError, StopIteration):
                supported = False
            if not supported:
                problems.append('unreviewed regression: '+str(row.get('action')))
    if document.get('source_stable') is not True:
        problems.append('source or harness drift')
    functions = document.get('functions', [])
    if not functions or any(not row.get('disposition') or not row.get('coverage_note') for row in functions):
        problems.append('incomplete function inventory')
    decisions = document.get('residuals', [])
    if not decisions or any(row.get('status') == 'open' or not row.get('evidence') or
                            not (root/row['evidence']).is_file() for row in decisions):
        problems.append('open or unsupported residual decision')
    for path in document.get('receipts', []):
        if not (root/path).is_file():
            problems.append('missing receipt: '+path)
    if not document.get('receipts'):
        problems.append('no verification receipts')
    if document.get('focused_no_new_failures') is not True:
        problems.append('focused checks not verified')
    performance_verified = not problems
    if document.get('project_gate_passed') is not True:
        problems.append('required project gate has existing failures')
    return {'complete': not problems, 'performance_evidence_verified': performance_verified,
            'required_actions': len(required), 'problems': problems}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = check(json.loads(args.evidence.read_text()), args.evidence.parent)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if result['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
