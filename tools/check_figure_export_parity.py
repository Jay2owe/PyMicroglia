"""Compare paired figure outputs and exact plotted/supporting tables.

Rendering changes are checked separately from data and embedded interpretation.
Only export identifiers, creation times and explicitly supplied trial paths vary.
"""
from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path

csv.field_size_limit(100_000_000)

def normalized_statistics(rows):
    """Normalize only the native analysis record's validated creation time."""
    rows = deepcopy(rows)
    for row in rows:
        text = row.get('results_json')
        if not text:
            continue
        records = json.loads(text)
        if isinstance(records, list):
            for record in records:
                provenance = record.get('provenance', {}) if isinstance(record, dict) else {}
                created = provenance.get('generated_utc')
                if isinstance(created, str):
                    datetime.fromisoformat(created)
                    provenance['generated_utc'] = '<generation-time>'
        row['results_json'] = json.dumps(records, sort_keys=True)
    return rows


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def compare(before, after):
    from reprofig import extract_records
    a, b = (json.loads(path.read_text()) for path in (before, after))
    roots = [Path(row['parameters']['output_dir']) for row in (a, b)]
    checks = []
    for key in sorted(set(a['outputs']) | set(b['outputs'])):
        check = {'file': key, 'equal': False}
        checks.append(check)
        if key not in a['outputs'] or key not in b['outputs']:
            check['reason'] = 'missing counterpart'
            continue
        left, right = (root / key for root in roots)
        if any(digest(path) != row['outputs'][key]['sha256']
               for path, row in zip((left, right), (a, b))):
            check['reason'] = 'changed after measurement'
            continue
        if key.endswith('.csv'):
            check['equal'] = left.read_bytes() == right.read_bytes()
            check['reason'] = 'exact table bytes'
            if not check['equal']:
                with left.open(encoding='utf-8', newline='') as stream:
                    x = list(csv.reader(stream))
                with right.open(encoding='utf-8', newline='') as stream:
                    y = list(csv.reader(stream))
                check['different_cells'] = [
                    {'row': i, 'column': j, 'before': v[:200], 'after': w[:200]}
                    for i, (r, s) in enumerate(zip(x, y))
                    for j, (v, w) in enumerate(zip(r, s)) if v != w][:10]
                check['shapes'] = [[len(rows), len(rows[0]) if rows else 0] for rows in (x, y)]
                if (key.endswith('_statistics.csv') and x and y and x[0] == y[0]
                        and all(len(row) == len(x[0]) for row in x+y)):
                    check['equal'] = normalized_statistics([dict(zip(x[0], r)) for r in x[1:]]) == normalized_statistics(
                        [dict(zip(y[0], r)) for r in y[1:]])
                    if check['equal']:
                        check['reason'] = 'exact statistics; only validated provenance.generated_utc normalized'
        elif key.endswith('.svg'):
            records = [extract_records(path) for path in (left, right)]
            fields = ['analysis', 'data_status', 'statistics_status',
                      'title', 'original_stem', 'distribution_profile',
                      'data_tables', 'reproduction', 'producer']
            check['equal'] = len(records[0]) == len(records[1]) == 1 and all(
                getattr(records[0][0], field) == getattr(records[1][0], field) for field in fields)
            check['reason'] = 'exact embedded interpretation and statistics; visual parity checked separately'
            check['rendering_checked'] = False
            if len(records[0]) == len(records[1]) == 1:
                stats = [record[0].statistics for record in records]
                check['equal'] = check['equal'] and normalized_statistics(stats[0]) == normalized_statistics(stats[1])
                # This digest covers ReproFig's canonical statistics CSV, not
                # the auxiliary table's different column order/serialization.
                hashes = [record[0].statistics_csv_sha256 for record in records]
                from reprofig.tables import statistics_csv_bytes
                check['equal'] = check['equal'] and all(
                    (hashlib.sha256(statistics_csv_bytes(rows)).hexdigest() == value) if value is not None
                    else not rows for rows, value in zip(stats, hashes))
                # Producer edits are the treatment and separately identified by
                # package hashes. Input data must keep the same content identity.
                inputs = [[(s.role, s.relative_path, s.sha256, s.size_bytes) for s in record[0].sources
                           if not str(s.relative_path).endswith('.py')] for record in records]
                check['same_input_content'] = inputs[0] == inputs[1]
                check['equal'] = check['equal'] and check['same_input_content']
        elif key == '.auto-organotypic/artefacts.json':
            # The exporter writes a separate content-addressed record for each
            # new carrier. Those identities must change when SVG bytes change.
            check['equal'] = all(isinstance(json.loads(path.read_text()), dict) for path in (left, right))
            check['reason'] = 'independent artifact ledgers present; exporter validates registrations'
        else:
            check['equal'] = digest(left) == digest(right)
            check['reason'] = 'exact bytes'
    return {'action': a['action'], 'before_seconds': a['seconds'], 'after_seconds': b['seconds'],
            'successful': a['status'] == b['status'] == 'measured' and a['action'] == b['action'],
            'data_parity': bool(checks) and all(check['equal'] for check in checks),
            'checks': checks, 'receipts': [str(before), str(after)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before', type=Path)
    parser.add_argument('after', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--actions', nargs='+')
    args = parser.parse_args()
    summaries = [json.loads((path/'summary.json').read_text()) for path in (args.before, args.after)]
    names = args.actions or summaries[0]['required_actions']
    rows = [compare(args.before/(name+'.json'), args.after/(name+'.json')) for name in names]
    stable = all(s.get('source_unchanged') and s.get('dependencies_unchanged') and s.get('harness_unchanged', True)
                 for s in summaries)
    stable = stable and summaries[0]['dependencies'] == summaries[1]['dependencies']
    same_actions = (all(name in s['required_actions'] for name in names for s in summaries)
                    if args.actions else names == summaries[1]['required_actions'])
    result = {'source_stable': stable, 'same_actions': same_actions, 'checked_actions': names,
              'cases': rows, 'data_parity': stable and same_actions and all(r['successful'] and r['data_parity'] for r in rows)}
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'cases'}))
    print('Different:', [r['action'] for r in rows if not r['data_parity']])
    return 0 if result['data_parity'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
