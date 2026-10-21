"""Find identical recorded fit inputs across the completed export catalogue.

This is diagnostic evidence about existing records, not a cache implementation.
The complete engine identity and requested arguments must agree.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from ai_research_assistant.tracking import RunTracker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipts', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    csv.field_size_limit(100_000_000)
    groups = {}
    scanned = []
    missing_records = 0
    paths = [p for p in args.receipts.glob('*.json') if p.name != 'summary.json']
    with RunTracker('Check repeated scientific work in figure exports', project='PyMicroglia') as tracker:
        with tracker.stage('Inspect saved fit records', total=len(paths)):
            for path in paths:
                receipt = json.loads(path.read_text())
                with tracker.item(receipt['action']):
                    seen = set()
                    for filename in receipt.get('outputs', {}):
                        if not filename.endswith('.csv'):
                            continue
                        table = Path(receipt['parameters']['output_dir'])/filename
                        with table.open(encoding='utf-8', newline='') as stream:
                            reader = csv.DictReader(stream)
                            columns = [c for c in reader.fieldnames or [] if c.endswith('run_record_json')]
                            if not columns:
                                continue
                            scanned.append(str(table))
                            for row in reader:
                                for column in columns:
                                    value = row[column]
                                    if value in seen:
                                        continue
                                    seen.add(value)
                                    if value.strip() in {'', 'nan', 'NaN', 'None', 'null'}:
                                        missing_records += 1
                                        continue
                                    record = json.loads(value)
                                    if record.get('action') != 'estimate_period':
                                        continue
                                    identity = json.dumps({key: record[key] for key in ['action', 'inputs', 'environment']},
                                                          sort_keys=True, separators=(',', ':'))
                                    key = hashlib.sha256(identity.encode()).hexdigest()
                                    group = groups.setdefault(key, {'actions': set(), 'recording_hashes': [
                                        r['sha256'] for r in record['recordings']]})
                                    group['actions'].add(receipt['action'])
    rows = [{'identity': key, **group, 'actions': sorted(group['actions'])} for key, group in groups.items()]
    args.output.write_text(json.dumps({'unique_calls': len(rows), 'repeated_across_figures': [
        row for row in rows if len(row['actions']) > 1], 'calls': rows, 'tables': scanned,
        'missing_record_values': missing_records,
        'limit': 'Only exported estimate_period run records are visible; hidden surrogate fits are not counted.'},
        indent=2), encoding='utf-8')
    print('Unique recorded calls:', len(rows), 'shared across figures:', sum(len(r['actions']) > 1 for r in rows))


if __name__ == '__main__':
    main()
