"""Render every original timing group, keeping unmeasured totals visible.

The frozen scope supplies names/order and original estimates. A constituent
operation is never accepted as a measured whole-group replacement.
"""
import argparse
import json
import math
from pathlib import Path


def indexed(rows):
    result = {}
    for row in rows:
        key = row['id']
        if key in result:
            raise ValueError(f'Duplicate timing group: {key}')
        result[key] = row
    return result


def duration(seconds):
    if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError('Timing must be a finite nonnegative number')
    seconds = round(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f'{hours} h {minutes:02d} min'
    return f'{minutes} min {seconds:02d} s'


def render(scope, comparison, root):
    original = indexed(scope['groups'])
    current = indexed(comparison['groups'])
    if original.keys() != current.keys():
        raise ValueError(f'Timing scope changed: missing={sorted(original.keys() - current.keys())}; '
                         f'extra={sorted(current.keys() - original.keys())}')
    lines = ['# Complete original timing comparison', '',
             '476-frame recording. Experimental replay measurements; shared production is unchanged.', '',
             '| Stage group | Original before | Previously estimated after | Measured after |',
             '|---|---:|---:|---:|']
    notes = []
    for key, before in original.items():
        after = current[key]
        if after['after_status'] == 'measured':
            if after.get('complete_group') is not True or not after.get('receipt'):
                raise ValueError(f'Whole-group measurement lacks coverage or receipt: {key}')
            if not (root / after['receipt']).is_file():
                raise ValueError(f'Missing measurement receipt: {key}')
            measured = f"[{duration(after['after_seconds'])}]({after['receipt']})"
        elif after['after_status'] == 'unmeasured':
            if after.get('complete_group') or after.get('after_seconds') is not None:
                raise ValueError(f'Unmeasured group has a claimed complete timing: {key}')
            measured = 'Not yet measured'
        else:
            raise ValueError(f'Unknown measurement status: {key}')
        estimated = before.get('old_estimate_seconds')
        estimate_text = duration(estimated) + ' (estimate)' if estimated is not None else '—'
        original_text=duration(before['before_seconds'])
        if 'before_complete_seconds' in after:
            receipt=after.get('before_receipt')
            path=root/receipt if receipt else None
            if path is None or not path.is_file():raise ValueError(f'Missing complete before receipt: {key}')
            recorded=json.loads(path.read_text(encoding='utf-8'))['seconds']
            if recorded!=after['before_complete_seconds']:raise ValueError(f'Complete before timing differs from receipt: {key}')
            original_text=f"[{duration(recorded)}]({receipt})"
        lines.append(f"| {before['name']} | {original_text} | {estimate_text} | {measured} |")
        for note in (before.get('before_scope'), after.get('note')):
            if note:
                notes.append(f"- **{before['name']}:** {note}")
    lines += ['', 'Earlier estimates are retained for audit, not asserted as current measurements.', '', *notes, '',
              'No complete optimized 476-frame total is established by this table. '
              'Constituent measurements and the separate 99-frame trial cannot fill those missing totals.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', type=Path, required=True)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = render(json.loads(args.scope.read_text(encoding='utf-8')),
                    json.loads(args.comparison.read_text(encoding='utf-8')),
                    args.comparison.resolve().parent)
    args.output.write_text(report, encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
