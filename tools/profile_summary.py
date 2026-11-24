"""Summarise a cProfile dump as text and JSON, for machines whose binary files cannot be fetched.

python profile_summary.py <profile.pstats> <output stem> [count]

Writes <stem>.txt (top functions by own and by inclusive time, with their
heaviest callers) and <stem>.json (every function's calls, own and inclusive
seconds and callers), so the profile can be read anywhere.
"""
import json
import pstats
import re
import sys


def name(key):
    path, line, function = key
    return '%s:%d %s' % (re.split(r'[\\/]', path)[-1], line, function)


def main():
    source, stem = sys.argv[1], sys.argv[2]
    count = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    raw = pstats.Stats(source).stats
    total = sum(row[2] for row in raw.values())
    lines = ['total own seconds %.1f' % total, '-- own time']
    for key, (cc, nc, tt, ct, callers) in sorted(raw.items(), key=lambda kv: -kv[1][2])[:count]:
        heavy = sorted(callers.items(), key=lambda kv: -kv[1][2])[:3]
        lines.append('%8.2f s %5.1f%% %9d calls  %s  <- %s' % (
            tt, 100 * tt / total, nc, name(key), '; '.join('%s %.1fs' % (name(c), v[2]) for c, v in heavy)))
    lines.append('-- inclusive time (project code)')
    skip = re.compile(r'site-packages|[\\/]lib[\\/]|<frozen|contextlib|threading|benchmark_', re.I)
    rows = [(key, row) for key, row in raw.items() if key[0] != '~' and not skip.search(key[0])]
    for key, (cc, nc, tt, ct, callers) in sorted(rows, key=lambda kv: -kv[1][3])[:count]:
        lines.append('%8.2f s %8.2f own %9d calls  %s' % (ct, tt, nc, name(key)))
    with open(stem + '.txt', 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')
    table = [dict(function=name(key), file=key[0], calls=nc, own=round(tt, 4), inclusive=round(ct, 4),
                  callers={name(c): [v[1], round(v[2], 4), round(v[3], 4)] for c, v in callers.items()})
             for key, (cc, nc, tt, ct, callers) in raw.items() if ct >= 0.05]
    with open(stem + '.json', 'w', encoding='utf-8') as handle:
        json.dump(dict(total_own_seconds=total, functions=table), handle)
    print('\n'.join(lines[:30]))


if __name__ == '__main__':
    main()
