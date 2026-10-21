"""Hash a complete earlier-tracking run's outputs against the reference parity hashes."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--engine-output', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--run', default='', help='run folder between the stage and its out/mid/qc folders')
    a = p.parse_args()
    reference = json.loads(a.reference.read_text())
    rows, bad, missing = [], [], []
    for check in reference['checks']:
        parts = check['file'].replace(chr(92), '/').split('/')
        if a.run:
            parts.insert(1, a.run)
        path = a.engine_output / Path(*parts)
        if not path.is_file():
            missing.append(check['file'])
            continue
        digest = hashlib.sha256()
        with open(path, 'rb') as handle:
            for block in iter(lambda: handle.read(1 << 24), b''):
                digest.update(block)
        equal = digest.hexdigest() == check['before_sha256']
        rows.append(dict(file=check['file'], equal=equal, reference_sha256=check['before_sha256'],
                         candidate_sha256=digest.hexdigest()))
        if not equal:
            bad.append(check['file'])
    report = dict(status='pass' if rows and not bad and not missing else 'fail', compared=len(rows),
                  mismatched=bad, missing=missing, engine_output=str(a.engine_output), checks=rows)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(report['status'], len(rows), bad[:10], missing[:10])


if __name__ == '__main__':
    main()
