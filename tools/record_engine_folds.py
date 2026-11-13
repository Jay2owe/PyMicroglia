"""Refresh a shipped engine's source_hashes.json after speed patches are folded into it (release 2).

Recomputes every file digest and appends one entry per folded patch to the
manifest's "folded" list, so the engine records which PyMicroglia patch each
change came from and the A1 replay that accepted it.

    python tools/record_engine_folds.py --engine motion_engine --batch fold1 \
        --fold bounded_marker_geometry=code/bounded_owner_excursion.py:_marker [--fold ...]
"""
import argparse
import hashlib
import json
from pathlib import Path

from pymicroglia.tracking import shipped_engines


def record(engine, batch, folds):
    root = shipped_engines.folder(engine)
    manifest_path = root / shipped_engines.MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    digests = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
               for name in shipped_engines.names(root) if name != shipped_engines.MANIFEST}
    # Keep the recorded order, so the diff shows only changed digests; new files go last.
    order = [name for name in manifest.get('files_sha256', {}) if name in digests]
    manifest['files_sha256'] = {name: digests[name] for name in order + sorted(set(digests) - set(order))}
    known = {entry['patch'] for entry in manifest.get('folded', [])}
    for patch, target in folds:
        if 'pymicroglia/tracking/%s.py' % patch in known:
            raise SystemExit(patch + ' is already recorded as folded')
        path, _, function = target.partition(':')
        if path not in manifest['files_sha256']:
            raise SystemExit(path + ' is not an engine file')
        manifest.setdefault('folded', []).append(dict(patch='pymicroglia/tracking/%s.py' % patch, into=path,
                                                      function=function, batch=batch))
    manifest_path.write_bytes((json.dumps(manifest, indent=2) + '\n').encode())
    shipped_engines.verify(root)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--engine', required=True, choices=(shipped_engines.MOTION, shipped_engines.COMPLETED_FEEDBACK))
    parser.add_argument('--batch', required=True)
    parser.add_argument('--fold', action='append', default=[], metavar='PATCH=FILE:FUNCTION')
    args = parser.parse_args()
    folds = [item.split('=', 1) for item in args.fold]
    manifest = record(args.engine, args.batch, folds)
    print(json.dumps(dict(engine=args.engine, files=len(manifest['files_sha256']),
                          folded=len(manifest.get('folded', [])),
                          fingerprint=shipped_engines.fingerprint(shipped_engines.folder(args.engine)))))


if __name__ == '__main__':
    main()
