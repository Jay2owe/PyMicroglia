"""Replay correction evidence up to one final stage and compare its fingerprints.

Runs completed_evidence.generate from read-only native checkpoints with the
final operations cut after --stage, fingerprints the evidence as
benchmark_completed_evidence does, and requires every fingerprint present in
both runs to equal the reference (a complete evidence_hashes.json), with the
stage's own keys present. Tracker batch counters are reported.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time


def main():
    p = argparse.ArgumentParser()
    for name in ('engine', 'snapshot', 'config', 'output', 'reference_evidence'):
        p.add_argument('--' + name.replace('_', '-'), dest=name, type=Path, required=True)
    p.add_argument('--run', required=True)
    p.add_argument('--stem', required=True)
    p.add_argument('--stage', required=True)
    a = p.parse_args()
    for name in ('engine', 'snapshot', 'config', 'output', 'reference_evidence'):
        setattr(a, name, getattr(a, name).resolve())
    platform.platform = lambda *x, **k: 'Windows-complete-evidence-replay'
    import numpy as np
    import pandas as pd
    import psutil
    if psutil.virtual_memory().available < 3 * 1024**3:
        raise RuntimeError('Need 3 GiB free memory')
    a.output.mkdir(exist_ok=False)
    sys.path[:0] = [str(a.snapshot), str(a.engine / 'code')]
    from pymicroglia.tracking import completed_evidence, correction_storage, tracking_feature_batch
    from pymicroglia.tracking.correction_audit import array_sha256
    from benchmark_working_storage import load
    load(a.snapshot / 'array_storage.py')
    import optimized_pipeline
    original_recipe = completed_evidence.recipe

    def recipe(corrections):
        settings = original_recipe(corrections)
        final = settings['operations']['final']
        cut = next(i for i, spec in enumerate(final) if spec['stage'] == a.stage)
        settings['operations']['final'] = final[:cut + 1]
        return settings
    completed_evidence.recipe = recipe
    counters = []
    original_install = tracking_feature_batch.install

    def install(module):
        context = original_install(module)

        class Held:
            def __enter__(self):
                stats = context.__enter__()
                counters.append(stats)
                return stats

            def __exit__(self, *exc):
                return context.__exit__(*exc)
        return Held()
    tracking_feature_batch.install = install
    hashes = {}

    def fingerprint(value, key):
        if isinstance(value, np.ndarray):
            hashes[key] = dict(shape=list(value.shape), dtype=value.dtype.str, sha256=array_sha256(value))
        elif isinstance(value, pd.DataFrame):
            hashes[key] = dict(table_sha256=hashlib.sha256(value.to_csv(index=False).encode()).hexdigest())
        elif isinstance(value, dict):
            for name, child in value.items():
                fingerprint(child, key + '/' + str(name))
        elif isinstance(value, (list, tuple)):
            for i, child in enumerate(value):
                fingerprint(child, key + '/' + str(i))
        elif isinstance(value, np.generic):
            hashes[key] = value.item()
        else:
            hashes[key] = value
    timings = {}

    def execute():
        with correction_storage.bounded_corrections(a.output, minimum_bytes=1024**2) as pool:
            tick = time.perf_counter()
            data = completed_evidence.generate(a.engine, a.stem, a.run, a.config, a.output, array_pool=pool)
            timings['evidence_seconds'] = time.perf_counter() - tick
            fingerprint(data, 'evidence')
    optimized_pipeline.main(['--no-evidence-cache'], runner=execute)
    expected = json.loads(a.reference_evidence.read_text())
    stage_prefix = f'evidence/final_evidence/{a.stage}/'
    stage_keys = [k for k in expected if k.startswith(stage_prefix)]
    shared = [k for k in hashes if k in expected]
    differences = [k for k in shared if hashes[k] != expected[k]]
    missing = [k for k in stage_keys if k not in hashes]
    timing_file = a.output / f'{a.stem}_correction_evidence.json'
    stages = json.loads(timing_file.read_text())['stages'] if timing_file.is_file() else None
    report = dict(status='pass' if stage_keys and not differences and not missing else 'fail',
                  stage=a.stage, compared=len(shared), stage_keys=len(stage_keys),
                  differences=differences[:20], missing=missing[:20], timings=timings, stages=stages,
                  counters=[{k: v for k, v in c.items() if isinstance(v, (int, float, bool))} for c in counters])
    (a.output / 'check.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps(dict(status=report['status'], compared=len(shared), stage_keys=len(stage_keys),
                          differences=len(differences), missing=len(missing), timings=timings,
                          stages=[(s['stage'], round(s['seconds'])) for s in stages or []])), flush=True)
    if report['status'] != 'pass':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
