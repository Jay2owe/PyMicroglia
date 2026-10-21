"""Capture exact gap-operation evidence without replaying prior corrections.

Only succeeds when saved audit hashes prove the final movie is identical to
the target operation's parent. This is a diagnostic fixture, not a rule input.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import platform
import shutil
import sys
import time


def restore_original(manifest, output):
    """Reconstruct the sequence's initial restored input, proved by its audit."""
    import numpy as np
    import pandas as pd
    import tifffile
    from pymicroglia.tracking import completed_feedback
    from pymicroglia.tracking.correction_audit import array_sha256
    output=Path(output)
    native=Path(manifest.get('original_native',manifest['original']))
    receipt=json.loads((Path(manifest['parent']).parent.parent/'run.json').read_text())
    expected=next(r['label_array_sha256'] for r in receipt['stages']
                  if r['family']=='initial' and r['stage']=='lineage_union')
    settings=completed_feedback.recipe(Path(manifest['correction_root']))
    with completed_feedback.methods(Path(manifest['correction_root'])) as sequence:
        restored,_,_=sequence.initial.apply(tifffile.imread(native),
            pd.read_csv(manifest['points'],low_memory=False),settings['density_parameters'])
    assert array_sha256(restored)==expected, 'Initial restoration differs from saved audit'
    target=output/'original_restored.npy'
    if target.exists():
        assert array_sha256(np.load(target,allow_pickle=False))==expected
    else:
        np.save(target,restored,allow_pickle=False)
    manifest.update(original_native=str(native),original=str(target.resolve()),
        restored_original_verified=True,restored_original_sha256=expected)


def main():
    p=argparse.ArgumentParser()
    for name in ('source','engine','config','output'):
        p.add_argument('--'+name,type=Path,required=True)
    for name in ('run','stem','stage'):
        p.add_argument('--'+name,required=True)
    a=p.parse_args()
    platform.platform=lambda *a,**k:'Windows-gap-memory-fixture'
    import psutil
    import numpy as np
    import tifffile
    if psutil.virtual_memory().available < 3*1024**3 or shutil.disk_usage(a.output.parent).free < 4*1024**3:
        raise RuntimeError('Need 3 GiB free RAM and 4 GiB free disk for evidence capture')
    sys.path[:0]=[str(a.source.resolve()),str((a.engine/'code').resolve())]
    from pymicroglia.tracking import completed_evidence,completed_feedback,detector_acceleration
    from pymicroglia.tracking.correction_audit import array_sha256
    assert Path(completed_evidence.__file__).resolve().is_relative_to(a.source.resolve())
    result=a.engine/'m23_completed_feedback'/a.run
    receipt=json.loads((result/'run.json').read_text())
    records=receipt['stages']
    index=next(i for i,r in enumerate(records) if r['family']=='final' and r['stage']==a.stage)
    expected_parent=records[index-1]['label_array_sha256']
    parent_path=result/'out'/f'{a.stem}.tif'
    parent=tifffile.imread(parent_path)
    if array_sha256(parent)!=expected_parent:
        raise ValueError('Final labels cannot reproduce this operation parent exactly')
    del parent
    corrections=completed_feedback.extract(a.engine.parent)
    settings=completed_feedback.recipe(corrections)
    spec=next(s for s in settings['operations']['final'] if s['stage']==a.stage)
    assert spec['kind']=='gap'
    a.output.mkdir(exist_ok=False)
    # Reuse the verified scientific archive; never create another extraction or
    # overwrite the completed run's evidence receipt.
    completed_evidence.extract=lambda folder:corrections
    optimized=importlib.import_module('optimized_pipeline')
    detector=importlib.import_module('raw_physical_hypotheses')
    capture={}
    def run():
        with detector_acceleration.install(detector):
            capture['data']=completed_evidence.generate(a.engine,a.stem,a.run,a.config,a.output)
    start=time.perf_counter()
    optimized.main(['--no-evidence-cache'],runner=run)
    data=capture.pop('data')
    chosen=data['final_evidence'][a.stage]
    traces=[]
    for number,(before,after,links) in enumerate(chosen['traces']):
        entry={}
        for role,values in (('before',before),('after',after)):
            path=a.output/f'trace_{number}_{role}.npy'
            np.save(path,values,allow_pickle=False)
            entry[role]=str(path.resolve())
        path=a.output/f'trace_{number}_links.pkl'
        links.to_pickle(path)
        entry['links']=str(path.resolve())
        traces.append(entry)
    native=a.engine/'m22_accepted_history'/a.run
    points_path=native/'mid/accepted_history/26_latent_body_tracks/out/latent_track_points.csv'
    manifest=dict(status='captured',stage=a.stage,stem=a.stem,
        evidence_seconds=time.perf_counter()-start,parent=str(parent_path.resolve()),
        parent_sha256=expected_parent,expected_label_sha256=records[index]['label_array_sha256'],
        original=str((native/'out'/f'{a.stem}.tif').resolve()),points=str(points_path.resolve()),
        expected_audit=str((result/'mid/completed_feedback/final'/a.stage/'source_branch_audit.csv').resolve()),
        source_frame_offset=receipt['source_frame_offset'],traces=traces,spec=spec,
        correction_root=str(corrections.resolve()),source=str(a.source.resolve()),
        config=str(a.config.resolve()),scope='Owned diagnostic fixture with hash-proved parent; not production selectors')
    restore_original(manifest,a.output)
    def sha(path):
        with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()
    files=[manifest[k] for k in ('parent','original','original_native','points','expected_audit','config')]
    files.extend(value for trace in traces for value in trace.values())
    manifest['files']={str(path):sha(path) for path in files}
    manifest['sources']={str(path.resolve()):sha(path) for path in a.source.rglob('*.py')}
    manifest['sources'][str(Path(__file__).resolve())]=sha(__file__)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(status='captured',evidence_seconds=manifest['evidence_seconds'],traces=len(traces),parent_hash_proved=True)),flush=True)


if __name__=='__main__':
    main()
