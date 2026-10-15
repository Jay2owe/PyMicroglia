"""Isolated, source-verified runner for the complete accepted correction sequence.

The caller supplies evidence from the recording being analysed. No study input
paths, saved feedback, recording selectors or precomputed accepted labels occur
in the packaged recipe.
"""
from contextlib import contextmanager
import hashlib
import importlib
from importlib import resources
import json
from pathlib import Path
import sys
import zipfile

METHOD_VERSION = 'R218-A004-complete-corrections'


def extract(folder):
    archive = resources.files('pymicroglia.tracking').joinpath('data/completed_feedback_engine.zip')
    data = archive.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    root = Path(folder) / 'completed_feedback_engine'
    if root.exists():
        if (root/'archive.sha256').read_text().strip() != digest:
            raise ValueError('Correction engine differs from this immutable run')
    else:
        root.mkdir(parents=True)
        with zipfile.ZipFile(archive.open('rb')) as z:
            for name in z.namelist():
                if name.startswith('/') or '..' in Path(name).parts:
                    raise ValueError('Unsafe correction engine archive path')
            z.extractall(root)
        (root/'archive.sha256').write_text(digest+'\n')
    manifest = json.loads((root/'source_hashes.json').read_text())
    for name, expected in manifest['files_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Correction source changed: '+name)
    return root


@contextmanager
def methods(root, *, preserve_native=False):
    """Keep historical flat module names isolated from the native engine."""
    code = Path(root)/'code'
    names = {p.stem for p in code.glob('*.py')}
    if preserve_native:
        # Evidence contexts patch the tracker used by accepted_history. A
        # second copy of tracking would silently patch an unused module.
        for name in list(names):
            loaded = sys.modules.get(name)
            path = getattr(loaded,'__file__',None)
            if path and Path(path).is_file() and Path(path).read_bytes() == (code/(name+'.py')).read_bytes():
                names.remove(name)
    previous = {name:sys.modules.pop(name) for name in names if name in sys.modules}
    old_path = sys.path.copy()
    sys.path.insert(0,str(code))
    try:
        yield importlib.import_module('completed_feedback_sequence')
    finally:
        for name in names:
            sys.modules.pop(name,None)
        sys.modules.update(previous)
        sys.path[:] = old_path


def recipe(root):
    settings = json.loads((Path(root)/'settings.json').read_text())
    for operations in settings['operations'].values():
        for spec in operations:
            path = (Path(root)/spec['code_dir']).resolve()
            if not path.is_relative_to(Path(root).resolve()):
                raise ValueError('Correction source outside packaged engine')
            spec['code_dir'] = str(path)
    return settings


def apply(movies, folder, *, source_frame_offset, callback, inventory_callback):
    root = extract(folder)
    settings = recipe(root)
    with methods(root) as sequence:
        return sequence.apply(movies,settings['density_parameters'],settings['operations'],
            source_frame_offset,settings['revise_isolated_prior_reassignment'],callback,inventory_callback)


def run_current(root, stem, run_name, config_path, folder, *, resume=False):
    """Finish a fresh native run and publish the corrected file contract."""
    import time
    import numpy as np
    import tifffile
    import pipeline
    from common import Config, save_stack
    from .completed_evidence import generate
    from .correction_audit import CorrectionAudit

    output = Path(root)/'m23_completed_feedback'/run_name
    if output.exists():
        if not resume:
            raise FileExistsError(f'Immutable completed correction run already exists: {output}')
        receipt = output/'completion_files.json'
        if receipt.is_file():
            from .checkpoints import sha
            for relative, expected in json.loads(receipt.read_text()).items():
                path=(output/relative).resolve()
                if not path.is_relative_to(output.resolve()) or sha(path)!=expected:
                    raise ValueError('Completed correction checkpoint changed: '+relative)
            if json.loads((output/'run.json').read_text()).get('status')!='complete':
                raise ValueError('Correction checkpoint is incomplete')
            print('CHECKPOINT reused m23_completed_feedback',flush=True)
            return output
        output.rename(output.with_name(output.name+'_interrupted_'+str(time.time_ns())))
    (output/'out').mkdir(parents=True)
    started = time.perf_counter()
    cfg = Config.load(Path(config_path))
    data = generate(root,stem,run_name,config_path,folder)
    records = []
    inventories = []
    last = time.perf_counter()
    # Preserve a record of every ownership inference, including those later
    # restored to an earlier name by another accepted correction.
    audit = CorrectionAudit(data['labels'])
    inferred = audit.inferred
    def checkpoint(key,family,stage,labels,ledger,tables,arrays=None):
        nonlocal last
        statistics = audit.record(labels, ledger)
        directory = output/'mid/completed_feedback'/family/stage
        directory.mkdir(parents=True)
        for name,table in tables.items():
            if isinstance(table,dict):
                (directory/(name+'.json')).write_text(json.dumps(table,indent=2)+'\n')
            else:
                table.to_csv(directory/(name+'.csv'),index=False)
        records.append(dict(family=family,stage=stage,**statistics,
            seconds=time.perf_counter()-last))
        last = time.perf_counter()
    def inventory(stage,table):
        inventories.append(stage)
        directory = output/'mid/completed_feedback/inventory'
        directory.mkdir(parents=True,exist_ok=True)
        table.to_csv(directory/(stage+'.csv'),index=False)
    result = apply({stem:data},folder,source_frame_offset=cfg.source_frame_offset,
        callback=checkpoint,inventory_callback=inventory)
    # Publication only needs the original foreground and raw signal. Release
    # restoration/gap ancestors and detection evidence before writing outputs.
    original_labels, raw = data['labels'], data['raw']
    del data
    source_manifest=json.loads((Path(folder)/'completed_feedback_engine/source_hashes.json').read_text())
    actual=[(row['family'],row['stage']) for row in records]
    expected=[tuple(pair) for pair in source_manifest['required_boundaries']]
    if len(actual)!=len(expected) or set(actual)!=set(expected):
        raise RuntimeError('Incomplete accepted correction sequence; no final labels published')
    if sorted(inventories)!=source_manifest['required_inventories']:
        raise RuntimeError('Incomplete correction evidence inventory; no final labels published')
    labels,ledger = result[stem]
    del audit
    interval = cfg.values['frame_interval_min']
    save_stack(output/'out'/f'{stem}.tif',labels,interval)
    save_stack(output/'out'/f'{stem}_unclaimed_original_ids.tif',ledger,interval)
    native = Path(root)/'m22_accepted_history'/run_name
    provenance_path = native/'out'/f'{stem}_provenance.tif'
    if provenance_path.is_file():
        provenance = tifffile.imread(provenance_path)
        for t in range(len(labels)):
            provenance[t][inferred[t]] |= np.uint8(pipeline.PROVENANCE_INFERRED)
            added = (labels[t]>0)&(original_labels[t]==0)
            provenance[t][added] |= np.uint8(pipeline.PROVENANCE_ADDED)
        save_stack(output/'out'/f'{stem}_provenance.tif',provenance,interval)
        del provenance
    del original_labels
    frames,tracks = pipeline._identity_tables(labels,raw,inferred)
    frames.to_csv(output/'out/frame_identities.csv',index=False)
    tracks.to_csv(output/'out/tracks.csv',index=False)
    manifest = dict(status='complete',method_version=METHOD_VERSION,
        native_parent=str(native),source_frame_offset=cfg.source_frame_offset,
        correction_archive_sha256=(Path(folder)/'completed_feedback_engine/archive.sha256').read_text().strip(),
        stages=records,seconds=time.perf_counter()-started)
    (output/'run.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if resume:
        from .checkpoints import sha, write
        files={str(path.relative_to(output)):sha(path) for path in output.rglob('*') if path.is_file()}
        write(output/'completion_files.json',files)
    return output
