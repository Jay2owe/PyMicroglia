"""Verify completed measurement outputs using a captured pre-execution plan.

This does not rerun the producer or infer historical code from current files.
The original timeout remains part of the receipt's recovery evidence.
"""
import argparse
import importlib.util
import json
from pathlib import Path


def receipt_api():
    path = Path.home()/'.claude/skills/run-receipt/scripts/run_receipt.py'
    spec = importlib.util.spec_from_file_location('shared_run_receipt', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preview_from_log(log, run_file):
    decoder = json.JSONDecoder()
    text = Path(log).read_text(encoding='utf8')
    candidates = []
    for offset, char in enumerate(text):
        if char != '{' or (offset and text[offset-1] != '\n'):
            continue
        try:
            item, _ = decoder.raw_decode(text[offset:])
        except json.JSONDecodeError:
            continue
        if (isinstance(item, dict) and item.get('status') == 'preview'
                and item.get('run_file') and Path(item['run_file']).resolve() == Path(run_file).resolve()):
            candidates.append(item)
    if len(candidates) != 1:
        raise ValueError('Require exactly one captured pre-execution plan for this run')
    return candidates[0]


def verify_measurement(run_file, api):
    import numpy as np
    import pandas as pd
    import tifffile
    run = json.loads(run_file.read_text())
    if run.get('status') not in api.DONE:
        raise ValueError('Measurement producer did not finish')
    manifest_path = run_file.parent/'measurement/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest != run.get('result'):
        raise ValueError('Measurement manifest differs from completed producer record')
    for row in manifest['outputs']:
        if api.sha256(Path(row['path'])) != row['sha256']:
            raise ValueError('Measurement output changed: '+row['path'])
    table = pd.read_csv(run_file.parent/'measurement/cell_frame.csv',
                        usecols=['identity','channel','source_frame_index'])
    count = manifest['source_frames']
    if (len(table) != count*len(manifest['identities'])*len(manifest['channels'])
            or table.duplicated(['identity','channel','source_frame_index']).any()
            or set(table.source_frame_index) != set(range(count))
            or set(table.identity) != set(manifest['identities'])
            or set(table.channel) != set(manifest['channels'])):
        raise ValueError('Measurement table does not cover the complete declared frame/cell/channel grid')
    with tifffile.TiffFile(run_file.parent/'measurement/expanded_labels.tif') as movie:
        if len(movie.pages) != manifest['labels_frames']:
            raise ValueError('Expanded outline movie is incomplete')
        # Hashes cover all bytes; decoding both ends also rejects a broken final page.
        if movie.pages[0].asarray().shape != movie.pages[-1].asarray().shape:
            raise ValueError('Expanded outline movie changes shape')
    return run, manifest


def recover(log, run_file, api=None):
    api = api or receipt_api()
    log, run_file = Path(log).resolve(), Path(run_file).resolve()
    plan = preview_from_log(log, run_file)
    project = Path(plan['project']).resolve()
    if not run_file.is_relative_to(project):
        raise ValueError('Recovered run must be inside its declared project')
    receipt_path = run_file.parent/api.RECEIPT
    if receipt_path.exists():
        raise ValueError('Receipt already exists; verify it instead')
    records = [*plan['inputs'].values(), *plan['code'], plan['parameters_file']]
    for record in records:
        if not record or api.file_record(Path(record['path'])) != record:
            raise ValueError('Captured input, source or settings changed: '+str(record))
    if api.code_fingerprint(plan['code']) != plan['code_version']:
        raise ValueError('Captured code fingerprint is inconsistent')
    run, manifest = verify_measurement(run_file, api)
    settings = json.loads(Path(plan['parameters_file']['path']).read_text())
    if run.get('params') != settings:
        raise ValueError('Producer did not confirm the captured settings')
    source_hashes = {Path(r['path']).name:r['sha256'] for r in plan['code']}
    if (manifest['code_sha256'] != source_hashes['expanded_channels.py']
            or manifest['upstream_code_sha256'] != source_hashes['expanded_outlines.py']):
        raise ValueError('Measurement producer hashes differ from the pre-execution plan')
    receipt = dict(schema_version=1, status='complete', project=str(project),
                   run_file=str(run_file), run_sha256=api.sha256(run_file),
                   started_at_utc=None, finished_at_utc=None,
                   command=plan['command'], timeout_seconds=plan['timeout_seconds'],
                   extended_timeout_reason=plan.get('extended_timeout_reason',''),
                   inputs=plan['inputs'], input_scope='run', producer=plan['code'][0],
                   code=plan['code'], code_version=plan['code_version'], code_roots=[],
                   parameters=settings, parameter_coverage='explicit-settings-file',
                   parameters_file=plan['parameters_file'],
                   outputs=api.outputs_from_run(project,run_file,run), gaps=[],
                   recovery=dict(verified_at_utc=api.stamp(), preflight_log=api.file_record(log),
                                 reason='Wrapper timed out; producer completed and saved all declared outputs',
                                 producer_exit_code=None, producer_elapsed_seconds=run.get('elapsed_seconds'),
                                 verification='Captured pre-execution hashes, producer output hashes, complete frame/cell/channel grid; no recomputation'))
    api._atomic_new_json(receipt_path, receipt)
    checked = api.verify(receipt_path, strict=True)
    if not checked['ok']:
        raise ValueError(checked)
    return receipt_path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--log',type=Path,required=True)
    parser.add_argument('--run-file',type=Path,required=True)
    args=parser.parse_args()
    print(recover(args.log,args.run_file),flush=True)


if __name__=='__main__':
    main()
