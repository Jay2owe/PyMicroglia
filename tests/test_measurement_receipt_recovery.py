import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

SPEC=importlib.util.spec_from_file_location('measurement_receipt_recovery',Path(__file__).parents[1]/'tools/recover_measurement_receipt.py')
recovery=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(recovery)


@pytest.fixture
def completed(tmp_path):
    api=recovery.receipt_api()
    sources=[]
    for name in ['producer.py','expanded_channels.py','expanded_outlines.py']:
        p=tmp_path/name;p.write_text('# captured source\n');sources.append(api.file_record(p))
    settings=tmp_path/'settings.json';settings.write_text('{"source_frame_offset":0}')
    raw=tmp_path/'raw.dat';raw.write_bytes(b'input')
    root=tmp_path/'photometry';m=root/'measurement';m.mkdir(parents=True)
    labels=m/'expanded_labels.tif';tifffile.imwrite(labels,np.ones((2,4,5),np.uint16),photometric='minisblack')
    table=m/'cell_frame.csv'
    pd.DataFrame([dict(identity=1,channel=c,source_frame_index=t) for t in range(2) for c in ['Red','Green']]).to_csv(table,index=False)
    manifest=dict(source_frames=2,labels_frames=2,identities=[1],channels=['Red','Green'],
                  code_sha256=sources[1]['sha256'],upstream_code_sha256=sources[2]['sha256'],
                  outputs=[dict(path=str(p),sha256=api.sha256(p)) for p in [labels,table]])
    (m/'manifest.json').write_text(json.dumps(manifest))
    run=root/'run.json';run.write_text(json.dumps(dict(status='complete',params={'source_frame_offset':0},
        result=manifest,outputs=dict(table=str(table),labels=str(labels),manifest=str(m/'manifest.json')),elapsed_seconds=5)))
    plan=dict(status='preview',project=str(tmp_path),run_file=str(run),command=['python',str(tmp_path/'producer.py')],
              code=sources,code_version=api.code_fingerprint(sources),inputs={'raw':api.file_record(raw)},
              parameters_file=api.file_record(settings),timeout_seconds=1)
    log=tmp_path/'failed.log';log.write_text('prior output\n'+json.dumps(plan,indent=2)+'\nrun receipt: producer timed out\n')
    return log,run,api


def test_recovery_verifies_completed_outputs_without_rerun(completed):
    log,run,api=completed
    before=run.read_bytes()
    receipt=recovery.recover(log,run,api)
    assert api.verify(receipt,strict=True)['ok']
    data=json.loads(receipt.read_text())
    assert data['recovery']['producer_exit_code'] is None
    assert data['started_at_utc'] is None
    assert before==run.read_bytes()
    with pytest.raises(ValueError,match='already exists'):
        recovery.recover(log,run,api)


def test_reject_changed_source(completed):
    log,run,api=completed
    (log.parent/'expanded_channels.py').write_text('# different source')
    with pytest.raises(ValueError,match='Captured input, source or settings changed'):
        recovery.recover(log,run,api)
    assert not (run.parent/api.RECEIPT).exists()


def test_reject_incomplete_table_even_with_current_output_hash(completed):
    log,run,api=completed
    table=run.parent/'measurement/cell_frame.csv';d=pd.read_csv(table);d.iloc[:-1].to_csv(table,index=False)
    data=json.loads(run.read_text());data['result']['outputs'][1]['sha256']=api.sha256(table)
    run.write_text(json.dumps(data));(table.parent/'manifest.json').write_text(json.dumps(data['result']))
    with pytest.raises(ValueError,match='complete declared'):
        recovery.recover(log,run,api)
    assert not (run.parent/api.RECEIPT).exists()


def test_missing_preflight_is_not_invented(completed):
    log,run,api=completed;log.write_text('failure without captured plan')
    with pytest.raises(ValueError,match='exactly one captured'):
        recovery.recover(log,run,api)
