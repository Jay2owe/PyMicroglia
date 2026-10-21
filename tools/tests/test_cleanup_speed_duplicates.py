"""Compact only authenticated task outputs, retaining exact recovery bytes."""
import gzip
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cleanup_speed_duplicates import compact_large_tables,current_record


def fixture(base, *, checksum=None, name='result.csv'):
    trial=base/'trial-before'/'action'
    trial.mkdir(parents=True)
    source=trial/name
    source.write_bytes(b'1,2,3\n'*(2*1024*1024))
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    (trial/'receipt.json').write_text(json.dumps({'source_before':{'test':'fixture'},
        'outputs':{name:{'sha256':checksum or digest}}}))
    return source,digest


def test_compaction_has_exact_recovery_and_never_removes_inputs(tmp_path):
    source,digest=fixture(tmp_path)
    inputs=source.parent/'inputs.csv';inputs.write_bytes(source.read_bytes())
    expected=source.read_bytes()
    result=compact_large_tables(tmp_path)
    assert result['retained_tables']==1 and not source.exists() and inputs.exists()
    journal=json.loads(Path(result['recovery']).read_text())
    row=journal['files'][0]
    restored=gzip.decompress((tmp_path/row['archive']).read_bytes())
    assert restored==expected and hashlib.sha256(restored).hexdigest()==digest
    assert row['status']=='compacted'
    assert compact_large_tables(tmp_path)['retained_tables']==1


def test_modified_output_is_preserved_and_refused(tmp_path):
    source,_=fixture(tmp_path,checksum='0'*64)
    with pytest.raises(ValueError,match='changed'):compact_large_tables(tmp_path)
    assert source.exists()


def test_receipt_cannot_select_an_output_outside_its_trial(tmp_path):
    source,_=fixture(tmp_path)
    receipt=source.parent/'receipt.json'
    receipt.write_text(json.dumps({'source_before':{},'outputs':{'../../outside.csv':{'sha256':'0'*64}}}))
    with pytest.raises(ValueError,match='Unsafe'):compact_large_tables(tmp_path)
    assert source.exists()


def test_shared_manifest_uses_a_matching_verified_revision_and_refuses_unknown_changes(tmp_path):
    source=tmp_path/'manifest.json'
    source.write_bytes(b'after legitimate later action')
    accepted={'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'stage':'later'}
    old={'sha256':hashlib.sha256(b'earlier action').hexdigest(),'stage':'earlier'}
    assert current_record(source,[accepted,old])==accepted
    source.write_bytes(b'unrecorded edit')
    with pytest.raises(ValueError,match='changed after verification'):
        current_record(source,[accepted,old])
    assert source.read_bytes()==b'unrecorded edit'
