"""Creation times are volatile; scientific values and engine identities are not."""
import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_action_execution import equivalent


def write(path, *, created, engine='accepted-engine', pvalue=.012):
    path.parent.mkdir(exist_ok=True)
    record={'p_value':pvalue,'provenance':{'generated_utc':created,'code_sha256':engine}}
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['identity','results_json'])
        writer.writeheader()
        writer.writerow({'identity':1,'results_json':json.dumps([record])})


@pytest.mark.parametrize('change,expected',[
    ({},True),
    ({'engine':'changed-engine'},False),
    ({'pvalue':.013},False),
    ({'created':None},False),
])
def test_only_valid_record_creation_time_is_volatile(tmp_path,change,expected):
    before=tmp_path/'before'/'cell_statistics.csv'
    after=tmp_path/'after'/'cell_statistics.csv'
    write(before,created='2026-10-11T12:00:00+00:00')
    options={'created':'2026-10-12T12:00:00+00:00',**change}
    write(after,**options)
    assert equivalent(before,after,before.parent,after.parent)[0] is expected


@pytest.mark.parametrize('change,expected', [('time',True),('vertex',False),('missing',False),('empty',False),('comment',False)])
def test_roi_archive_checks_actual_members_not_creation_time(tmp_path,change,expected):
    import zipfile
    from compare_action_execution import tables
    before,after=tmp_path/'before.zip',tmp_path/'after.zip'
    for path,year in [(before,2025),(after,2026)]:
        with zipfile.ZipFile(path,'w') as archive:
            if path==after and change=='comment':archive.comment=b'changed semantic annotation'
            if change=='empty' or (path==after and change=='missing'):continue
            member=zipfile.ZipInfo('cell.roi',date_time=(year,1,1,0,0,0))
            archive.writestr(member,b'roi-vertex-2' if path==after and change=='vertex' else b'roi-vertex-1')
    receipt=tmp_path/'receipt.json'
    receipt.write_text(json.dumps({'outputs':{'before.zip':{}}}))
    assert 'before.zip' in tables(receipt)[1]
    assert equivalent(before,after,tmp_path,tmp_path)[0] is expected
