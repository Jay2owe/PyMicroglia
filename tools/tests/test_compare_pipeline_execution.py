"""Run-label normalization cannot hide a different report or duplicate page."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from compare_pipeline_execution import inventory, metadata_references
from compare_action_execution import equivalent
from benchmark_pipeline_execution import digest


def receipt(root, names):
    outputs={}
    for name in names:
        target=root/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text('identity,value\n1,2\n')
        outputs[name]={'sha256':digest(target)}
    path=root/'receipt.json';path.write_text(json.dumps({'outputs':outputs}))
    return path


def test_only_actual_invocation_prefix_in_report_name_is_normalized(tmp_path):
    invocation='a'*32
    _,files=inventory(receipt(tmp_path,[f'outputs/{invocation}/cell-{invocation[:8]}-p1.csv',
                                      f'outputs/{invocation}/cell-12345678-p2.csv']))
    assert set(files)=={'outputs/cell-<invocation>-p1.csv','outputs/cell-12345678-p2.csv'}


def test_two_runs_cannot_collapse_into_one_comparison_page(tmp_path):
    names=[f'outputs/{c*32}/cell-{c*8}-p1.csv' for c in ['a','b']]
    with pytest.raises(ValueError,match='Ambiguous stage output'):
        inventory(receipt(tmp_path,names))


@pytest.mark.parametrize('changed', [False, True])
def test_confirmation_only_opening_time_is_volatile(tmp_path, changed):
    from pymicroglia.pipelines._contracts import content_id
    canonical=[]
    for index in range(2):
        root=tmp_path/str(index);root.mkdir()
        document={'consumption':{'opened_utc':f'2026-10-12T0{index}:00:00+00:00'},
                  'final_decisions':[{'p_value':.03 if changed and index else .02}],
                  'engine_sha256':'f'*64}
        document['confirmation_id']=content_id(document)
        name='outputs/science/confirmation/confirmation_record.json'
        target=root/name;target.parent.mkdir(parents=True);target.write_text(json.dumps(document))
        data={'outputs':{name:{'sha256':digest(target)}}}
        aliases,values=metadata_references(root/'receipt.json',data)
        assert aliases['confirmation_id'][document['confirmation_id']]==values[0]
        canonical.append(values)
    assert (canonical[0]==canonical[1]) is (not changed)


def test_report_alias_requires_retained_bytes_and_preserves_values(tmp_path):
    records=[]
    for label in ['a','b']:
        root=tmp_path/label;root.mkdir()
        svg=root/('report-'+label*12+'-p1.svg');svg.write_text('<svg/>')
        data={'outputs':{svg.name:{'sha256':digest(svg)}}}
        aliases,_=metadata_references(root/'receipt.json',data)
        table=root/'entries.csv'
        table.write_text('master,p_value\n'+svg.name+',0.02\n')
        records.append((table,aliases))
    a,b=records
    assert equivalent(a[0],b[0],a[0].parent,b[0].parent,(a[1],b[1]))[0]
    b[0].write_text(b[0].read_text().replace('0.02','0.03'))
    assert not equivalent(a[0],b[0],a[0].parent,b[0].parent,(a[1],b[1]))[0]
    svg.write_text('<svg>changed</svg>')
    with pytest.raises(ValueError,match='Report changed'):
        metadata_references(root/'receipt.json',data)
