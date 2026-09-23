"""Single-copy tables retain nested evidence, types and exact numeric values."""
import json
import numpy as np
import pandas as pd
import pytest
from auto_organotypic import store,layout
from pymicroglia.pipelines._screening import write_table,read_table,file_hash


def test_portable_report_copy_retains_table_schema(tmp_path):
    from pymicroglia._results import copy_artifact
    source=write_table(tmp_path/'cache'/'evidence.json',
        pd.DataFrame({'nested':[{'coefficient':[1.,None]}], 'cell':['001']}))
    target=tmp_path/'report'/'evidence.csv'
    copy_artifact(source,target)
    assert file_hash(source)==file_hash(target)
    assert source.read_bytes()==target.read_bytes()
    pd.testing.assert_frame_equal(read_table(source),read_table(target))


def test_single_csv_roundtrip_and_type_tampering(tmp_path):
    frame=pd.DataFrame({'float':[np.nextafter(1.,2.),np.nan,-0.], 'text':['001','',None], 'nested':[{'a':[True,1,None]},[1.,2.],None],'boolean':pd.Series([True,False,None],dtype='boolean'),'integer':pd.Series([1,None,3],dtype='Int64')})
    path=write_table(tmp_path/'evidence.json',frame)
    assert path.name=='evidence.csv'
    assert not list(tmp_path.glob('*.json'))
    assert not list(tmp_path.rglob('evidence.json'))
    result=read_table(path)
    pd.testing.assert_frame_equal(result,frame)
    assert result['float'].to_numpy().tobytes()==frame['float'].to_numpy().tobytes()
    before=file_hash(path)
    entry=store.ledger.entry_for(path);entry['extra']['pipeline_table']['dtypes']['float']='object'
    store.ledger.record(path,entry)
    assert file_hash(path)!=before


def test_empty_typed_table(tmp_path):
    original=pd.DataFrame({'identity':pd.Series(dtype='int64'),'diagnostic':pd.Series(dtype='object')})
    path=write_table(tmp_path/'empty.json',original)
    pd.testing.assert_frame_equal(read_table(path),original)


def test_legacy_table_is_read_without_rewriting(tmp_path):
    path=tmp_path/'old.json';path.write_text(json.dumps(dict(schema_version=1,columns=['x'],dtypes={'x':'float64'},records=[{'x':1.25}])))
    before=path.read_bytes();assert read_table(path).x.tolist()==[1.25];assert path.read_bytes()==before


def test_old_table_artifact_name_resolves_only_a_typed_table(tmp_path):
    import pytest
    import pandas as pd
    from pymicroglia.pipelines._screening import write_table,read_table,file_hash
    from pymicroglia.pipelines._contracts import ArtifactRef,StepResult
    from pymicroglia.pipelines._runner import SavedResult,Unavailable
    frame=pd.DataFrame({'value':[1.,2.]})
    path=write_table(tmp_path/'entries.json',frame)
    ref=ArtifactRef('entries.csv',path.name,file_hash(path),'saved-science')
    saved=SavedResult(tmp_path,StepResult('render','saved-science','completed','test',(ref,)))
    assert saved.artifact('entries.json')==path
    pd.testing.assert_frame_equal(read_table(saved.artifact('entries.json')),frame)
    plain=tmp_path/'metadata.csv'
    plain.write_text('field,value\na,b\n',encoding='utf-8')
    ref=ArtifactRef('metadata.csv',plain.name,file_hash(plain),'saved-science')
    saved=SavedResult(tmp_path,StepResult('render','saved-science','completed','test',(ref,)))
    with pytest.raises(Unavailable):saved.artifact('metadata.json')
