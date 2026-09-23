"""One CSV per scientific table; types live in the shared artefact ledger."""
from pathlib import Path
import csv
import hashlib
import io
import json
import pandas as pd
from auto_organotypic import store

# Native spectral diagnostics can exceed csv's small text-field default.
csv.field_size_limit(2**31-1)


def schema(path):
    entry=store.ledger.entry_for(path) or {}
    return entry.get('extra',{}).get('pipeline_table')


def fingerprint(path):
    """Hash table bytes and their type contract, so schema tampering is detected."""
    path=Path(path);data=path.read_bytes();contract=schema(path) if path.suffix=='.csv' else None
    if contract is not None:data+=b'\0'+json.dumps(contract,sort_keys=True,separators=(',',':')).encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def write(path,frame):
    from ._screening import _json_value
    from auto_organotypic.io import replace_with_retry
    from uuid import uuid4
    target=Path(path).with_suffix('.csv');target.parent.mkdir(parents=True,exist_ok=True)
    contract={'schema_version':2,'columns':list(frame.columns),'dtypes':{c:str(frame[c].dtype) for c in frame},'cell_encoding':'JSON scalar or nested value'}
    temporary=target.with_name('.'+target.name+'.'+uuid4().hex+'.partial')
    try:
        with temporary.open('w',encoding='utf-8',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(frame.columns)
            for row in frame.itertuples(index=False,name=None):writer.writerow([json.dumps(_json_value(v),ensure_ascii=False,allow_nan=False,separators=(',',':'),sort_keys=True) for v in row])
        replace_with_retry(temporary,target)
        store.claim('pipeline-table',store.fingerprint(target),contract,path=target,display_only=False,method_version='2',extra={'pipeline_table':contract})
    finally:
        if temporary.exists():temporary.unlink()
    return target


def read(path):
    path=Path(path)
    if path.suffix=='.json':
        # Existing immutable results are read without migrating their bytes.
        from .._results import read_document
        data=read_document(path)
        if data.get('schema_version')!=1:raise ValueError('unsupported pipeline table schema')
        frame=pd.DataFrame.from_records(data['records'],columns=data['columns']);types=data['dtypes']
    else:
        contract=schema(path)
        if contract is None or contract.get('schema_version')!=2:raise ValueError('Missing or unsupported pipeline table type contract')
        with path.open(encoding='utf-8',newline='') as stream:
            rows=csv.reader(stream);header=next(rows)
            if header!=contract['columns']:raise ValueError('Saved pipeline table columns changed')
            decoded=[]
            for row in rows:
                if len(row)!=len(header):raise ValueError('Saved pipeline table row width changed')
                decoded.append([json.loads(cell) for cell in row])
        frame=pd.DataFrame(decoded,columns=header);types=contract['dtypes']
    for column,dtype in types.items():frame[column]=frame[column].astype(dtype)
    return frame
