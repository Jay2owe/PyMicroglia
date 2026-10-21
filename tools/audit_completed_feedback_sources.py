"""Connect every packaged correction source to the accepted replay evidence.

Reuses verified unchanged work instead of repeating multi-hour calculations.
New integration/evidence generation still needs its own execution check.
"""
from pathlib import Path
import ast
import hashlib
import json
import zipfile

PACKAGE=Path(__file__).resolve().parents[1]
MOTION=PACKAGE.parent/'Motion'
T=MOTION/'issues/ISSUE-009-completed-feedback-rule-optimization/analysis/completed_feedback_tuning'
R=T/'rounds/R223-speed-and-normalisation'


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    receipt_path=T/'diagnostics/r223_a162_complete_replay/run_receipt.json'
    receipt=json.loads(receipt_path.read_text())
    sources={str(Path(row['path']).resolve()):row['sha256'] for row in receipt['code']}
    replay=receipt['outputs']['completed_boundaries.json']
    assert digest(replay['path'])==replay['sha256']
    boundaries=json.loads(Path(replay['path']).read_text())
    recipe=json.loads((R/'a162_settings.json').read_text())
    expected={(movie,family,stage) for movie,values in recipe['movies'].items()
              for family,stages in values['expected'].items() for stage in stages}
    actual={(row['well'],row['family'],row['stage']) for row in boundaries if row['passed']}
    assert actual==expected and len(boundaries)==len(expected)
    adaptations={}
    for version in range(51,56):
        path=R/f'execution_adoption_v{version}.json'
        adoption=json.loads(path.read_text())
        assert adoption['status']=='verified_enabled'
        assert digest(adoption['proof_path'])==adoption['proof_sha256']
        assert json.loads(Path(adoption['strict_receipt']).read_text())['ok']
        for change in adoption['changes']:
            adaptations[str(Path(change['path']).resolve())]=(adoption,change,path)
    archive=PACKAGE/'src/pymicroglia/tracking/data/completed_feedback_engine.zip'
    rows=[]
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read('source_hashes.json'))
        for name,origin in manifest['source_origins'].items():
            data=z.read(name)
            value=hashlib.sha256(data).hexdigest()
            assert value==manifest['files_sha256'][name]
            path=str((MOTION/origin).resolve())
            if name.startswith('frozen/'):
                _,family,stage,module=name.split('/')
                op=next(op for op in recipe['operations'][family] if op['stage']==stage)
                assert value==op['code_sha256'][Path(module).stem]
                proof='Accepted replay operation specification, unchanged source'
            elif sources.get(path)==value:
                proof='Complete nine-movie replay, unchanged source'
            elif path in adaptations:
                adoption,change,record=adaptations[path]
                assert change['before_sha256']==sources[path]
                assert change['after_sha256']==value
                proof=str(record)
            elif name in manifest['packaging_adaptations']:
                original=(MOTION/origin).read_bytes()
                assert hashlib.sha256(original).hexdigest()==sources[path]
                def definitions(source):
                    return [ast.dump(node) for node in ast.parse(source.decode('utf-8-sig')).body
                            if isinstance(node,(ast.FunctionDef,ast.ClassDef))]
                assert definitions(original)==definitions(data)
                proof='Only unused command-line root lookup adapted; every function/class AST identical'
            elif name=='code/process_group_composition_v2.py':
                text=data.decode().replace('\r\n','\n')
                old='''        table = module.scan(movies, table)'''
                new='''        if table is None or len(table):
            table = module.scan(movies, table)
        if not len(table) and 'well' not in table:
            # No eligible pairs is valid for an independent new recording.
            # Later inventories only refine these pairs; none can create one.
            table = table.assign(well=pd.Series(dtype=str))'''
                assert text.count(new)==1
                restored=text.replace(new,old)
                assert sources[path] in {hashlib.sha256(restored.encode()).hexdigest(),
                    hashlib.sha256(restored.replace('\n','\r\n').encode()).hexdigest()}
                proof='Original nonempty execution unchanged; empty inventory regression checked separately'
            else:
                raise AssertionError('Uncovered packaged source: '+name)
            rows.append(dict(source=name,sha256=value,evidence=proof))
        declared={key:recipe[key] for key in ('density_parameters','revise_isolated_prior_reassignment','operations')}
        for family,ops in declared['operations'].items():
            for op in ops: op['code_dir']='frozen/'+family+'/'+op['stage']
        assert json.loads(z.read('settings.json'))==declared
    result=dict(status='pass',archive_sha256=digest(archive),sources=len(rows),
        accepted_movie_count=len(recipe['movies']),accepted_boundary_count=len(boundaries),
        receipt_sha256=digest(receipt_path),source_evidence=rows,
        scope='Source/setting equivalence to all-nine accepted replay plus verified later speed changes. Does not replace new evidence-generation and package integration execution checks.')
    output=PACKAGE/'validation/completed_base_20261012/source_coverage.json'
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_evidence'}))


if __name__=='__main__': main()
