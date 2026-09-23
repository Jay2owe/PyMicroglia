"""Pin optional scientific inputs to completed, verified execution artifacts."""
from pymicroglia._results import read_document
import json
from pathlib import Path

from pymicroglia.pipelines._contracts import content_id, result_from_dict
from pymicroglia.pipelines._runner import SavedResult, Unavailable, _validate
from pymicroglia.pipelines._screening import file_hash


def load_source(declaration, *, recipe, step):
    path = Path(declaration['execution_record']).expanduser().resolve()
    if not path.is_file(): raise Unavailable('Requested saved '+step+' execution record is unavailable')
    record = read_document(path)
    from pymicroglia.pipelines import _records
    invocation = None
    if 'invocations' in record:
        from auto_organotypic.layout import owning_folder
        root = owning_folder(path)
        if _records.path(root).resolve()!=path:
            raise ValueError('Optional source must be a pipeline execution ledger')
        record['invocations']={key:_records.invocation(root,key) for key in record['invocations']}
        candidates = [(key,row) for key,row in record['invocations'].items()
            if row.get('recipe',{}).get('name')==recipe and row.get('finished')
            and any(item.get('step')==step and item.get('result',{}).get('scientific_id')==declaration['scientific_id']
                    for item in row.get('steps',[]))]
        if not candidates:
            raise ValueError('Optional source scientific identity differs from the recorded executions')
        # Later replays of the same science do not change the pinned input.
        invocation,record = min(candidates,key=lambda item:item[1]['started'])
    else:
        if path.parent.name != 'executions':
            raise ValueError('Optional source must be an original pipeline execution record')
        root = path.parent.parent
    if record.get('schema_version') != 1 or record.get('recipe', {}).get('name') != recipe or not record.get('finished'):
        raise ValueError('Optional source is not a finished execution of '+recipe)
    rows = [row for row in record.get('steps', []) if row.get('step') == step]
    if len(rows) != 1: raise ValueError('Optional source must identify exactly one '+step+' outcome')
    row = rows[0]; result = result_from_dict(row['result'])
    if result.step != step or result.scientific_id != declaration['scientific_id']:
        raise ValueError('Optional source scientific identity or step differs from the declared result')
    if result.status not in {'completed', 'reused'}: raise Unavailable('Requested '+step+' source did not complete: '+result.reason)
    directory = (root/row['result_directory']).resolve()
    if not directory.is_relative_to(root): raise ValueError('Optional result directory escapes the source pipeline')
    _,_,receipt = _records.locate(directory)
    if receipt.get('schema_version') != 1 or content_id(receipt['result']) != receipt.get('result_sha256'):
        raise ValueError('Optional scientific result receipt is invalid')
    immutable = result_from_dict(receipt['result'])
    if immutable.status != 'completed' or immutable.step != step or immutable.scientific_id != result.scientific_id:
        raise ValueError('Optional result receipt does not identify the declared completed science')
    if immutable.artifacts != result.artifacts or immutable.selections != result.selections:
        raise ValueError('Optional execution outcome differs from its immutable scientific artifacts or selections')
    saved = SavedResult(directory, immutable); _validate(saved)
    return saved, {'execution_record': str(path), 'execution_record_sha256': content_id(record),
        'invocation':invocation,'fingerprint_scope':'selected immutable invocation and completion',
        'recipe': recipe, 'step': step, 'scientific_id': immutable.scientific_id,
        'receipt_sha256': content_id(receipt), 'artifacts': {ref.name: ref.sha256 for ref in immutable.artifacts}}
