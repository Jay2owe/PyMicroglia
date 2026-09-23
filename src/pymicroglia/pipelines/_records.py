"""Atomic invocation and completion records, shared by all six workflows."""
from __future__ import annotations

import json
from pathlib import Path

from auto_organotypic import io, layout
from auto_organotypic.store.ledger import _folder_lock
from .._results import read_document


def path(folder):
    return layout.document(Path(folder), "executions.json")


def read(folder):
    target = path(folder)
    if not target.is_file():
        return {"schema_version": 1, "invocations": {}, "completions": {}}
    return read_document(target)


def update(folder, section, key, value):
    folder = Path(folder).resolve()
    with _folder_lock(folder):
        data = read(folder)
        previous = data[section].get(key)
        if section == "completions" and previous is not None and previous != value:
            raise ValueError("A completed pipeline result cannot be replaced")
        data[section][key] = value
        io.write_json(folder / "executions.json", data, workings=True)
    return path(folder)


def locate(root):
    root = Path(root).resolve()
    for folder in (root, *root.parents):
        target = path(folder)
        if target.is_file():
            key = root.relative_to(folder).as_posix()
            data = read(folder)
            if key in data["completions"]:
                return target, key, expand_completion(folder, data["completions"][key])
    legacy = layout.document(root, "execution-result.json")
    if legacy.is_file():
        return legacy, None, read_document(legacy)
    legacy = layout.document(root, "result.json")
    if legacy.is_file():
        from ._contracts import content_id
        encoded = read_document(legacy)
        return legacy, None, {"result": encoded, "result_sha256": content_id(encoded)}
    raise FileNotFoundError(f"No saved pipeline completion for {root}")


def completion(folder, root, value):
    from ._contracts import content_id, result_from_dict
    root,folder=Path(root).resolve(),Path(folder).resolve()
    encoded=value['result'];old=layout.document(root,'result.json')
    # A display snapshot can describe different inputs from the completed render.
    # Preserve that distinct document; otherwise store the same outcome once.
    filename='completion.json'
    if old.is_file():
        previous=read_document(old)
        if previous in (encoded,result_from_dict(encoded).as_dict()):filename='result.json'
    target=io.write_json(root/filename,encoded,workings=True)
    target=layout.document(root,filename)
    if filename=='result.json' and (root/filename).is_file() and (root/filename).resolve()!=target.resolve():
        (root/filename).unlink()
    stored={k:v for k,v in value.items() if k!='result'}
    stored['result_document']=target.relative_to(folder).as_posix()
    return update(folder, "completions", root.relative_to(folder).as_posix(), stored)


def expand_completion(folder,receipt):
    """Read the one outcome document; old inline records remain readable."""
    from ._contracts import content_id
    if 'result_document' not in receipt:return receipt
    root=Path(folder).resolve();target=(root/receipt['result_document']).resolve()
    if not target.is_relative_to(root):raise ValueError('Pipeline result document escapes its root')
    encoded=read_document(target)
    if content_id(encoded)!=receipt['result_sha256']:raise ValueError('Saved outcome or selections were changed')
    return {**{k:v for k,v in receipt.items() if k!='result_document'},'result':encoded}


def invocation(folder,key):
    """Expand references for readers without writing another copy of results."""
    import copy
    data=read(folder);record=copy.deepcopy(data['invocations'][key])
    for item in record['steps']:
        reference=item.pop('completion',None)
        if reference is not None:
            receipt=expand_completion(folder,data['completions'][reference])
            item['result']={**receipt['result'],**item.pop('outcome')}
    return record


def candidates(folder, cache_root):
    prefix = Path(cache_root).relative_to(folder).as_posix() + "/"
    recorded = [Path(folder) / name for name in read(folder)["completions"] if name.startswith(prefix)]
    legacy = [p.parent for p in Path(cache_root).glob("*/execution-result.json")]
    return sorted(set(recorded + legacy))
