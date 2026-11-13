"""Preserve ownership-table columns when temporal evidence is empty."""
from contextlib import contextmanager
import ast
import hashlib
import inspect
import json
from pathlib import Path

VERSION = '2026-10-22-core-replacement-empty-evidence'
FUNCTION = 'preserve_core_replacement_owners'
FILTERS = (
    ('g', 'n>0 and len(by_node.get(int(n),()))==1'),
    ('later', 'n>0 and len(by_node.get(int(n),()))==1'),
    ('prior', 'len(by_node.get(int(n),()))==1'),
)


def repaired_function(source):
    source = source.replace('\r\n', '\n')
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == FUNCTION)
    original = ast.get_source_segment(source, node)
    patched = original
    for table, predicate in FILTERS:
        old = f'{table}[{table}.node.map(lambda n:{predicate})]'
        new = f'{table}.loc[{table}.node.map(lambda n:{predicate}).astype(bool)]'
        if new in patched:
            continue
        if patched.count(old) != 1:
            raise ValueError('Unrecognised core-replacement evidence filter')
        patched = patched.replace(old, new)
    return original, patched


@contextmanager
def activate(feedback, folder):
    old_methods = feedback.methods
    sha = lambda b: hashlib.sha256(b).hexdigest()
    signature = dict(version=VERSION, repair_sha256=sha(Path(__file__).read_bytes()))
    path = Path(folder)/'core_replacement_evidence_repair.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError):
        record = {}
    if record.get('signature') != signature:
        # Unfinished correction work is never reused, so a newer repair simply applies.
        record = dict(signature=signature, functions={})
        path.write_text(json.dumps(record, indent=2))

    @contextmanager
    def methods(root, **kwargs):
        with old_methods(root, **kwargs) as sequence:
            original_loader = sequence.reviewed.frozen_method

            @contextmanager
            def frozen_method(spec):
                with original_loader(spec) as method:
                    target = inspect.unwrap(method)
                    if target.__name__ != FUNCTION:
                        yield method
                        return
                    source = Path(inspect.getsourcefile(target)).read_text(encoding='utf8')
                    original, patched = repaired_function(source)
                    key = sha(original.encode())
                    entry = dict(original_function_sha256=key,
                                 patched_function_sha256=sha(patched.encode()))
                    record['functions'][key] = entry
                    path.write_text(json.dumps(record, indent=2))
                    namespace = {}
                    exec(compile(patched, str(Path(__file__)), 'exec'), target.__globals__, namespace)
                    old_code = target.__code__
                    target.__code__ = namespace[FUNCTION].__code__
                    try:
                        yield method
                    finally:
                        target.__code__ = old_code

            sequence.reviewed.frozen_method = frozen_method
            try:
                yield sequence
            finally:
                sequence.reviewed.frozen_method = original_loader

    feedback.methods = methods
    try:
        yield
    finally:
        feedback.methods = old_methods
