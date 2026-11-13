"""Audit identity retirement instead of forbidding combined owner corrections.

No assignment decision changes: the existing consensus producer supplies all
proposals. Two independently supported excursions may exhaust a false name.
Keep the segmentation and duplicate-component checks in the original stage.
Ported from the CD68 full-duration round (2026-10-23), where it let A2 finish.
"""
from contextlib import contextmanager
import __future__
import ast
import hashlib
import importlib
import json
from pathlib import Path
from types import ModuleType
from .checkpoints import reconcile_repair

VERSION = '2026-10-23-owner-identity-accounting'
FUNCTION = '_owner_consensus'


def repaired_function(source):
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == FUNCTION)
    original = ast.get_source_segment(source, node)
    guards = [n for n in node.body if isinstance(n, ast.If)
              and ast.unparse(n.test) == 'candidate_identities != baseline_identities']
    if len(guards) != 1 or 'owner consensus changed the active identity set' not in ast.unparse(guards[0]):
        raise ValueError('Unrecognised owner identity accounting guard')
    node.body.remove(guards[0])
    metrics = [n.value for n in node.body if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'metrics' for t in n.targets)
               and isinstance(n.value, ast.Dict)]
    if len(metrics) != 1:
        raise ValueError('Unrecognised owner consensus metrics')
    for key, expression in [('retired_identities', 'sorted(baseline_identities - candidate_identities)'),
                            ('introduced_identities', 'sorted(candidate_identities - baseline_identities)')]:
        metrics[0].keys.append(ast.Constant(key))
        metrics[0].values.append(ast.parse(expression, mode='eval').body)
    return original, ast.unparse(ast.fix_missing_locations(node)) + '\n'


@contextmanager
def install(history, checkpoints, root):
    original, patched = repaired_function(Path(history.__file__).read_text(encoding='utf8'))
    sha = lambda b: hashlib.sha256(b).hexdigest()
    signature = dict(version=VERSION, repair_sha256=sha(Path(__file__).read_bytes()),
                     original_function_sha256=sha(original.encode()), patched_function_sha256=sha(patched.encode()))
    root = Path(root)
    record = root / 'owner_identity_accounting_repair.json'
    reconcile_repair(root, record, signature, 'mid/accepted_history/16_issue032_owner_consensus.tif',
                     'owner identity accounting')
    record.write_text(json.dumps(signature, indent=2))
    # A real, recorded source file lets later inspect/AST acceleration see this
    # exact stage. The frozen archive and extracted code remain unchanged.
    source_path = root / 'owner_identity_accounting_stage.py'
    source_path.write_text(patched, encoding='utf8')
    namespace = {}
    exec(compile(patched, str(source_path), 'exec', flags=__future__.annotations.compiler_flag), vars(history), namespace)
    cache = importlib.import_module('physical_evidence_cache')
    old_function = getattr(history, FUNCTION)
    setattr(history, FUNCTION, namespace[FUNCTION])
    old_builder = cache._owner_function_with_evidence

    def builder(actual_history, entry):
        if actual_history is not history:
            return old_builder(actual_history, entry)
        proxy = ModuleType(history.__name__)
        proxy.__dict__.update(vars(history))
        proxy.__file__ = str(source_path)
        return old_builder(proxy, entry)

    cache._owner_function_with_evidence = builder
    old_call = checkpoints.Stages.call

    def call(self, method, *args, **kwargs):
        if method != 'stage_m22_accepted_history':
            return old_call(self, method, *args, **kwargs)
        base = self.signature
        self.signature = {**base, 'owner_identity_accounting_repair': signature}
        try:
            return old_call(self, method, *args, **kwargs)
        finally:
            self.signature = base

    checkpoints.Stages.call = call
    try:
        yield signature
    finally:
        checkpoints.Stages.call = old_call
        cache._owner_function_with_evidence = old_builder
        setattr(history, FUNCTION, old_function)
