"""Count a parent/fragment observation once, independent of parent fragmentation."""
from contextlib import contextmanager
import ast
import hashlib
import json
from pathlib import Path

VERSION = '2026-10-22-unique-parent-fragment-observations'

NORMALISE = '''    # Nearby pieces with one parent identity describe one observation,
    # not independent votes or extra fragment area. Keep the least favourable
    # brightness ratio so repeated contacts cannot strengthen weak-signal proof.
    keys = ["frame", "small_component", "large_name"]
    if evidence.duplicated(keys).any():
        grouped = evidence.groupby(keys, sort=True, dropna=False)
        same_fragment = ["small_name", "small_area", "x", "y", "absorption",
                         "strong_count", "protected_pixels"]
        if grouped[same_fragment].nunique(dropna=False).gt(1).any().any():
            raise ValueError("inconsistent repeated fragment observation")
        aggregation = {column: "first" for column in same_fragment}
        # The representative component is audit metadata, never an eligibility
        # decision. Eligibility uses the complete parent identity and evidence.
        aggregation.update(large_component="min", q95_ratio="max")
        evidence = grouped.agg(aggregation).reset_index().loc[:, EVIDENCE_COLUMNS]
'''


def repaired_function(source):
    source = source.replace('\r\n', '\n')
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == 'select_paths')
    original = ast.get_source_segment(source, node)
    if 'inconsistent repeated fragment observation' in original: return original, original
    marker = '    evidence = inventory.loc[:, EVIDENCE_COLUMNS].copy()\n'
    if original.count(marker) != 1:
        raise ValueError('Unrecognised weak-fragment evidence selection')
    return original, original.replace(marker, marker + NORMALISE, 1)


@contextmanager
def install(module):
    original, patched = repaired_function(Path(module.__file__).read_text(encoding='utf8'))
    sha = lambda b: hashlib.sha256(b).hexdigest()
    signature = dict(version=VERSION, repair_sha256=sha(Path(__file__).read_bytes()),
        original_function_sha256=sha(original.encode()), patched_function_sha256=sha(patched.encode()))
    old = module.select_paths
    namespace = {}
    exec(compile(patched, str(Path(module.__file__))+' [unique observations]', 'exec'), module.__dict__, namespace)
    module.select_paths = namespace['select_paths']
    try: yield signature
    finally: module.select_paths = old


@contextmanager
def activate(feedback, folder):
    old_methods = feedback.methods

    @contextmanager
    def methods(root, **kwargs):
        with old_methods(root, **kwargs) as sequence:
            with install(sequence.repeated.branches) as signature:
                record = Path(folder)/'fragment_observation_repair.json'
                if record.exists() and json.loads(record.read_text()) != signature:
                    raise ValueError('Fragment observation repair changed within recording')
                record.write_text(json.dumps(signature, indent=2)+'\n')
                yield sequence

    feedback.methods = methods
    try: yield
    finally: feedback.methods = old_methods
