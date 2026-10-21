"""Resolve competing process-fragment proposals before any label is changed.

The frozen recipe is retained. This explicit compatibility adapter changes only
its unsupported overlap case; uncontested results retain their original bytes.
"""
import ast
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

VERSION = '2026-10-21-process-proposal-conflicts'

APPLICATION = '''                    mask = masks[t, int(row.component)]
                    if np.any(candidate[t][mask] != int(row.small_name)):
                        raise AssertionError("overlapping process proposals")
                    candidate[t][mask] = int(large_name)
                    pixels = int(mask.sum())
                    audit["applied_pixels"] += pixels
                    applications.append({"frame": int(row.frame),
                                         "old_name": int(row.small_name),
                                         "new_name": int(large_name),
                                         "changed_pixels": pixels,
                                         "small_track": int(row.small_track),
                                         "gap_px": float(row.gap_px),
                                         "aligned_absorption": float(row.aligned_absorption),
                                         "raw_q95_ratio": float(row.raw_q95_ratio),
                                         "elongation": float(row.elongation)})'''

RESOLUTION = '''    # A fragment can be close to two large bodies. Collect claims before
    # mutation so neither identity ordering nor inventory ordering picks a winner.
    owners = {}
    for row, owner, audit in proposals:
        owners.setdefault((int(row.frame)-1, int(row.component)), set()).add(owner)
    conflicts = {key for key, claims in owners.items() if len(claims) > 1}
    if conflicts:
        frame_evidence["proposal_conflict"] = [
            (int(row.frame)-1, int(row.component)) in conflicts
            for row in frame_evidence.itertuples(index=False)]
        for audit in audits:
            audit["ambiguous_pixels"] = 0
    applied = set()
    for row, owner, audit in proposals:
        t, component = int(row.frame)-1, int(row.component)
        key = t, component
        mask = masks[key]
        if key in conflicts:
            audit["ambiguous_pixels"] += int(mask.sum())
            continue
        if key in applied:
            continue
        # This remains a real invariant: competing claims were handled above.
        if np.any(labels[t][mask] != int(row.small_name)):
            raise AssertionError("process proposal source changed")
        candidate[t][mask] = owner
        applied.add(key)
        pixels = int(mask.sum())
        audit["applied_pixels"] += pixels
        applications.append({"frame": int(row.frame),
                             "old_name": int(row.small_name),
                             "new_name": owner,
                             "changed_pixels": pixels,
                             "small_track": int(row.small_track),
                             "gap_px": float(row.gap_px),
                             "aligned_absorption": float(row.aligned_absorption),
                             "raw_q95_ratio": float(row.raw_q95_ratio),
                             "elongation": float(row.elongation)})
'''


def repaired_function(source):
    source = source.replace('\r\n', '\n')
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == 'correct')
    original = ast.get_source_segment(source, node)
    if 'process proposal source changed' in original:
        return original, original
    if original.count(APPLICATION) != 1:
        raise ValueError('Unrecognised process-group application; refusing implicit repair')
    patched = original.replace('audits, applications = [], []',
                               'audits, applications, proposals = [], [], []')
    patched = patched.replace(APPLICATION, '                    proposals.append((row, int(large_name), audit))')
    patched = patched.replace('    if (np.any((candidate != labels) & protected)',
                              RESOLUTION + '    if (np.any((candidate != labels) & protected)')
    return original, patched


@contextmanager
def install(module):
    original, patched = repaired_function(Path(module.__file__).read_text(encoding='utf8'))
    signature = dict(version=VERSION,
                     repair_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     original_function_sha256=hashlib.sha256(original.encode()).hexdigest(),
                     patched_function_sha256=hashlib.sha256(patched.encode()).hexdigest())
    old = module.correct
    namespace = {}
    exec(compile(patched, str(Path(__file__)), 'exec'), module.__dict__, namespace)
    module.correct = namespace['correct']
    try:
        yield signature
    finally:
        module.correct = old


@contextmanager
def activate(feedback, folder):
    """Apply to a frozen runner without modifying its archived source files."""
    old_methods = feedback.methods

    @contextmanager
    def methods(root, **kwargs):
        with old_methods(root, **kwargs) as sequence:
            with install(sequence.process.groups) as signature:
                record = Path(folder) / 'process_proposal_repair.json'
                if record.is_file() and json.loads(record.read_text()) != signature:
                    raise ValueError('Process proposal repair changed within this recording')
                record.write_text(json.dumps(signature, indent=2)+'\n')
                yield sequence

    feedback.methods = methods
    try:
        yield
    finally:
        feedback.methods = old_methods
