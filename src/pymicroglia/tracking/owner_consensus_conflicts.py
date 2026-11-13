"""Keep zero-core attachments with their recipient through simultaneous renames."""
from contextlib import contextmanager
import ast
import hashlib
import importlib
import inspect
import json
from pathlib import Path
from .checkpoints import reconcile_repair

VERSION = '2026-10-22-owner-consensus-attachments'
FUNCTION = 'apply_dependency_ordered_proposals'

RESOLVE = '''    # Resolve attachment ownership on the receiving physical component, after
    # the release fixed point. A simultaneous rename must carry its attached
    # zero-core fragments; an identity lookup alone leaves the old name behind.
    for release in orphan_releases.values():
        if not (release["dependents"] & set(prepared)):
            continue
        border = ndi.binary_dilation(
            release["mask"], structure=np.ones((3, 3), bool)) & ~release["mask"]
        recipient_owners = set()
        for item in prepared.values():
            if int(item["proposal"].source_owner) != release["proposed_owner"]:
                continue
            for frame, mask in item["changes"]:
                if frame == release["frame"] and np.any(mask & border):
                    recipient_owners.add(int(item["proposal"].proposed_owner))
        if len(recipient_owners) > 1:
            # Two final owners claim the recipient: choose neither and keep
            # the attachment with its current owner.
            release["proposed_owner"] = release["source_owner"]
        elif recipient_owners:
            release["proposed_owner"] = recipient_owners.pop()

'''


def repaired_function(source):
    source = source.replace('\r\n', '\n')
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == FUNCTION)
    original = ast.get_source_segment(source, node)
    if 'recipient_owners = set()' in original:
        return original, original
    marker = '    candidate = labels.copy()\n'
    count = '"changed_pixels": int(release["mask"].sum()) if applied else 0,'
    if original.count(marker) != 1 or original.count(count) != 1:
        raise ValueError('Unrecognised owner-consensus implementation')
    patched = original.replace(marker, RESOLVE + marker)
    patched = patched.replace(count, '"changed_pixels": (int(release["mask"].sum()) if applied and\n'
        '                               release["source_owner"] != release["proposed_owner"] else 0),')
    return original, patched


@contextmanager
def install(history, checkpoints, root):
    module = importlib.import_module('owner_consensus')
    original, patched = repaired_function(Path(module.__file__).read_text(encoding='utf8'))
    if original == patched:
        yield None
        return
    sha = lambda b: hashlib.sha256(b).hexdigest()
    signature = dict(version=VERSION, repair_sha256=sha(Path(__file__).read_bytes()),
        original_function_sha256=sha(original.encode()), patched_function_sha256=sha(patched.encode()))
    root = Path(root)
    record = root/'owner_consensus_attachment_repair.json'
    reconcile_repair(root, record, signature, 'mid/accepted_history/16_issue032_owner_consensus.tif',
                     'owner-consensus repair')
    record.write_text(json.dumps(signature, indent=2))
    target = inspect.unwrap(getattr(module, FUNCTION))
    old_code, old_call = target.__code__, checkpoints.Stages.call
    namespace = {}
    exec(compile(patched, str(Path(__file__)), 'exec'), module.__dict__, namespace)
    target.__code__ = namespace[FUNCTION].__code__

    def call(self, method, *args, **kwargs):
        if method != 'stage_m22_accepted_history':
            return old_call(self, method, *args, **kwargs)
        base = self.signature
        self.signature = {**base, 'owner_consensus_repair': signature}
        try:
            return old_call(self, method, *args, **kwargs)
        finally:
            self.signature = base

    checkpoints.Stages.call = call
    try:
        yield signature
    finally:
        checkpoints.Stages.call = old_call
        target.__code__ = old_code
