"""Allow repeated same-owner traces; audit conflicting connected event groups."""
from contextlib import contextmanager
from functools import wraps
import ast
import hashlib
import importlib
import inspect
import json
from pathlib import Path
from .checkpoints import reconcile_repair

VERSION = '2026-10-21-local-lineage-overlaps'

INITIALISE = '''    # Multiple detections can describe the same physical path. Domain
    # numbers retain provenance only; they must never decide cell ownership.
    parents = {int(e["event_number"]): int(e["event_number"]) for e in events}
    owners = {int(e["event_number"]): int(e["owner"]) for e in events}
    event_ids = {int(e["event_number"]): e["event_id"] for e in events}
    overlap_frames = {}
    def find(number):
        while parents[number] != number:
            parents[number] = parents[parents[number]]
            number = parents[number]
        return number
'''

OLD = '''            if np.any(occupied) and not np.all(occupied == number):
                raise AssertionError("independent local-lineage events overlap")
            domains[frame][body] = number'''

NEW = '''            for other in np.unique(occupied):
                other = int(other)
                if not other or other == number:
                    continue
                parents[find(other)] = find(number)
                overlap_frames.setdefault(number, frame + 1)
                overlap_frames.setdefault(other, frame + 1)
            # Record shared support once; all ownership decisions wait until all
            # proposals are traced, so event order cannot choose a winner.
            domains[frame][body & (domains[frame] == 0)] = number'''

RESOLVE = '''    group_owners = {}
    for number, owner in owners.items():
        group_owners.setdefault(find(number), set()).add(owner)
    rejected = {number for number in owners if len(group_owners[find(number)]) > 1}
    for number, frame in overlap_frames.items():
        boundary_rows.append({
            "event_id": event_ids[number], "direction": "composition",
            "boundary_review_frame": frame,
            "reason": ("overlap_different_owners_parent_retained" if number in rejected
                       else "overlap_same_owner_shared_path")})
    if rejected:
        for frame in range(len(domains)):
            domains[frame][np.isin(domains[frame], list(rejected))] = 0
    if overlap_frames:
        retained = []
        for row in frame_rows:
            if row["event_number"] in rejected:
                continue
            frame = int(row["review_frame"]) - 1
            mask = domains[frame] == int(row["event_number"])
            if not np.any(mask):
                continue
            row["body_area_px"] = int(mask.sum())
            row["owner_pixels"] = int(np.count_nonzero(labels[frame][mask] == row["owner"]))
            row["nonowner_pixels"] = row["body_area_px"] - row["owner_pixels"]
            retained.append(row)
        frame_rows = retained
'''


def repaired_function(source):
    source = source.replace('\r\n', '\n')
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == '_trace_events')
    original = ast.get_source_segment(source, node)
    if 'overlap_same_owner_shared_path' in original: return original, original
    if original.count(OLD) != 1 or original.count('    frame_columns = [') != 1:
        raise ValueError('Unrecognised local-lineage trace implementation')
    patched = original.replace('    for event in events:\n', INITIALISE + '    for event in events:\n', 1)
    patched = patched.replace(OLD, NEW).replace('    frame_columns = [', RESOLVE + '    frame_columns = [', 1)
    return original, patched


@contextmanager
def install(history, checkpoints, root):
    module = importlib.import_module('local_lineage_reservations')
    original, patched = repaired_function(Path(module.__file__).read_text(encoding='utf8'))
    if original == patched:
        yield None
        return
    sha = lambda value: hashlib.sha256(value).hexdigest()
    signature = dict(version=VERSION, repair_sha256=sha(Path(__file__).read_bytes()),
        original_function_sha256=sha(original.encode()), patched_function_sha256=sha(patched.encode()))
    root = Path(root)
    record = root/'local_lineage_overlap_repair.json'
    reconcile_repair(root, record, signature, 'mid/accepted_history/12_issue023_generalized.tif',
                     'local-lineage repair')
    record.write_text(json.dumps(signature, indent=2))
    target = inspect.unwrap(module._trace_events)
    old_code, old_call, old_trace = target.__code__, checkpoints.Stages.call, module._trace_events
    namespace = {}
    exec(compile(patched, str(Path(module.__file__))+' [overlap repair]', 'exec'), module.__dict__, namespace)
    target.__code__ = namespace['_trace_events'].__code__

    @wraps(old_trace)
    def traced(*args, **kwargs):
        result = old_trace(*args, **kwargs)
        result[2].to_csv(root/'local_lineage_overlap_audit.csv', index=False)
        return result

    def call(self, method, *args, **kwargs):
        if method != 'stage_m22_accepted_history': return old_call(self, method, *args, **kwargs)
        base = self.signature
        self.signature = {**base, 'lineage_overlap_repair': signature}
        try: return old_call(self, method, *args, **kwargs)
        finally: self.signature = base

    module._trace_events, checkpoints.Stages.call = traced, call
    try: yield signature
    finally:
        module._trace_events, checkpoints.Stages.call = old_trace, old_call
        target.__code__ = old_code
