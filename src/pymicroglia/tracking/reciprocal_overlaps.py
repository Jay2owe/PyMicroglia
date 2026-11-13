"""Compose reciprocal pair proposals before writing any identity pixels."""
from contextlib import contextmanager
from functools import wraps
import ast
import hashlib
import importlib
import inspect
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from .checkpoints import reconcile_repair

VERSION = '2026-10-23-reciprocal-proposal-composition'


def compose(labels, events, pair_rows, offset, minimum_area, components, connectivity):
    """Share agreeing support; defer whole connected groups with contradictory owners.

    All proposals use immutable parent labels. Only two frame-sized ownership
    buffers are needed for conflict discovery; no event-sized movie copies.
    """
    parents = list(range(len(events) + 1))
    conflicts, shared = set(), set()
    by_frame = {}
    for number, event in enumerate(events, 1):
        expected = {(int(r['source_frame']), str(r['component_id'])): int(r['identity'])
                    for r in event['assignments'] if bool(r['observed'])}
        for frame in sorted({int(r['source_frame']) for r in event['assignments']}):
            by_frame.setdefault(frame, []).append((number, event, expected))

    def find(n):
        while parents[n] != n:
            parents[n] = parents[parents[n]]
            n = parents[n]
        return n

    def proposals(frame):
        index = frame - offset - 1
        for number, event, expected in by_frame[frame]:
            pair = event['pair']
            connected, _ = ndi.label(np.isin(labels[index], pair), structure=connectivity)
            for component in components(labels[index], pair, frame, minimum_area):
                target = expected[(frame, str(component['component_id']))]
                yield number, component, target, connected == int(component['_raw_component_number'])

    for frame in sorted(by_frame):
        owners = np.zeros(labels.shape[1:], dtype=labels.dtype)
        claims = np.zeros(labels.shape[1:], dtype=np.int32)
        for number, component, target, mask in proposals(frame):
            for other in np.unique(claims[mask]):
                other = int(other)
                if not other or other == number:
                    continue
                parents[find(other)] = find(number)
                shared.update((number, other))
                if np.any(mask & (claims == other) & (owners != target)):
                    conflicts.update((number, other))
            free = mask & (claims == 0)
            claims[free] = number
            owners[free] = target
    bad_roots = {find(n) for n in conflicts}
    rejected = {n for n in range(1, len(events) + 1) if find(n) in bad_roots}
    row_by_id = {r['candidate_id']: r for r in pair_rows}
    for number, event in enumerate(events, 1):
        if number in rejected:
            row_by_id[event['candidate_id']].update(
                accepted=False, decision='refused_conflicting_pair_group_parent_retained')
        elif number in shared:
            row_by_id[event['candidate_id']]['decision'] = 'accepted_agreeing_pair_overlap'

    candidate = labels.copy()
    domains = np.zeros(labels.shape, dtype=np.uint16)
    actions = []
    # Preserve the original action/table order for nonoverlapping inputs.
    pending = {n: [] for n in range(1, len(events) + 1)}
    for frame in sorted(by_frame):
        index = frame - offset - 1
        for number, component, target, mask in proposals(frame):
            if number in rejected:
                continue
            changed_mask = mask & (candidate[index] != target)
            changed = int(np.count_nonzero(changed_mask))
            if not changed:
                continue
            previous = sorted(map(int, np.unique(candidate[index][changed_mask])))
            candidate[index][mask] = target
            domains[index][changed_mask] = number
            pending[number].append(dict(
                mechanism='L_two_lineage_lane_graph', source_frame=frame,
                component_id=component['component_id'],
                from_identity=previous[0] if len(previous) == 1 else '|'.join(map(str, previous)),
                to_identity=target, changed_pixels=changed,
                evidence='minimum whole-track displacement and area cost'))
    for number in pending:
        actions.extend(pending[number])
    return candidate, domains, actions, {events[n-1]['candidate_id'] for n in rejected}


def repaired_function(source):
    source = source.replace('\r\n', '\n')
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
                and n.name == 'reconcile_reciprocal_lineages')
    original = ast.get_source_segment(source, node)
    start = original.index('    candidate = labels.copy()')
    end = original.index('        for row in event["components"]:', start)
    if original.count('raise AssertionError("reciprocal pair repair domains overlap")') != 1:
        raise ValueError('Unrecognized reciprocal pair implementation')
    replacement = '''    candidate, domains, action_rows, rejected = _reciprocal_compose(
        labels, accepted, pair_rows, offset, minimum_area, frame_components, CONNECTIVITY)
    for event in accepted:
        if event["candidate_id"] in rejected:
            continue
'''
    return original, original[:start] + replacement + original[end:]


@contextmanager
def install(history, checkpoints, root):
    module = importlib.import_module('reciprocal_lineages')
    original, patched = repaired_function(Path(module.__file__).read_text(encoding='utf8'))
    digest = lambda value: hashlib.sha256(value).hexdigest()
    signature = dict(version=VERSION, repair_sha256=digest(Path(__file__).read_bytes()),
                     original_function_sha256=digest(original.encode()),
                     patched_function_sha256=digest(patched.encode()))
    root = Path(root)
    record = root/'reciprocal_overlap_repair.json'
    reconcile_repair(root, record, signature, 'checkpoint.json', 'reciprocal-pair repair')
    record.write_text(json.dumps(signature, indent=2))
    target = inspect.unwrap(module.reconcile_reciprocal_lineages)
    old_code, old_call = target.__code__, checkpoints.Stages.call
    old_function = module.reconcile_reciprocal_lineages
    sentinel = object()
    old_helper = module.__dict__.get('_reciprocal_compose', sentinel)
    namespace = {}
    exec(compile(patched, str(Path(module.__file__))+' [reciprocal overlap repair]', 'exec'), module.__dict__, namespace)
    target.__code__ = namespace['reconcile_reciprocal_lineages'].__code__
    def audited_compose(*args, **kwargs):
        import pandas as pd
        result = compose(*args, **kwargs)
        pd.DataFrame(args[2]).to_csv(root/'reciprocal_overlap_audit.csv', index=False)
        return result
    module._reciprocal_compose = audited_compose

    @wraps(old_function)
    def audited(*args, **kwargs):
        result = old_function(*args, **kwargs)
        result.pair_audit.to_csv(root/'reciprocal_overlap_audit.csv', index=False)
        return result

    def call(self, method, *args, **kwargs):
        if method != 'stage_m22_accepted_history':
            return old_call(self, method, *args, **kwargs)
        base = self.signature
        self.signature = {**base, 'reciprocal_overlap_repair': signature}
        try:
            return old_call(self, method, *args, **kwargs)
        finally:
            self.signature = base

    module.reconcile_reciprocal_lineages, checkpoints.Stages.call = audited, call
    try:
        yield signature
    finally:
        module.reconcile_reciprocal_lineages, checkpoints.Stages.call = old_function, old_call
        target.__code__ = old_code
        if old_helper is sentinel:
            module.__dict__.pop('_reciprocal_compose', None)
        else:
            module._reciprocal_compose = old_helper
