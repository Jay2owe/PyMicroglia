"""Keep a recording's run going when one correction stage cannot handle it.

The accepted correction chain was tuned on one recording. On another, a stage
can meet data it never saw (two claims on one identity, or nothing to fix) and
raise, which used to end the whole multi-hour run. Each chain stage that takes
and returns ``(labels, unclaimed, ...)`` is wrapped here. If it raises a data
error and provably left its input untouched, the run continues with that input:
the stage's changes are skipped, never partly applied, and the skip is printed
and written to ``stage_fallback_audit.json``.

Inputs are made read-only while a stage runs, so a failed stage cannot have
changed them. A stage that writes into its own input in place is run again
without that protection, giving its normal result. When no stage raises,
every output is identical to the unwrapped chain.

The fixing stage's blocks each yield a fixed list of required steps. If a step
fails, the block's remaining steps are recorded as unchanged copies of the last
good movie, written to ``feedback_stage_fallback_audit.json``, so the run still
publishes every required step.

Two claims on one midpoint soma name, with no nearby bookend name for either,
use the existing ``new_identity`` anchor policy: the second soma gets its own
new name instead of stopping the run. A recording that resolves under the
default policy is unchanged.
"""
from contextlib import contextmanager
from dataclasses import replace
from functools import wraps
import ast
import hashlib
import importlib
import inspect
import json
import sys
import threading
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = '2026-10-29-stage-fallbacks'
AUDIT = 'stage_fallback_audit.json'
ANCHOR_DUPLICATE = 'duplicate anchor identity'


class Nothing(int):
    """One metric of a skipped stage: zero as a count, empty as a list or mapping."""

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0

    def items(self):
        return iter(())

    keys = values = items


class SkippedMetrics(dict):
    """Metrics of a skipped stage: it applied nothing, so any count is zero."""

    def __missing__(self, key):
        return Nothing(0)


def stage_shapes(source):
    """Return ``{stage: result count}`` for uniform chain stages in run_accepted_history."""
    tree = ast.parse(source.replace('\r\n', '\n'))
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_accepted_history')
    # A stage is a private function of this module; chosen by shape, not by name.
    defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    sites = {}
    for node in ast.walk(run):
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)):
            continue
        name = node.value.func.id
        if not (name.startswith('_') and name in defined):
            continue
        target = node.targets[0]
        names = [ast.unparse(e) for e in target.elts] if isinstance(target, ast.Tuple) else []
        arguments = [ast.unparse(a) for a in node.value.args[:2]]
        uniform = (names[:2] == ['labels', 'unclaimed'] and arguments == ['labels', 'unclaimed']
                   and (len(names) == 2 or names[-1].endswith('metrics')))
        sites.setdefault(name, []).append(len(names) if uniform else None)
    return {name: counts[0] for name, counts in sites.items()
            if len(set(counts)) == 1 and counts[0] is not None}


CODE_FAULTS = (ImportError, NameError, SyntaxError, NotImplementedError)


def data_error(error, skipped):
    """True for errors a recording's data can cause; never for memory, disk or missing code.

    Duplicate-identity gates, empty tables, unexpected types and failed graph
    solvers all count: a recording can trigger any of them. Memory and disk
    errors are left to the caller's retry, and a missing module or name would
    fail every recording the same way.
    """
    if isinstance(error, FileNotFoundError):
        # A later stage may need files that an already skipped stage never wrote.
        return bool(skipped)
    if isinstance(error, (MemoryError, OSError) + CODE_FAULTS):
        return False
    if isinstance(error, AttributeError) and str(error).startswith('module '):
        return False
    return isinstance(error, Exception)


def read_only_write(error):
    return isinstance(error, ValueError) and 'read-only' in str(error)


class Protection:
    """Reference-counted read-only flags, so nested or concurrent calls share them."""

    def __init__(self):
        self.lock = threading.Lock()
        self.held = {}

    def acquire(self, values):
        arrays = []
        with self.lock:
            for value in values:
                if not isinstance(value, np.ndarray):
                    continue
                key = id(value)
                if key in self.held:
                    self.held[key][1] += 1
                elif value.flags.writeable:
                    value.flags.writeable = False
                    self.held[key] = [value, 1]
                else:
                    continue
                arrays.append(value)
        return arrays

    def release(self, arrays):
        with self.lock:
            for value in arrays:
                entry = self.held[id(value)]
                entry[1] -= 1
                if not entry[1]:
                    del self.held[id(value)]
                    value.flags.writeable = True

    def free_views(self, result, arrays):
        """Views of a protected input made during the stage must not stay read-only."""
        values = result if isinstance(result, tuple) else (result,)
        for value in values:
            if (isinstance(value, np.ndarray) and not value.flags.writeable and value.base is not None
                    and any(np.may_share_memory(value, a) for a in arrays)):
                try:
                    value.flags.writeable = True
                except ValueError:
                    pass


def guarded(name, function, count, protection, skipped, record):
    @wraps(function)
    def stage(*args, **kwargs):
        arrays = protection.acquire(list(args) + list(kwargs.values()))
        try:
            result = function(*args, **kwargs)
        except Exception as error:
            protection.release(arrays)
            arrays = []
            if read_only_write(error):
                record(dict(stage=name, action='rerun_without_protection', error=str(error)))
                return function(*args, **kwargs)
            if not data_error(error, skipped):
                raise
            skipped.append(name)
            record(dict(stage=name, action='skipped_input_kept', error_type=type(error).__name__,
                        error=str(error), at=_location(error)))
            print(f'STAGE SKIPPED {name}: {type(error).__name__}: {error}; its input passes on unchanged',
                  flush=True)
            return (args[0], args[1], *[pd.DataFrame() for _ in range(count - 3)],
                    *([SkippedMetrics(stage_skipped=1)] if count > 2 else []))
        else:
            protection.release(arrays)
            protection.free_views(result, arrays)
            return result
    return stage


def _location(error):
    frames = traceback.extract_tb(error.__traceback__)
    return [f'{Path(f.filename).name}:{f.lineno} {f.name}' for f in frames[-3:]]


def anchor_fallback(function, record):
    @wraps(function)
    def resolve(physical_objects, accepted_labels, anchor_t, params=None, *args, **kwargs):
        call = (physical_objects, accepted_labels, anchor_t) + ((params,) if params is not None else ())
        try:
            return function(*call, *args, **kwargs)
        except AssertionError as error:
            if not str(error).startswith(ANCHOR_DUPLICATE) or params is None \
                    or params.unmatched_anchor_policy != 'error':
                raise
            record(dict(stage='resolve_anchor_identity_map', action='new_identity_policy',
                        error=str(error)))
            print(f'ANCHOR NAMES: {error}; the unsupported soma gets its own new name', flush=True)
            return function(physical_objects, accepted_labels, anchor_t,
                            replace(params, unmatched_anchor_policy='new_identity'), *args, **kwargs)
    return resolve


def writer(path):
    """Record function that keeps every audit row written to ``path``."""
    rows, lock = [], threading.Lock()

    def record(row):
        with lock:
            rows.append(row)
            Path(path).write_text(json.dumps(rows, indent=2) + '\n')
    return record


@contextmanager
def anchor_names(record):
    """Apply the new-name anchor policy to every soma-ledger call while open.

    The final stage replays tracking from the start, so it meets the same
    duplicate anchor the accepted history did and needs the same policy.
    """
    ledger = importlib.import_module('global_soma_ledger')
    old_resolve = ledger.resolve_anchor_identity_map
    ledger.resolve_anchor_identity_map = anchor_fallback(old_resolve, record)
    try:
        yield
    finally:
        ledger.resolve_anchor_identity_map = old_resolve


class EmptyTables(dict):
    """Tables of a skipped fixing step: any table it would have made is empty."""

    def __missing__(self, key):
        return pd.DataFrame()


# Fixing-stage blocks by the family name the sequence records them under, with
# what each yields after the step name and which inputs hold the starting movie.
FEEDBACK_BLOCKS = {
    **{family: 'ledger' for family in (
        'ownerless', 'embedded', 'union', 'optical', 'branch', 'repeated', 'duplicate', 'terminal',
        'handoff', 'dim', 'anchored', 'birth', 'detector', 'reviewed', 'final')},
    **{family: 'arrays' for family in ('recovery', 'current', 'contact')},
    'owner': 'owner', 'process': 'process',
}


def _start(kind, args):
    if kind == 'process':
        return args[0]['branch_labels'], args[0]['ledger']
    return (args[0], args[3]) if kind == 'owner' else (args[0], args[1])


def _placeholder(kind, stage, state):
    labels, other = state
    if kind == 'owner':
        return stage, labels, EmptyTables(), other
    item = (stage, labels, other.copy(), EmptyTables())
    return item + ({},) if kind == 'arrays' else item


def resilient_block(family, function, kind, steps, protection, skipped, record):
    """Run one fixing block; if a step fails, keep the last good movie for its remaining steps."""
    @wraps(function)
    def apply(*args, **kwargs):
        inputs = list(args) + list(kwargs.values())
        if kind == 'process':
            inputs += list(args[0].values())
        state, done = _start(kind, args), []
        generator = function(*args, **kwargs)
        while True:
            arrays = protection.acquire(inputs + list(state))
            try:
                item = next(generator)
            except StopIteration:
                protection.release(arrays)
                return
            except Exception as error:
                protection.release(arrays)
                if read_only_write(error):
                    # Replay the whole block unprotected; its earlier steps are
                    # deterministic and were already recorded.
                    record(dict(stage=family, action='rerun_without_protection', error=str(error)))
                    generator = function(*args, **kwargs)
                    for name in done:
                        if next(generator)[0] != name:
                            raise RuntimeError(f'fixing block {family} replayed different steps') from error
                    yield from generator
                    return
                if not data_error(error, skipped):
                    raise
                remaining = [stage for stage in steps if stage not in done]
                skipped.append(family)
                record(dict(stage=family, action='remaining_steps_kept_last_movie', after=done[-1:],
                            steps=remaining, error_type=type(error).__name__, error=str(error),
                            at=_location(error)))
                print(f'FIXING STEPS SKIPPED {family} {remaining}: {type(error).__name__}: {error}; '
                      'the last good movie passes on unchanged', flush=True)
                for stage in remaining:
                    yield _placeholder(kind, stage, state)
                return
            protection.release(arrays)
            protection.free_views(item, arrays)
            done.append(item[0])
            state = (item[1], item[3] if kind == 'owner' else item[2])
            yield item
    return apply


def resilient_inventory(function, stages, protection, skipped, record):
    """Process inventory: a failed scan gives the engine's own "no eligible pairs" table."""
    @wraps(function)
    def inventory(movies):
        inputs = [value for movie in movies.values() for value in movie.values()]
        done = []
        generator = function(movies)
        while True:
            arrays = protection.acquire(inputs)
            try:
                item = next(generator)
            except StopIteration:
                protection.release(arrays)
                return
            except Exception as error:
                protection.release(arrays)
                if read_only_write(error):
                    record(dict(stage='inventory', action='rerun_without_protection', error=str(error)))
                    generator = function(movies)
                    for name in done:
                        if next(generator)[0] != name:
                            raise RuntimeError('process inventory replayed different scans') from error
                    yield from generator
                    return
                if not data_error(error, skipped):
                    raise
                remaining = [stage for stage in stages if stage not in done]
                skipped.append('inventory')
                record(dict(stage='inventory', action='no_eligible_process_pairs', steps=remaining,
                            error_type=type(error).__name__, error=str(error), at=_location(error)))
                print(f'PROCESS INVENTORY {remaining}: {type(error).__name__}: {error}; '
                      'treated as no eligible process pairs', flush=True)
                for stage in remaining:
                    yield stage, pd.DataFrame({'well': pd.Series(dtype=str)})
                return
            protection.release(arrays)
            done.append(item[0])
            yield item
    return inventory


CYCLIC_OLD = '            if len(track_transitions) else track_transitions\n'
CYCLIC_NEW = ('            if len(track_transitions) else track_transitions.reindex(\n'
              '                columns=["old_identity", "new_identity"])\n')


def cyclic_function(source):
    """Count no track handoffs, rather than fail, when the track table has no columns."""
    source = source.replace('\r\n', '\n')
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'discover')
    original = ast.get_source_segment(source, node)
    if original.count(CYCLIC_OLD) != 1:
        raise ValueError('Unrecognised cyclic-alias discovery')
    return original, original.replace(CYCLIC_OLD, CYCLIC_NEW)


@contextmanager
def cyclic_repair(module):
    original, patched = cyclic_function(Path(module.__file__).read_text(encoding='utf8'))
    target = inspect.unwrap(module.discover)
    namespace = {}
    exec(compile(patched, str(Path(module.__file__)) + ' [empty track table]', 'exec'), module.__dict__, namespace)
    old_code, target.__code__ = target.__code__, namespace['discover'].__code__
    try:
        yield hashlib.sha256(patched.encode()).hexdigest()
    finally:
        target.__code__ = old_code


@contextmanager
def feedback_blocks(feedback, folder):
    """Make each fixing-stage block finish its required steps even when one step fails."""
    old_methods = feedback.methods
    folder = Path(folder)
    sha = lambda value: hashlib.sha256(value).hexdigest()
    signature = dict(version=VERSION, repair_sha256=sha(Path(__file__).read_bytes()))
    path = folder/'feedback_stage_fallbacks.json'
    # Unfinished fixing-stage work is never reused, so newer rules simply apply.
    path.write_text(json.dumps(signature, indent=2))
    manifest = json.loads((folder/'completed_feedback_engine/source_hashes.json').read_text())
    steps = {}
    for family, stage in manifest['required_boundaries']:
        steps.setdefault(family, []).append(stage)
    rows, skipped, lock = [], [], threading.Lock()

    def record(row):
        with lock:
            rows.append(row)
            (folder/('feedback_' + AUDIT)).write_text(json.dumps(rows, indent=2) + '\n')

    @contextmanager
    def methods(root, **kwargs):
        with old_methods(root, **kwargs) as sequence:
            protection = Protection()
            saved = []
            for family, kind in FEEDBACK_BLOCKS.items():
                block = getattr(sequence, family)
                saved.append((block, 'apply', block.apply))
                block.apply = resilient_block(family, block.apply, kind, steps[family],
                                              protection, skipped, record)
            process = sequence.process
            saved.append((process, 'inventory', process.inventory))
            process.inventory = resilient_inventory(
                process.inventory, [stage for stage, _ in process.INVENTORIES], protection, skipped, record)
            try:
                with cyclic_repair(sys.modules['recurrent_cyclic_alias']):
                    yield sequence
            finally:
                for owner, name, function in reversed(saved):
                    setattr(owner, name, function)

    feedback.methods = methods
    try:
        yield signature
    finally:
        feedback.methods = old_methods


@contextmanager
def install(history, checkpoints, root):
    shapes = stage_shapes(Path(history.__file__).read_text(encoding='utf8'))
    sha = lambda value: hashlib.sha256(value).hexdigest()
    signature = dict(version=VERSION, repair_sha256=sha(Path(__file__).read_bytes()),
                     stages=len(shapes))
    root = Path(root)
    record_path = root/'stage_fallbacks.json'
    try:
        changed = record_path.exists() and json.loads(record_path.read_text()) != signature
    except ValueError:
        changed = True
    record_path.write_text(json.dumps(signature, indent=2))
    skipped, record = [], writer(root/AUDIT)
    protection = Protection()
    saved = [(name, getattr(history, name)) for name in shapes if callable(getattr(history, name, None))]
    for name, function in saved:
        setattr(history, name, guarded(name, function, shapes[name], protection, skipped, record))
    old_call = checkpoints.Stages.call

    def call(self, method, *args, **kwargs):
        nonlocal changed
        if method != 'stage_m22_accepted_history':
            return old_call(self, method, *args, **kwargs)
        base = self.signature
        stale = changed and Path(self.root)/'m22_accepted_history'/self.run_name
        if stale and stale.exists():
            # Accepted history saved under other skip rules restarts from its
            # first correction; the earlier folder is kept beside it.
            from .checkpoints import set_aside
            set_aside(stale, 'superseded')
        changed = False
        self.signature = {**base, 'stage_fallbacks': signature}
        try:
            return old_call(self, method, *args, **kwargs)
        finally:
            self.signature = base

    checkpoints.Stages.call = call
    try:
        with anchor_names(record):
            yield signature
    finally:
        checkpoints.Stages.call = old_call
        for name, function in reversed(saved):
            setattr(history, name, function)
