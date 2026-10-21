"""Replay one accepted-history operation from the reference checkpoints.

The operation's inputs and expected outputs are the labels/unclaimed files that
run_accepted_history saves immediately before and after it. Every file the
operation writes is compared with the reference run's file at the same path.
Runs on the calling thread, matching the bounded runner that made the long
reference. Optional cProfile output names the remaining repeated work.
"""
import argparse
import ast
from contextlib import ExitStack
import cProfile
import hashlib
import inspect
import json
from pathlib import Path
import platform
import pstats
import sys
import time


def _target(node):
    # output_root / "NN_name.tif" or output_root / ("a" "b")
    # Checkpoints named by variables (the resumable head) are not bracketable.
    if isinstance(node, ast.BinOp) and isinstance(node.right, ast.Constant):
        return node.right.value
    return None


def plan(source, operation):
    """Return input and output checkpoint names around one operation call."""
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_accepted_history')
    saved = {}
    found = None
    arguments = None
    after = {}
    statements = []
    def flatten(body):
        for s in body:
            statements.append(s)
            for field in ('body', 'orelse'):
                inner = getattr(s, field, None)
                if isinstance(inner, list) and not isinstance(s, ast.FunctionDef):
                    flatten(inner)
    flatten(function.body)
    for s in statements:
        call = s.value if isinstance(s, (ast.Assign, ast.Expr)) else None
        if not isinstance(call, ast.Call):
            continue
        name = ast.unparse(call.func)
        if name == '_save' and len(call.args) >= 2 and isinstance(call.args[1], ast.Name):
            variable = call.args[1].id
            if variable in ('labels', 'unclaimed') and _target(call.args[0]) is not None:
                if found is None:
                    saved[variable] = _target(call.args[0])
                elif variable not in after:
                    after[variable] = _target(call.args[0])
        elif name == operation:
            if found is not None:
                raise ValueError('Operation called more than once: ' + operation)
            arguments = [ast.unparse(a) for a in call.args]
            if arguments not in UNIFORM:
                raise ValueError(f'{operation} has a non-uniform signature: {arguments}')
            found = s
        if found is not None and len(after) == 2:
            break
    if found is None:
        raise ValueError('Operation not found: ' + operation)
    if set(saved) != {'labels', 'unclaimed'} or set(after) != {'labels', 'unclaimed'}:
        raise ValueError(f'Could not bracket {operation}: before={saved} after={after}')
    return saved, after, arguments


UNIFORM = (['labels', 'unclaimed', 'raw', 'config', 'output_root'],
           ['labels', 'unclaimed', 'config', 'output_root'],
           ['labels', 'unclaimed', 'raw', 'motion_full', 'config', 'output_root'])


# Head operations take mixed arguments; name their checkpoint inputs and
# outputs explicitly. Labels saved before 04 are uncropped source frames.
HEAD = {
    # First operation: inputs are the m12/m19 history fields, not checkpoints.
    '_issue005': (dict(), dict(labels='01_issue005.tif')),
    '_issue008': (dict(labels='03_issue007.tif'), dict(labels='04_issue008.tif')),
    '_issue009': (dict(labels='04_issue008.tif'), dict(labels='05_issue009.tif')),
    '_issue010': (dict(labels='05_issue009.tif'),
                  dict(labels='06_issue010.tif', unclaimed='06_issue010_unclaimed.tif')),
    '_issue011': (dict(labels='06_issue010.tif', unclaimed='06_issue010_unclaimed.tif'),
                  dict(labels='07_issue011.tif', unclaimed='07_issue011_unclaimed.tif')),
    '_issue012': (dict(labels='07_issue011.tif', unclaimed='07_issue011_unclaimed.tif',
                       issue010='06_issue010.tif'),
                  dict(labels='08_issue012.tif', unclaimed='08_issue012_unclaimed.tif')),
    '_seat_identity': (dict(labels='08_issue012.tif', unclaimed='08_issue012_unclaimed.tif'),
                       dict(labels='09_seat_identity.tif', unclaimed='09_seat_identity_unclaimed.tif')),
    # The checkpointed block that ends in 11_issue017_rebase (seat identity,
    # blob residency, lineage aliases and continuity repair).
    '_issue017_block': (dict(labels='08_issue012.tif', unclaimed='08_issue012_unclaimed.tif'),
                        dict(labels='11_issue017_rebase.tif', unclaimed='11_issue017_rebase_unclaimed.tif')),
    '_issue023_tail': (dict(labels='11_issue017_rebase.tif', unclaimed='11_issue017_rebase_unclaimed.tif'),
                       dict(labels='12_issue023_generalized.tif', unclaimed='12_issue023_generalized_unclaimed.tif')),
    # Stationary reconciliation and the short-blank recovery between saves 12 and 13.
    '_issue027_block': (dict(labels='12_issue023_generalized.tif', unclaimed='12_issue023_generalized_unclaimed.tif'),
                        dict(labels='13_issue027_short_blank_recovery.tif',
                             unclaimed='13_issue027_short_blank_recovery_unclaimed.tif')),
}


def issue017_block(ah, inputs, raw_full, lag_full, raw, lag, cfg, stem, work, offset, interval):
    """Copy of the run_accepted_history block that writes 11_issue017_rebase."""
    import numpy as np
    import pipeline
    neutral_full = ah.load_stack(pipeline._pinned_paths(cfg, stem)['neutral_tracks'])
    observations, observation_table, _ = pipeline._load_anchor_inputs(stem, REPLAY_RUN[0])
    def call():
        labels, unclaimed, registry = ah._seat_identity(
            inputs['labels'], inputs['unclaimed'], raw, stem, work, offset, frame_interval_min=interval)
        labels, unclaimed, registry, _, manual_review = ah._general_blob_residency(
            labels, unclaimed, raw, registry, offset)
        manual_review.to_csv(work / "18_seat_identity/out/blob_residency_manual_review.csv", index=False)
        issue015_labels = labels.copy()
        issue015_unclaimed = unclaimed.copy()
        alias = ah.reconcile_physical_lineages(
            labels, unclaimed, observations, observation_table, neutral_full, lag,
            ah._general("R05_generalise_full_lineage_alias", offset))
        alias_ids = set()
        if len(alias.alias_audit):
            accepted_aliases = alias.alias_audit[alias.alias_audit.accepted.astype(bool)]
            alias_ids.update(accepted_aliases.alias_identity.astype(int))
            alias_ids.update(accepted_aliases.canonical_identity.astype(int))
        continuity_labels, continuity_unclaimed = ah._continuity_repair(
            issue015_labels, issue015_unclaimed, raw_full, lag_full,
            registry, alias_ids, work, offset, frame_interval_min=interval)
        alias_delta = alias.labels != issue015_labels
        continuity_delta = continuity_labels != issue015_labels
        alias_unclaimed_delta = alias.unclaimed != issue015_unclaimed
        continuity_unclaimed_delta = continuity_unclaimed != issue015_unclaimed
        if np.any(alias_delta & continuity_delta & (alias.labels != continuity_labels)):
            raise AssertionError("field-derived alias and continuity repairs conflict")
        if np.any(alias_unclaimed_delta & continuity_unclaimed_delta &
                  (alias.unclaimed != continuity_unclaimed)):
            raise AssertionError("field-derived alias and continuity unclaimed ledgers conflict")
        labels = alias.labels.copy()
        labels[continuity_delta] = continuity_labels[continuity_delta]
        unclaimed = alias.unclaimed.copy()
        unclaimed[continuity_unclaimed_delta] = continuity_unclaimed[continuity_unclaimed_delta]
        return labels, unclaimed
    return call


def issue027_block(ah, inputs, raw, lag, cfg, work):
    """Copy of the run_accepted_history block that writes 13_issue027_short_blank_recovery."""
    import pandas as pd
    def call():
        labels, unclaimed = inputs['labels'], inputs['unclaimed']
        stationary_cfg = cfg.values["accepted_postprocessing"].get(
            "field_wide_stationary_reconciliation", {})
        stationary_params = ah.reconciliation_params(stationary_cfg.get("parameters", {}))
        stationary_params["enabled"] = bool(stationary_cfg.get("enabled", True))
        ah._assert_identity_blind_params(stationary_params)
        if stationary_params["enabled"]:
            labels, _, _, events, _ = ah.discover_stationary_takeovers(labels, raw, stationary_params)
        else:
            events = pd.DataFrame()
        labels, unclaimed, _, _ = ah._issue027_short_blank_recovery(
            labels, unclaimed, raw, lag, cfg, work)
        return labels, unclaimed
    return call


def issue005_call(ah, raw_full, lag_full, cfg, stem, work):
    """_issue005 with the history fields decoded read-only from the m12 and m19 checkpoints."""
    import json
    import pipeline
    import tifffile
    from pymicroglia.tracking.checkpoints import Stages
    engine = Path(ah.__file__).resolve().parents[1]
    decoder = Stages.__new__(Stages); decoder.root = engine; decoder.loader = tifffile.imread
    history_inputs = ah.AcceptedHistoryInputs()
    for prior in ('m12_identity_reservation', 'm19_layered_lineage'):
        manifest = json.loads((engine / prior / REPLAY_RUN[0] / 'checkpoint.json').read_text())
        decoder._verify(manifest['files'])
        for name, value in manifest['history'].items():
            setattr(history_inputs, name, decoder._load(value))
    physical, reservation, _, _, motion_reserved = history_inputs.required()
    observations, observation_table, anchor_t = pipeline._load_anchor_inputs(stem, REPLAY_RUN[0])
    return lambda: ah._issue005(cfg, physical, reservation, raw_full, lag_full, observations,
                                observation_table, anchor_t, motion_reserved)


REPLAY_RUN = ['']


def head_call(ah, name, inputs, raw_full, lag_full, cfg, stem, work):
    """Call one head operation exactly as run_accepted_history does."""
    offset = cfg.source_frame_offset
    interval = float(cfg.values['frame_interval_min'])
    if name == '_issue005':
        return issue005_call(ah, raw_full, lag_full, cfg, stem, work)
    labels = inputs['labels']
    if name == '_issue008':
        labels = labels[offset:].copy()
    raw = raw_full[offset:offset + len(labels)]
    lag = ah.align_transition_stack(lag_full, len(labels), offset)
    if name == '_issue027_block':
        return issue027_block(ah, inputs, raw, lag, cfg, work)
    if name == '_issue017_block':
        return issue017_block(ah, inputs, raw_full, lag_full, raw, lag, cfg, stem, work, offset, interval)
    operation = getattr(ah, name)
    if name in ('_issue008', '_issue009'):
        return lambda: operation(labels, raw, lag, stem, work, offset, frame_interval_min=interval)
    if name == '_issue010':
        import pandas as pd
        authority = pd.read_csv(work / '12_movie_authority/out/movie_wide_authority.csv')
        return lambda: operation(labels, authority, lag, offset)
    if name == '_issue011':
        return lambda: operation(labels, inputs['unclaimed'], raw, stem, work, offset)
    if name == '_issue012':
        return lambda: operation(labels, inputs['unclaimed'], inputs['issue010'], raw, lag, offset)
    if name == '_seat_identity':
        return lambda: operation(labels, inputs['unclaimed'], raw, stem, work, offset, frame_interval_min=interval)
    return lambda: operation(labels, inputs['unclaimed'], raw, lag, offset)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('engine', 'snapshot', 'config', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--run', required=True)
    p.add_argument('--stem', required=True)
    p.add_argument('--operation', required=True)
    p.add_argument('--profile', action='store_true')
    p.add_argument('--native', action='store_true',
                   help='Run the frozen engine without the candidate adapters (baseline timing)')
    a = p.parse_args()
    for name in ('engine', 'snapshot', 'config', 'output'):
        setattr(a, name, getattr(a, name).resolve())
    platform.platform = lambda *x, **k: 'Windows-history-operation-replay'
    import numpy as np
    import pandas as pd
    import psutil
    import tifffile
    a.output.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(a.snapshot), str(a.engine / 'code')]
    from pymicroglia.tracking import frozen_io
    from pymicroglia.tracking.correction_audit import array_sha256
    from benchmark_working_storage import load
    import accepted_history as ah
    import common, optimized_pipeline
    storage = load(a.snapshot / 'array_storage.py')
    reference = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
    arguments = UNIFORM[0]
    if a.operation in HEAD:
        before, after = HEAD[a.operation]
    else:
        before, after, arguments = plan(inspect.getsource(ah), a.operation)
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    cfg = common.Config.load(a.config)
    holder = {}
    work = a.output / 'output_root'
    work.mkdir()
    # Operations read evidence that earlier operations saved in their stage
    # folders. Copy (never link) those folders, then compare only the files
    # this operation creates or rewrites.
    import shutil
    for folder in sorted(reference.iterdir()):
        if folder.is_dir():
            shutil.copytree(folder, work / folder.name)
    seeded = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in work.rglob('*') if p.is_file()}

    def execute():
        with ExitStack() as stack:
            pool = storage.Pool(a.output / 'scratch', minimum_bytes=1024**2, temporary_ram_bytes=32 * 1024**2)
            stack.enter_context(storage.install(pool))
            if not a.native:
                stack.enter_context(frozen_io.install(common, a.engine / 'code', a.output / 'scratch', array_pool=pool))

            def read(path):
                target = a.output / 'scratch' / ('input_' + hashlib.sha1(str(path).encode()).hexdigest()[:10] + '.dat')
                with tifffile.TiffFile(path) as opened:
                    value = pool.adopt(opened.series[0].asarray(out=str(target)), target)
                value.flags.writeable = False
                return value
            if a.operation in HEAD:
                import pipeline
                REPLAY_RUN[0] = a.run
                sources = pipeline._pinned_paths(cfg, a.stem)
                inputs = {k: read(reference / v).copy() for k, v in before.items()}
                call = head_call(ah, a.operation, inputs, read(sources['registered_raw']),
                                 read(sources['lag_float']), cfg, a.stem, work)
                operation = lambda *_: call()
                labels = unclaimed = raw = None
            else:
                labels = read(reference / before['labels']).copy()
                unclaimed = read(reference / before['unclaimed']).copy()
                raw = read(reference / '07_trimmed_evidence/out' / f'{a.stem}_registered_raw.tif')
                operation = getattr(ah, a.operation)
            holder['numeric_context'] = getattr(np, '_get_promotion_state', lambda: None)()
            profiler = cProfile.Profile() if a.profile else None
            tick = time.perf_counter(); cpu = time.process_time()
            values = dict(labels=labels, unclaimed=unclaimed, raw=raw, config=cfg, output_root=work)
            if 'motion_full' in arguments:
                import pipeline
                values['motion_full'] = read(pipeline._pinned_paths(cfg, a.stem)['motion_composite'])
            values = [values[name] for name in arguments]
            if profiler:
                result = profiler.runcall(operation, *values)
            else:
                result = operation(*values)
            holder['seconds'] = time.perf_counter() - tick
            holder['presence_stats'] = dict(getattr(storage, 'presence_stats', {}))
            holder['cpu_seconds'] = time.process_time() - cpu
            produced = result if isinstance(result, tuple) else (result,)
            holder['labels_equal'] = array_sha256(produced[0]) == array_sha256(read(reference / after['labels']))
            holder['unclaimed_equal'] = ('unclaimed' not in after or
                array_sha256(produced[1]) == array_sha256(read(reference / after['unclaimed'])))
            if profiler:
                profiler.dump_stats(str(a.output / 'profile.pstats'))
                stats = pstats.Stats(profiler)
                rows = [dict(function=f'{Path(f).name}:{line}:{fn}', calls=c[1], own=round(c[2], 3), cumulative=round(c[3], 3))
                        for (f, line, fn), c in stats.stats.items()]
                holder['by_own'] = sorted(rows, key=lambda r: -r['own'])[:40]
                holder['by_cumulative'] = sorted(rows, key=lambda r: -r['cumulative'])[:60]

    optimized_pipeline.main(['--no-evidence-cache'], runner=execute)
    checks = {}
    for path in sorted(work.rglob('*')):
        if not path.is_file() or path.suffix not in ('.csv', '.tif', '.tiff', '.json'):
            continue
        if seeded.get(path) == (path.stat().st_size, path.stat().st_mtime_ns):
            continue
        rel = path.relative_to(work)
        expected = reference / rel
        if not expected.is_file():
            checks[str(rel)] = 'missing_in_reference'
        elif path.suffix == '.csv' or path.suffix == '.json':
            checks[str(rel)] = path.read_bytes() == expected.read_bytes()
        else:
            checks[str(rel)] = bool(np.array_equal(tifffile.imread(path), tifffile.imread(expected)))
    passed = holder['labels_equal'] and holder['unclaimed_equal'] and all(v is True for v in checks.values())
    memory = psutil.Process().memory_info()
    report = dict(status='pass' if passed else 'fail', operation=a.operation, native=a.native,
                  inputs=before, expected=after, checks=checks,
                  peak_resident_bytes=getattr(memory, 'peak_wset', memory.rss), **holder)
    (a.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    summary = {k: v for k, v in report.items() if k not in ('by_own', 'by_cumulative', 'checks')}
    summary['checks_failed'] = [k for k, v in checks.items() if v is not True]
    print(json.dumps(summary), flush=True)
    if a.profile:
        print('-- own time'); [print(r) for r in holder['by_own'][:25]]
        print('-- cumulative'); [print(r) for r in holder['by_cumulative'][:40]]
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
