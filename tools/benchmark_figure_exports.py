"""Measure complete figure exports on saved inputs, with optional diagnostic profiles.

Uses the live public action dispatcher; no scientific or export settings are weakened.
Keep exports in owned scratch and receipts in the performance work home.
"""
from __future__ import annotations
import argparse
import cProfile
import hashlib
import importlib.metadata
import importlib
import json
import os
from pathlib import Path
import pstats
import sys
import time
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
PROCESS_STARTED = time.perf_counter()


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def source_identity(package):
    return {p.relative_to(package).as_posix(): digest(p) for p in package.rglob('*')
            if p.is_file() and p.suffix in {'.py', '.json', '.yaml', '.yml', '.toml'}
            and '__pycache__' not in p.parts}


def main(tracker):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--exports', type=Path, required=True)
    parser.add_argument('--run', type=Path, default=ROOT/'tests/fixtures/motion_parity/run')
    parser.add_argument('--stem', default='parity_A1')
    parser.add_argument('--actions', nargs='+')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--options', type=Path)
    parser.add_argument('--source', type=Path, help='Frozen parent of the pymicroglia package')
    parser.add_argument('--cases', type=Path, help='Prior saved fixture selections by action')
    parser.add_argument('--all', action='store_true', help='Include saved pipeline figure families')
    parser.add_argument('--direct', action='store_true', help='Use the public Python figure function')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.exports.mkdir(parents=True, exist_ok=True)
    for prefix in ['PYMICROGLIA', 'AUTO_ORGANOTYPIC']:
        for kind in ['STORE', 'INDEX', 'DECISIONS']:
            os.environ[prefix+'_'+kind] = str(args.exports/kind.lower())
    os.environ['MPLBACKEND'] = 'Agg'
    if args.source:
        sys.path.insert(0, str(args.source.resolve()))
        os.environ['PYTHONPATH'] = str(args.source.resolve()) + os.pathsep + os.environ.get('PYTHONPATH', '')
    from pymicroglia.run import run_action
    from pymicroglia.visualisation.figures import load
    import pymicroglia
    from pymicroglia._optional import kit
    installed = kit()
    if installed is not None:
        installed.audit.index.DEFAULT_ROOT = args.exports/'run-index'
    package = Path(pymicroglia.__file__).parent
    identities = source_identity(package)
    dependency_roots = {name: Path(importlib.import_module(name).__file__).parent
                        for name in ['circadian_workbench', 'reprofig', 'auto_organotypic', 'analysis_kit']}
    dependencies = {name: {'path': str(root), 'sources': source_identity(root)}
                    for name, root in dependency_roots.items()}
    versions = {name: importlib.metadata.version(name) for name in
                ['PyMicroglia', 'matplotlib', 'pandas', 'numpy', 'reprofig', 'Auto-Organotypic', 'circadian-workbench']}
    options = json.loads(args.options.read_text()) if args.options else {}
    cases = json.loads(args.cases.read_text()) if args.cases else {}
    overrides = {'cell_report_card': {'identity': 1, 'images': 0, 'metrics': ['corrected_mean', 'area_px', 'turnover_index']},
                 'predictability_clock': {'period_hours': 24.}, 'tissue_tectonics': {'reporter_metric': 'corrected_mean'},
                 'mask_change_map': {'identity': 1}, 'cell_time_sculpture': {'identity': 1}, 'cell_trace_card': {'identity': 1}}
    names = args.actions or [key for key, spec in load().items() if args.all or not spec.saved]
    summary = {'profiled': args.profile, 'python': sys.executable, 'package': str(package),
               'direct': args.direct, 'versions': versions, 'sources': identities,
               'dependencies': dependencies, 'required_actions': names, 'cases': []}
    summary['harness_sha256'] = digest(Path(__file__))
    summary['startup_seconds'] = time.perf_counter() - PROCESS_STARTED
    batch_started = time.perf_counter()
    with tracker.stage('Export complete figures', total=len(names)):
        for name in names:
            receipt = {'action': name, 'status': 'failed'}
            params = {'run': args.run, 'stem': args.stem, 'output_dir': args.exports/name,
                      'claim': 'Synthetic saved-recording export performance check; no biological finding.',
                      **overrides.get(name, {}), **options.get(name, {})}
            case = cases.get(name, {})
            if case:
                params.update(run=Path(case['run']), **case.get('options', {}))
            params.update(options.get(name, {}))
            receipt['parameters'] = {k: str(v) if isinstance(v, Path) else v for k, v in params.items()}
            profiler = cProfile.Profile() if args.profile else None
            started = time.perf_counter()
            try:
                with tracker.item(name, output=str(args.exports/name)):
                    # Reporting may wait on the shared job-state lock. That is
                    # instrumentation, not part of the public figure operation.
                    started = time.perf_counter()
                    cpu_started = time.process_time()
                    receipt['started_utc'] = datetime.now(timezone.utc).isoformat()
                    if load()[name].saved:
                        from pymicroglia._results import figure_plans
                        plans = figure_plans(params['run'])
                        matching = [row for row in plans.values() if row['figure'].replace('-', '_') == name]
                        if matching:
                            params['item'] = matching[0]['name']
                    receipt['parameters'] = {k: str(v) if isinstance(v, Path) else v for k, v in params.items()}
                    if args.direct:
                        from pymicroglia.figure_tables import actions
                        call = getattr(actions, name)
                        positional = ()
                    else:
                        call, positional = run_action, (name,)
                    call_params = dict(params)
                    if not args.direct:
                        call_params['output_roots'] = [args.exports/'records'/name]
                    result = profiler.runcall(call, *positional, **call_params) if profiler else call(*positional, **call_params)
                    receipt['seconds'] = time.perf_counter() - started
                    receipt['process_cpu_seconds'] = time.process_time() - cpu_started
                    receipt['finished_utc'] = datetime.now(timezone.utc).isoformat()
                    import psutil
                    memory = psutil.Process().memory_info()
                    receipt['process_peak_rss_bytes'] = getattr(memory, 'peak_wset', memory.rss)
                    receipt['memory_note'] = 'Process lifetime peak through this action on Windows; conservative cumulative upper bound, excludes child processes.'
                    files = [Path(p) for p in result['figures']]
                    assert files and all(p.is_file() and p.stat().st_size for p in files)
                    receipt['status'] = 'measured'
                    receipt['files'] = [str(p) for p in files]
            except Exception:
                receipt['seconds'] = time.perf_counter() - started
                receipt['error'] = traceback.format_exc()
            receipt['outputs'] = {p.relative_to(args.exports/name).as_posix():
                                  {'sha256': digest(p), 'bytes': p.stat().st_size}
                                  for p in (args.exports/name).rglob('*') if p.is_file()}
            if profiler:
                stats = pstats.Stats(profiler)
                receipt['profile'] = sorted([{'file': f, 'line': line, 'function': func, 'calls': nc,
                                             'self_seconds': tt, 'cumulative_seconds': ct}
                                            for (f, line, func), (cc, nc, tt, ct, callers) in stats.stats.items()],
                                           key=lambda row: row['cumulative_seconds'], reverse=True)
            (args.output/(name+'.json')).write_text(json.dumps(receipt, indent=2), encoding='utf-8')
            summary['cases'].append({k: v for k, v in receipt.items() if k not in {'profile', 'outputs'}})
            print(json.dumps({k: receipt[k] for k in ['action', 'status', 'seconds']})+
                  (receipt.get('error', '').splitlines()[-1] if receipt.get('error') else ''), flush=True)
    summary['harness_seconds'] = time.perf_counter() - batch_started
    summary['action_total_seconds'] = sum(row['seconds'] for row in summary['cases'])
    summary['harness_unchanged'] = summary['harness_sha256'] == digest(Path(__file__))
    summary['source_unchanged'] = identities == source_identity(package)
    summary['dependencies_unchanged'] = all(dependencies[name]['sources'] == source_identity(root)
                                          for name, root in dependency_roots.items())
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    from ai_research_assistant.tracking import RunTracker
    with RunTracker('Measure complete figure exports', project='PyMicroglia') as tracker:
        main(tracker)
