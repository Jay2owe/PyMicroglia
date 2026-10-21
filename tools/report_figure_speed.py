"""Build the catalogue timing table and static function/profile coverage map."""
import argparse
import ast
import csv
import json
from pathlib import Path
import statistics
from itertools import chain


def retain_source_changes(root, before, after):
    """Keep recoverable source deltas before retiring duplicate benchmark trees."""
    import difflib
    import hashlib
    import zipfile
    baseline, candidate = Path(before['package']), Path(after['package'])
    archive = root/'figure-source-changes.zip'
    if not baseline.exists() or not candidate.exists():
        if not archive.is_file():
            raise FileNotFoundError('Frozen source trees and recovery archive are both missing')
        with zipfile.ZipFile(archive) as saved:
            assert saved.testzip() is None
        return
    changed = sorted(name for name in before['sources'].keys() | after['sources'].keys()
                     if before['sources'].get(name) != after['sources'].get(name))
    manifest = {'changed_paths': changed, 'versions': {}, 'note':
                'The complete baseline source inventory is archived, with changed-file overlays for candidate revisions. '
                'Dependency identities remain in trial summaries. Reconstruct in a separate folder; never replace a live checkout wholesale.'}
    stages = {'baseline': baseline, candidate.parent.name: candidate}
    for label in ['candidate1', 'candidate2', 'candidate3', 'candidate4']:
        folder = candidate.parent.parent/label/'pymicroglia'
        if label not in stages and folder.exists():
            stages[label] = folder
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as saved:
        for label, folder in stages.items():
            hashes = {}
            for name in sorted(before['sources']) if label == 'baseline' else changed:
                file = folder/name
                if file.is_file():
                    blob = file.read_bytes()
                    hashes[name] = hashlib.sha256(blob).hexdigest()
                    saved.writestr(label+'/'+name, blob)
            manifest['versions'][label] = hashes
        verification_files = [
            'tools/'+name for name in [
                'audit_figure_fit_reuse.py', 'benchmark_figure_exports.py',
                'benchmark_figure_maps.py', 'benchmark_figure_residuals.py',
                'benchmark_figure_workers.py', 'check_figure_export_parity.py',
                'check_figure_svg_parity.py', 'check_figure_speed_completion.py',
                'finish_figure_speed_measurements.py', 'finish_figure_speed_validation.py',
                'prepare_figure_speed_fixtures.py', 'report_figure_speed.py',
                'verify_figure_exports.py', 'motion_catalogue.py',
                'diagnose_figure_export_regressions.py', 'close_figure_speed_pass.py']]
        verification_files += ['tests/'+name for name in [
            'test_figure_export_speed.py', 'test_figure_export_checks.py',
            'test_figure_map_speed.py', 'test_pipeline_table_speed.py', 'test_trace_workers.py']]
        manifest['verification_sources'] = {}
        for name in verification_files:
            blob = Path(name).read_bytes()
            saved.writestr('verification/'+name, blob)
            manifest['verification_sources'][name] = hashlib.sha256(blob).hexdigest()
        saved.writestr('manifest.json', json.dumps(manifest, indent=2))
        assert saved.testzip() is None
    patches = []
    for name in changed:
        old, new = baseline/name, candidate/name
        patches.extend(difflib.unified_diff(old.read_text(encoding='utf-8-sig').splitlines(True) if old.exists() else [],
                       new.read_text(encoding='utf-8-sig').splitlines(True) if new.exists() else [],
                       fromfile='before/pymicroglia/'+name, tofile='after/pymicroglia/'+name))
    (root/'figure-source-changes.diff').write_text(''.join(patches), encoding='utf-8')
    (root/'source-recovery.json').write_text(json.dumps({
        **manifest, 'archive': archive.name, 'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest()
    }, indent=2))


def final_report(args, functions):
    """Retain early samples, but decide from the final frozen candidate."""
    root = args.evidence
    before = json.loads((root/('full-before3' if args.revision == 4 else 'full-before4')/'summary.json').read_text())
    after = json.loads((root/f'full-after{args.revision}/summary.json').read_text())
    retain_source_changes(root, before, after)
    trials = {}
    first_repeat = 2*(args.revision-2)
    before_trials = [f'repeat-before{first_repeat}', f'repeat-before{first_repeat+1}']
    after_trials = [f'repeat-after{first_repeat}', f'repeat-after{first_repeat+1}']
    for name in before_trials+after_trials:
        trials[name] = json.loads((root/name/'summary.json').read_text())
    parity = json.loads((root/'final-data-parity.json').read_text())
    visuals = json.loads((root/'final-svg-parity.json').read_text())
    by_data = {row['action']: row['data_parity'] for row in parity['cases']}
    by_visual = {}
    for row in visuals['cases']:
        by_visual[row['action']] = by_visual.get(row['action'], True) and row['parity']
    rows = []
    reviews = []
    for name in before['required_actions']:
        first, second = [next(row for row in summary['cases'] if row['action'] == name)
                         for summary in [before, after]]
        repeated = {key: [row for row in summary['cases'] if row['action'] == name]
                    for key, summary in trials.items()}
        repeated = {key: value[0] for key, value in repeated.items() if value}
        a = [first['seconds']]
        b = [second['seconds']]
        if repeated:
            a = [repeated[key]['seconds'] for key in before_trials]
            b = [repeated[key]['seconds'] for key in after_trials]
        old, new = statistics.median(a), statistics.median(b)
        regression = new-old > max(.1, .05*old)
        row = {'action': name, 'executed': first['status'] == second['status'] == 'measured',
               'before_seconds': a, 'after_seconds': b, 'before_median': old, 'after_median': new,
               'speedup': old/new, 'material_regression': regression,
               'parity': by_data.get(name, False) and by_visual.get(name, False),
               'full_batch_before_seconds': first['seconds'], 'full_batch_after_seconds': second['seconds'],
               'before_cpu_seconds': first.get('process_cpu_seconds'),
               'after_cpu_seconds': second.get('process_cpu_seconds'),
               'peak_rss_bytes': max(first.get('process_peak_rss_bytes', 0), second.get('process_peak_rss_bytes', 0)),
               'sampling_note': 'Two reverse-order repeat pairs' if repeated else 'One final full-batch pair; earlier raw trials retained separately'}
        rows.append(row)
        if regression:
            reviews.append({'action': name, 'before_seconds': a, 'after_seconds': b, 'status': 'open'})
    totals = {'before_seconds': before['action_total_seconds'], 'after_seconds': after['action_total_seconds'],
              'before_cpu_seconds': sum(r.get('process_cpu_seconds', 0) for r in before['cases']),
              'after_cpu_seconds': sum(r.get('process_cpu_seconds', 0) for r in after['cases']),
              'before_peak_rss_bytes': max(r['process_peak_rss_bytes'] for r in before['cases']),
              'after_peak_rss_bytes': max(r['process_peak_rss_bytes'] for r in after['cases'])}
    totals['speedup'] = totals['before_seconds']/totals['after_seconds']
    all_summaries = [before, after, *trials.values()]
    stable = all(s.get('source_unchanged') and s.get('dependencies_unchanged') and s.get('harness_unchanged')
                 for s in all_summaries)
    stable = stable and all(s['dependencies'] == after['dependencies'] for s in all_summaries)
    from benchmark_figure_exports import source_identity
    stable = stable and source_identity(args.package) == after['sources']
    result = {'cases': rows, 'full_batch': totals, 'source_stable': stable,
              'scope': 'All 91 declared static figure families on completed synthetic inputs; one saved page per family, all its default views.',
              'limits': 'Small recording fixture. Earlier interrupted and cold trials are retained, not silently averaged into the final candidate. Process memory peaks are cumulative upper bounds, not per-action allocations.'}
    (root/'final-timings.json').write_text(json.dumps(result, indent=2))
    with (root/'final-timings.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (root/'regression-review.json').write_text(json.dumps(reviews, indent=2))
    scope = {'required': ['action:'+row['action'] for row in rows]}
    coverage = {'operations': [{'id': 'action:'+row['action'], 'kind': 'execution',
                 'status': 'measured' if row['executed'] else 'failed',
                 'before_seconds': row['before_seconds'], 'after_seconds': row['after_seconds'],
                 'peak_rss_bytes': row['peak_rss_bytes'], 'parity': row['parity'],
                 'receipt': 'final-timings.json'} for row in rows]}
    (root/'scope.json').write_text(json.dumps(scope, indent=2))
    (root/'coverage-evidence.json').write_text(json.dumps(coverage, indent=2))
    print(json.dumps({'full_batch': totals, 'source_stable': stable, 'remaining_regressions': reviews}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', required=True, type=Path)
    parser.add_argument('--package', type=Path, default=Path('src/pymicroglia'))
    parser.add_argument('--final', action='store_true')
    parser.add_argument('--revision', type=int, choices=[4, 5], default=5)
    args = parser.parse_args()
    profiles = {}
    for receipt in chain((args.evidence/'catalogue-profile').glob('*.json'),
                         *(folder.glob('*.json') for folder in args.evidence.glob('final-profile*')
                           if folder.is_dir())):
        document = json.loads(receipt.read_text())
        for row in document.get('profile', []):
            path = row['file'].replace('\\', '/')
            if '/pymicroglia/' not in path:
                continue
            key = (path.split('/pymicroglia/', 1)[1], row['function'])
            aggregate = profiles.setdefault(key, {'calls': 0, 'self_seconds': 0, 'actions': set()})
            aggregate['calls'] += row['calls']
            aggregate['self_seconds'] += row['self_seconds']
            aggregate['actions'].add(document['action'])
    functions = []
    for path in args.package.rglob('*.py'):
        relative = path.relative_to(args.package).as_posix()
        # Include all figure-preparation/rendering functions and observed
        # package helpers, without implying the whole analysis app was tested.
        if not relative.startswith(('figure_tables/', 'visualisation/')) and not any(
                filename == relative for filename, _ in profiles):
            continue
        module = ast.parse(path.read_text(encoding='utf-8-sig'))
        for node in ast.walk(module):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            observed = profiles.get((relative, node.name))
            changed = (relative, node.name) in {
                ('pipelines/_tables.py', 'read'), ('pipelines/_tables.py', '_cell_decoder'),
                ('pipelines/_tables.py', 'decode'), ('figure_tables/actions.py', 'draw'),
                ('figure_tables/territory_values.py', 'cumulative_occupancy'),
                ('figure_tables/tissue.py', 'prepare'), ('figure_tables/recurrence.py', 'prepare'),
                ('figure_tables/all_cell_traces.py', 'trace_data'),
                ('figure_tables/cell_grid.py', 'prepare'),
                ('figure_tables/trace_workers.py', 'parallel_traces'),
                ('visualisation/_delivery.py', 'native_vector_images'),
                ('visualisation/panels/__init__.py', 'save')}
            functions.append({'file': relative, 'line': node.lineno, 'function': node.name,
                              'observed': observed is not None,
                              'calls': observed['calls'] if observed else None,
                              'self_seconds_profiled': observed['self_seconds'] if observed else None,
                              'actions': sorted(observed['actions']) if observed else [],
                              'disposition': 'improved' if changed else
                                  'unobserved' if not observed else
                                  'required exact output serialization' if relative == 'visualisation/bundle.py'
                                  else 'low captured self cost; callee costs reviewed separately'
                                  if observed['self_seconds'] < .05 else
                                  'scientific preparation retained; fit and data costs reviewed in residual decisions'
                                  if relative.startswith('figure_tables/') else
                                  'required current layout; alternative renderer checked in residual decisions'
                                  if relative.startswith('visualisation/') else
                                  'shared validation, gateway or provenance work retained; measured costs remain visible',
                              'coverage_note': 'Function-name coverage across baseline and candidate diagnostics; these aggregate self times are not final-candidate latency. Nested names may share a profile label.'
                              if observed else
                              'Exercised by separate worker parity/growth checks; child functions are outside parent profiles.'
                              if relative == 'figure_tables/trace_workers.py' else
                              'No captured calls in the declared default static-figure profiles; alternate options, video/legacy helpers and error branches remain unprofiled. No timing inferred.'})
    (args.evidence/'function-coverage.json').write_text(json.dumps(functions, indent=2), encoding='utf-8')
    if args.final:
        final_report(args, functions)
        return
    summaries = {}
    for trial in ['before1', 'after1', 'after2', 'before2']:
        path = args.evidence/('full-'+trial)/'summary.json'
        if path.exists():
            summaries[trial] = json.loads(path.read_text())
    if len(summaries) != 4:
        print('Function coverage written; complete timing trials still pending.')
        return
    names = summaries['before1']['required_actions']
    rows = []
    for name in names:
        samples = {trial: next(r for r in summary['cases'] if r['action'] == name)
                   for trial, summary in summaries.items()}
        before = [samples[k]['seconds'] for k in ['before1', 'before2']]
        after = [samples[k]['seconds'] for k in ['after1', 'after2']]
        a, b = statistics.median(before), statistics.median(after)
        rows.append({'action': name, 'complete': all(r['status'] == 'measured' for r in samples.values()),
                     'before_seconds': before, 'after_seconds': after,
                     'before_median': a, 'after_median': b, 'speedup': a/b,
                     'saved_seconds': a-b,
                     'needs_regression_review': b-a > max(.1, .05*a)})
    report = {'cases': rows,
              'action_seconds_by_trial': {k: sum(r['seconds'] for r in v['cases']) for k, v in summaries.items()},
              'unchanged_sources': all(v.get('source_unchanged') and v.get('dependencies_unchanged')
                                       for v in summaries.values()),
              'profiled_timing_warning': 'Function self times are diagnostic; paired action timings are uninstrumented.',
              'scope': '91 declared figure actions on completed synthetic recording/pipeline inputs; UI excluded.'}
    (args.evidence/'timing-comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with (args.evidence/'timing-comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k: v for k, v in report.items() if k != 'cases'}))


if __name__ == '__main__':
    main()
