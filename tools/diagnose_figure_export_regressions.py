"""Alternate old and current implementations in one process to test small regressions.

This is a causal diagnostic, not a frozen-package acceptance benchmark. It
restores precisely the changed functions reached by the four selected actions;
unchanged dependencies, interpreter, import caches and input files are shared.
"""
import argparse
from contextlib import nullcontext
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys

from ai_research_assistant.tracking import RunTracker
import benchmark_figure_exports as benchmark


ACTIONS = ['metric_rhythm_matrix', 'radial_occupancy_rhythms',
           'pixel_fate_flow', 'measurement_relationship_populations']


def old_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scratch', required=True, type=Path)
    parser.add_argument('--evidence', required=True, type=Path)
    args = parser.parse_args()
    os.environ['MPLBACKEND'] = 'Agg'
    from pymicroglia.figure_tables import actions
    from pymicroglia.pipelines import _tables
    from pymicroglia.visualisation import _delivery
    baseline = args.scratch/'baseline/pymicroglia'
    previous_actions = old_module(baseline/'figure_tables/actions.py',
                                 'pymicroglia.figure_tables._speed_control')
    previous_tables = old_module(baseline/'pipelines/_tables.py',
                                'pymicroglia.pipelines._speed_control')
    current = (actions.draw, _tables.read, _delivery.native_vector_images)
    def previous_draw(*positional, workers=1, **options):
        # Current catalogue supplies the new runtime default to every action.
        # The four control figures do not use workers.
        assert workers == 1
        return previous_actions.draw(*positional, **options)

    control = (previous_draw, previous_tables.read,
               lambda figure, destination: nullcontext())
    rows = []
    # Two complete ABBA sequences reduce both import bias and monotonic drift.
    sequence = ['control', 'current', 'current', 'control'] * 2
    with RunTracker('Diagnose four small figure timing regressions', project='PyMicroglia') as parent:
        try:
            for index, mode in enumerate(sequence, 1):
                actions.draw, _tables.read, _delivery.native_vector_images = (
                    control if mode == 'control' else current)
                name = f'causal-v2-{index}-{mode}'
                sys.argv = ['benchmark_figure_exports.py', '--output', str(args.evidence/name),
                            '--exports', str(args.scratch/name), '--actions', *ACTIONS,
                            '--cases', str(args.evidence/'prior-cases.json'),
                            '--options', str(args.evidence/'fixture-options.json')]
                with RunTracker('Same-process export diagnostic: '+name,
                                project='PyMicroglia', root=parent.runs.root) as child:
                    benchmark.main(child)
                summary = json.loads((args.evidence/name/'summary.json').read_text())
                if any(row['status'] != 'measured' for row in summary['cases']):
                    raise RuntimeError('Incomplete diagnostic: '+name)
                rows.append({'mode': mode, 'receipt': name+'/summary.json',
                             'cases': summary['cases']})
        finally:
            actions.draw, _tables.read, _delivery.native_vector_images = current
    result = {
        'diagnostic_only': True,
        'intervention': 'Restore baseline figure draw and pipeline-table read; disable native SVG images. All other runtime code and imports shared. These are all changed functions reachable by the four selected actions.',
        'sequence': sequence, 'trials': rows,
        'baseline_functions': {
            str(path.relative_to(baseline)): benchmark.digest(path)
            for path in [baseline/'figure_tables/actions.py', baseline/'pipelines/_tables.py']},
        'summary': [],
    }
    for action in ACTIONS:
        samples = {mode: [next(case for case in row['cases'] if case['action'] == action)
                          for row in rows if row['mode'] == mode]
                   for mode in ['control', 'current']}
        result['summary'].append({
            'action': action,
            **{mode+'_'+metric: [case[metric] for case in cases]
               for mode, cases in samples.items()
               for metric in ['seconds', 'process_cpu_seconds']},
            'median_change_seconds': statistics.median(c['seconds'] for c in samples['current']) -
                                     statistics.median(c['seconds'] for c in samples['control']),
        })
    (args.evidence/'causal-regression-diagnostic.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
