"""Restore unavailable benchmark inputs through their native public producers."""
import argparse
import json
import os
from pathlib import Path
import sys
import types

from ai_research_assistant.tracking import RunTracker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scratch', required=True, type=Path)
    parser.add_argument('--cases', required=True, type=Path)
    args = parser.parse_args()
    os.environ['PYMICROGLIA_STORE'] = str(args.scratch/'store')
    os.environ['PYMICROGLIA_INDEX'] = str(args.scratch/'index')
    os.environ['MPLBACKEND'] = 'Agg'
    tests = types.ModuleType('tests')
    tests.__path__ = [str(Path(__file__).resolve().parents[1]/'tests')]
    sys.modules['tests'] = tests
    from tests.pipelines.audit import demo
    from pymicroglia.run import run_action
    from pymicroglia._results import figure_plans, read_document
    cases = json.loads(args.cases.read_text())
    with RunTracker('Restore saved figure benchmark inputs', project='PyMicroglia') as tracker:
        with tracker.stage('Prepare completed audit', total=1):
            with tracker.item('Native synthetic audit', output=str(args.scratch/'audit')):
                source = demo.prepare(args.scratch/'audit-inputs')
                result = run_action('method_audit', run=source/'run',
                                    pipeline_request=read_document(source/'audit.json'),
                                    presentation=read_document(source/'presentation.json'),
                                    output_dir=args.scratch/'audit',
                                    claim='Synthetic figure export verification; no biological finding.')
                if not result.successful:
                    raise ValueError({key: value.outcome.reason for key, value in result.results.items()
                                      if value.outcome.status != 'complete'})
                run = args.scratch/'audit/synthetic-audit'
                names = {p['figure'].replace('-', '_') for p in figure_plans(run).values()}
                for key in [k for k in cases if k.startswith('audit_')]:
                    if key not in names:
                        raise ValueError('Missing native audit figure plan: '+key)
                    cases[key]['run'] = str(run.resolve())
        cases['neighbour_coordination']['options']['stem'] = 'synthetic'
        cases['state_histories']['run'] = str(Path(
            'development/speed-pass-20261011/complete/dataset-before/dataset/states/states').resolve())
        cases['state_histories']['options']['stem'] = None
        args.cases.write_text(json.dumps(cases, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
