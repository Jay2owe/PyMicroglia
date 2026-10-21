"""Verify figure changes, distinguish existing fixture failures, and run the project gate."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

from ai_research_assistant.tracking import RunTracker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    historical = ['tests/test_figures_spatial.py', 'tests/test_figures_final_ports.py',
                  'tests/test_project_contracts.py', 'tests/test_ui_catalog.py']
    focused = ['tests/test_figure_export_speed.py', 'tests/test_figure_map_speed.py',
               'tests/test_pipeline_table_speed.py', 'tests/test_figure_export_checks.py',
               'tests/test_trace_workers.py']
    python = shutil.which('python')
    bootstrap = ("from pathlib import Path; import sys,types,shutil; sys.executable=shutil.which('python'); "
                 "testpackage=types.ModuleType('tests'); testpackage.__path__=[str(Path('tests').resolve())]; "
                 "sys.modules['tests']=testpackage; ")
    preload = (bootstrap +
               f"frozen=Path({str(args.baseline.resolve())!r}); sys.path.insert(0,str(frozen)); "
               "import pymicroglia; assert Path(pymicroglia.__file__).resolve().is_relative_to(frozen); "
               "print('Verified baseline:',pymicroglia.__file__,flush=True); ")
    jobs = [
        ('baseline-fixtures', ['-c', preload + f"import pytest; raise SystemExit(pytest.main({historical+['-q','-o','pythonpath=']!r}))"]),
        ('candidate-focused', ['-c', bootstrap + f"import pytest; raise SystemExit(pytest.main({historical+focused+['-q']!r}))"]),
        ('project-contract', ['-c', "import sys,shutil,runpy; sys.executable=shutil.which('python'); "
          "sys.path.insert(0,'scripts'); sys.argv=['scripts/verify.py','--json']; runpy.run_path('scripts/verify.py',run_name='__main__')"]),
    ]
    rows = []
    with RunTracker('Verify figure export performance changes', project='PyMicroglia') as tracker:
        with tracker.stage('Focused and required checks', total=len(jobs)):
            for name, arguments in jobs:
                with tracker.item(name):
                    env = {**os.environ, **tracker.environment(), 'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1'}
                    env.pop('PYTHONPATH', None)
                    log = args.output/(name+'.log')
                    started = time.perf_counter()
                    with log.open('w', encoding='utf-8') as stream:
                        completed = subprocess.run([python, *arguments], env=env, stdout=stream, stderr=subprocess.STDOUT)
                    output = log.read_text(encoding='utf-8')
                    rows.append({'check': name, 'exit_code': completed.returncode,
                                 'seconds': time.perf_counter()-started, 'command': [python, *arguments],
                                 'failed_tests': re.findall(r'^FAILED (\S+)', output, re.MULTILINE),
                                 'summary': re.findall(r'.*\d+ passed.*', output)[-1:],
                                 'log': str(log)})
                    print(name, completed.returncode, flush=True)
                    (args.output/'receipt.json').write_text(json.dumps(rows, indent=2))
    baseline, candidate, contract = rows
    # Existing failures remain visible; a matching failure set is not a claim
    # that the entire historical suite passed.
    result = {'checks': rows, 'new_test_failures': sorted(set(candidate['failed_tests'])-set(baseline['failed_tests'])),
              'baseline_failures_reproduced': baseline['failed_tests'] == candidate['failed_tests'],
              'contract_passed': contract['exit_code'] == 0,
              'tests_executed': all(row['summary'] and row['exit_code'] in (0, 1) for row in [baseline, candidate])}
    (args.output/'verification.json').write_text(json.dumps(result, indent=2))
    return int(bool(result['new_test_failures']) or not result['contract_passed'] or not result['tests_executed'])


if __name__ == '__main__':
    raise SystemExit(main())
