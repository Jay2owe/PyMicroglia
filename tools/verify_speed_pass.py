"""Run the final affected checks against live installed project dependencies."""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--auto',required=True,type=Path)
    args=parser.parse_args()
    folder=ROOT/'development/speed-pass-20261011/complete/final-validation'
    folder.mkdir(exist_ok=True)
    env=dict(os.environ)
    env.pop('PYTHONPATH',None)
    env['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'
    tests=[
        'tests/test_lazy_public_api.py','tests/test_catalogue_speed.py',
        'tests/test_viewer_speed.py','tests/test_ui_exploration_images.py',
        'tests/test_ui_exploration_previews.py','tests/test_measure_index_batch.py',
        'tests/test_measure_analysis_scope.py','tests/test_action_batch_capture.py',
        'tests/test_photon_video_caption.py','tests/test_ui_group_comparison.py',
        'tests/test_state_review.py',
        'tests/test_period_method_audit.py::test_native_figure_draws_every_selected_filter_without_refitting',
    ]
    pytest_code="import sys,shutil;sys.executable=shutil.which('python');import pytest;raise SystemExit(pytest.main("+repr(tests+['-q'])+"))"
    verify_code="import sys,shutil,runpy;sys.executable=shutil.which('python');sys.path.insert(0,'scripts');sys.argv=['scripts/verify.py','--json'];runpy.run_path('scripts/verify.py',run_name='__main__')"
    jobs=[('pymicroglia-focused',ROOT,['-c',pytest_code]),
          ('pymicroglia-contract',ROOT,['-c',verify_code]),
          ('auto-organotypic-contract',args.auto,['-c',verify_code])]
    records=[]
    for name,cwd,arguments in jobs:
        started=time.time()
        with (folder/(name+'.log')).open('w',encoding='utf-8') as stream:
            result=subprocess.run([shutil.which('python'),*arguments],cwd=cwd,env=env,stdout=stream,stderr=subprocess.STDOUT)
        records.append({'check':name,'cwd':str(cwd),'command':arguments,
                        'exit_code':result.returncode,'seconds':time.time()-started,
                        'log':name+'.log','dependencies':'live installed projects; benchmark PYTHONPATH removed'})
        (folder/'receipt.json').write_text(json.dumps(records,indent=2))
        print(name+' exit='+str(result.returncode),flush=True)
    return int(any(row['exit_code'] for row in records))


if __name__=='__main__':raise SystemExit(main())
