"""Run the declared performance checks sequentially, preserving each receipt."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import psutil
from ai_research_assistant.tracking import RunTracker


def main():
    p=argparse.ArgumentParser()
    p.add_argument('manifest',type=Path);a=p.parse_args()
    manifest=json.loads(a.manifest.read_text())
    root=Path(manifest['root']).resolve()
    receipt=a.manifest.with_suffix('.results.json')
    if receipt.exists():raise FileExistsError(receipt)
    results=[]
    with RunTracker('Sequential complete tracking performance validation',project='PyMicroglia performance') as run:
        wait=manifest.get('wait_for')
        if wait:
            with run.stage('Finish existing validation',total=1):
                with run.item('Existing complete identity-history group'):
                    try:
                        process=psutil.Process(wait['pid'])
                        if abs(process.create_time()-wait['created'])>.01:raise RuntimeError('Worker process identity changed')
                        if wait['command_token'] not in ' '.join(process.cmdline()):raise RuntimeError('Unexpected worker command')
                        while process.is_running():
                            try:process.wait(timeout=15)
                            except psutil.TimeoutExpired:continue
                    except psutil.NoSuchProcess:pass
        with run.stage('Complete scoped equivalence checks',total=len(manifest['checks'])):
            for check in manifest['checks']:
                try:
                    with run.item(check['name'],output=check['report']):
                        started=time.time()
                        command=[sys.executable if value=='{python}' else value for value in check['command']]
                        log=root/check['log']
                        if log.exists():raise FileExistsError(log)
                        with log.open('x') as handle:
                            result=subprocess.run(command,cwd=root,env=run.environment(),stdout=handle,stderr=subprocess.STDOUT)
                        row=dict(name=check['name'],exit_code=result.returncode,elapsed_seconds=time.time()-started,
                            report=check['report'],log=check['log'])
                        results.append(row)
                        receipt.write_text(json.dumps(dict(status='running',checks=results),indent=2)+'\n')
                        if result.returncode:raise RuntimeError(f"Check failed: {check['name']}; see {log}")
                except BaseException:
                    receipt.write_text(json.dumps(dict(status='stopped_on_failure',checks=results),indent=2)+'\n')
                    raise
    receipt.write_text(json.dumps(dict(status='complete',checks=results),indent=2)+'\n')


if __name__=='__main__':main()
