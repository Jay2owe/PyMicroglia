"""Build and exercise the optional test from isolated wheel contents.

Pass a session-owned scratch directory. Sources are copied before building so
the active checkouts receive no build/egg-info folders. No publication or pip
installation is performed; dependencies use the existing test environment.
"""
import argparse,json,os,shutil,subprocess,sys,zipfile
from pathlib import Path


def main(scratch,report):
    scratch=Path(scratch).resolve();repo=Path(__file__).resolve().parents[1]
    workbench=repo.parent.parent/'CircadianWorkbench'
    root=scratch/'wheel_check';root.mkdir(parents=True,exist_ok=False)
    wheels=root/'wheels';wheels.mkdir()
    for name,source in [('pymicroglia',repo),('workbench',workbench)]:
        clone=root/name;clone.mkdir()
        for file in ['pyproject.toml','MANIFEST.in','LICENSE','README.md','PYPI_README.md','README_AI.md','circadian_workbench_context.json']:
            path=source/file
            if path.is_file():shutil.copy2(path,clone/file)
        shutil.copytree(source/'src',clone/'src',ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.pyo','.git','*.egg-info'))
        if (source/'scripts/generate_agent_context.py').is_file():
            (clone/'scripts').mkdir();shutil.copy2(source/'scripts/generate_agent_context.py',clone/'scripts/generate_agent_context.py')
        completed=subprocess.run([sys.executable,'-m','build','--wheel','--no-isolation','--outdir',str(wheels)],cwd=clone,capture_output=True,text=True)
        (root/f'{name}_build.log').write_text(completed.stdout+'\n'+completed.stderr,encoding='utf8')
        if completed.returncode:raise RuntimeError(name+' wheel build failed; see owned scratch log')
    installed=root/'installed';installed.mkdir()
    found=sorted(wheels.glob('*.whl'))
    if len(found)!=2:raise ValueError('Expected two wheels')
    for wheel in found:
        with zipfile.ZipFile(wheel) as z:
            for entry in z.namelist():
                target=(installed/entry).resolve()
                if not target.is_relative_to(installed):raise ValueError('Wheel entry escaped extraction root')
            z.extractall(installed)
    script=r'''
import sys,json
from pathlib import Path
root=Path(sys.argv[1]).resolve();sys.path.insert(0,str(root))
import pymicroglia as pm
import circadian_workbench as cw
import numpy as np
assert Path(pm.__file__).resolve().is_relative_to(root)
assert Path(cw.__file__).resolve().is_relative_to(root)
assert 'fast_simulation' not in sys.modules
assert pm.describe('test_simulation_rhythm')['pending'] is False
t=np.arange(81)*.5
outlines=np.zeros((len(t),50,50),np.uint16);outlines[:,20:30,20:30]=1
pixels=np.full((len(t),50,50),10.)
for i in range(len(t)):
    pixels[i,20:30,20:30]+=20+4*np.cos(2*np.pi*t[i]/12.3)
outlines[10:13]=0
result=pm.test_simulation_rhythm(pixels,outlines=outlines,times_h=t,
    control={'findings':['known-input synthetic instrumental control']},
    selection='fixed',pattern='uniform',blur_pixels=2.5,options={'surrogates':9})
row=result['rows'][0]
assert abs(row['period_hours']-12.3)<.15
assert np.isnan(row['prediction'][10:13]).all()
assert 'fast_simulation' not in sys.modules
assert not any('reporter_dynamics_simulation' in str(getattr(m,'__file__','')) for m in list(sys.modules.values()))
print(json.dumps(dict(status='passed',period_hours=row['period_hours'],public_action=True,detected_outline_gaps=True,packages_loaded_from_wheels=True,no_development_imports=True)))
'''
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
    tested=subprocess.run([sys.executable,'-c',script,str(installed)],cwd=root,env=env,capture_output=True,text=True)
    (root/'wheel_execution.log').write_text(tested.stdout+'\n'+tested.stderr,encoding='utf8')
    if tested.returncode:raise RuntimeError('Isolated wheel execution failed; see owned scratch log')
    value=json.loads(tested.stdout.strip().splitlines()[-1]);value['wheels']=[p.name for p in found]
    Path(report).write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    print(json.dumps(value))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scratch',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();main(a.scratch,a.report)
