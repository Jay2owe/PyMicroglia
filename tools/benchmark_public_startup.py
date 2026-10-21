"""Fresh-process import and complete public-description coverage, before/after."""
import argparse
import hashlib
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


def child(baseline=None):
    start=time.perf_counter()
    if baseline:
        source=Path(__file__).resolve().parents[1]/'src/pymicroglia/__init__.py'
        loader=importlib.machinery.SourceFileLoader('pymicroglia',str(source))
        loader.get_code=lambda name:compile(baseline.read_text(encoding='utf-8'),str(source),'exec')
        spec=importlib.util.spec_from_file_location('pymicroglia',source,loader=loader,
                                                  submodule_search_locations=[str(source.parent)])
        package=importlib.util.module_from_spec(spec);sys.modules['pymicroglia']=package
        loader.exec_module(package)
    else:
        import pymicroglia as package
    imported=time.perf_counter()-start; modules=len(sys.modules)
    parameter_start=time.perf_counter(); parameter=package.ParamBlock
    parameter_seconds=time.perf_counter()-parameter_start
    from pymicroglia import catalogue
    rows=[]
    for name in catalogue.action_names():
        action_start=time.perf_counter(); descriptor=package.describe(name)
        rows.append({'action':name,'describe_seconds':time.perf_counter()-action_start,
                     'descriptor_sha256':hashlib.sha256(json.dumps(descriptor,sort_keys=True,default=str).encode()).hexdigest(),
                     'pending':descriptor.get('pending'), 'analysis_execution':'not benchmarked',
                     'method':catalogue.action(name)['method']})
    return {'import_seconds':imported,'import_module_count':modules,'parameter_seconds':parameter_seconds,
            'complete_catalogue_seconds':time.perf_counter()-start,'actions':rows}


def catalogue(out):
    from pymicroglia import knowledge
    spec=importlib.util.spec_from_file_location('pymicroglia._speed_knowledge',out/'baseline/knowledge.py')
    before=importlib.util.module_from_spec(spec);spec.loader.exec_module(before)
    rows=[]
    for iteration in range(3):
        for label,module in ([('before',before),('after',knowledge)] if iteration%2==0 else [('after',knowledge),('before',before)]):
            start=time.perf_counter();result=module.describe()
            row={'label':label,'seconds':time.perf_counter()-start,'actions':len(result['actions']),
                 'sha256':hashlib.sha256(json.dumps(result,sort_keys=True,default=str).encode()).hexdigest()}
            rows.append(row);print(row,flush=True)
    (out/'describe-catalogue.json').write_text(json.dumps(rows,indent=2))
    assert len({row['sha256'] for row in rows})==1


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--child',action='store_true')
    parser.add_argument('--baseline',type=Path);parser.add_argument('--out',type=Path)
    parser.add_argument('--catalogue',action='store_true')
    args=parser.parse_args()
    if args.child:print(json.dumps(child(args.baseline)))
    elif args.catalogue:catalogue(args.out)
    else:
        samples=[]
        for i in range(3):
            for label in (['before','after'] if i%2==0 else ['after','before']):
                command=[shutil.which('python'),str(Path(__file__).resolve()),'--child']
                if label=='before':command+=['--baseline',str(args.out/'baseline/package_init.py')]
                row=json.loads(subprocess.check_output(command,text=True,timeout=120))
                samples.append({'label':label,**row})
                print(label,{k:v for k,v in row.items() if k!='actions'},flush=True)
        (args.out/'public-actions.json').write_text(json.dumps(samples,indent=2))
        reference={r['action']:r['descriptor_sha256'] for r in samples[0]['actions']}
        assert all({r['action']:r['descriptor_sha256'] for r in sample['actions']}==reference for sample in samples)
