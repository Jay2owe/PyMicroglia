"""Freeze editable analysis dependencies for isolated local benchmark processes.

Development-only snapshots are not imported by or included in the shipped app.
Verify the source did not change during copying; never edit the source checkout.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def inventory(root):
    return {p.relative_to(root).as_posix():digest(p) for p in root.rglob('*')
            if p.is_file() and '__pycache__' not in p.parts and p.suffix not in {'.pyc','.pyo'}}


def main():
    out=ROOT/'development/speed-pass-20261011/complete/frozen-dependencies'
    if out.exists():raise FileExistsError(out)
    sources={name:Path(importlib.util.find_spec(name).origin).parent
             for name in ('circadian_workbench','analysis_kit','reprofig')}
    sizes={name:sum(p.stat().st_size for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
           for name,folder in sources.items()}
    print(json.dumps({'source_bytes':sizes,'free_bytes':shutil.disk_usage(ROOT).free}),flush=True)
    if sum(sizes.values())>500*1024**2:raise OSError('Dependency snapshot exceeds bounded 500 MiB preparation')
    if shutil.disk_usage(ROOT).free<sum(sizes.values())+1024**3:raise OSError('Insufficient snapshot headroom')
    out.mkdir()
    receipt={'schema_version':1,'packages':{},'purpose':'Local benchmark isolation only; not a production dependency override'}
    for name,source in sources.items():
        before=inventory(source)
        shutil.copytree(source,out/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.pyo'))
        copied=inventory(out/name);after=inventory(source)
        if before!=copied or before!=after:raise ValueError('Source changed during snapshot: '+name)
        receipt['packages'][name]={'source':str(source),'bytes':sizes[name],'files':copied}
    (out/'snapshot.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print('Verified dependency snapshot: '+str(out),flush=True)


if __name__=='__main__':main()
