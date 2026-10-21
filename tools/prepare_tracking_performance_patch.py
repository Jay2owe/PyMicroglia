"""Prepare reviewed performance changes without replacing live source consumers."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();previous=json.loads(a.previous.read_text())
    home=a.previous.parent;old=(home/previous['candidate']).resolve();candidate=a.candidate.resolve()
    package=Path(__file__).resolve().parents[1]
    auto=package.parent.parent/'Auto-Organotypic'
    sources={}
    for target,record in previous['sources'].items():
        sources[Path(target)]=(candidate/Path(record['candidate']).relative_to(old),
                              {record['before_sha256'],record['after_sha256']})
    composition=json.loads((candidate/'composition.json').read_text())
    for relative,digest in composition['sources'].items():
        relative=Path(relative)
        if relative.parent!=Path('pymicroglia/tracking') or relative.suffix!='.py':continue
        base_digest=composition['base_sources'].get(str(relative))
        if base_digest==digest:continue
        target=package/'src'/relative
        if target not in sources:
            sources[target]=(candidate/relative,{base_digest})
    changes={};conflicts=[];patches={'pymicroglia':[],'auto_organotypic':[]}
    for target,(source,allowed) in sources.items():
        before=sha(target);after=sha(source)
        if after is None:raise FileNotFoundError(source)
        if before not in allowed|{after}:
            conflicts.append(str(target));continue
        changes[str(target)]=dict(before_sha256=before,after_sha256=after,candidate=str(source))
        if before==after:continue
        kind='pymicroglia' if target.is_relative_to(package) else 'auto_organotypic'
        relative=target.relative_to(package if kind=='pymicroglia' else auto).as_posix()
        left=target.read_text(encoding='utf-8').splitlines(keepends=True) if target.exists() else []
        right=source.read_text(encoding='utf-8').splitlines(keepends=True)
        patches[kind].extend(difflib.unified_diff(left,right,fromfile='a/'+relative,tofile='b/'+relative))
    report=dict(status='blocked_source_conflicts' if conflicts else 'prepared_not_applied',
        candidate=str(candidate.relative_to(home.resolve())),sources=changes,conflicts=conflicts,
        reason='Prepared only. Original recording and queued storage-repair source consumers must release before installation; recheck these hashes immediately before any write.')
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    for name,lines in patches.items():(a.output.parent/f'combined_{name}.patch').write_text(''.join(lines))
    print(json.dumps(dict(status=report['status'],files=len(changes),conflicts=conflicts)))
    if conflicts:raise SystemExit(1)


if __name__=='__main__':main()
