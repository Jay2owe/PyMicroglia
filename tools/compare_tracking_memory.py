"""Compare scientific files and peak-memory receipts from two complete runs."""
from pathlib import Path
import argparse
import hashlib
import json

import tifffile


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def image_signature(path):
    h=hashlib.sha256()
    with tifffile.TiffFile(path) as image:
        description=[(tuple(s.shape),str(s.dtype)) for s in image.series]
        for page in image.pages:
            h.update(page.asarray().tobytes())
    return description,h.hexdigest()


def compare(before,after,*,deferred_native_review=False):
    def files(root):
        return {str(p.relative_to(root)):p for p in root.glob('m*/profile/**/*')
                if p.is_file() and (p.suffix.lower() in ('.tif','.tiff','.csv')
                or p.suffix=='.json' and 'completed_feedback' in p.parts)}
    a,b=files(before),files(after)
    keys=set(a)|set(b);checks=[];deferred=[]
    for name in sorted(keys):
        if name not in a or name not in b:
            # Bounded execution deliberately delegates only the native display
            # movie to the package renderer. Never exempt scientific stages.
            if (deferred_native_review and name in a and name not in b
                    and Path(name).parts[0]=='m6_review'
                    and Path(name).suffix.lower() in ('.tif','.tiff')):
                deferred.append(name);continue
            checks.append(dict(file=name,equal=False,reason='missing'));continue
        ah,bh=sha(a[name]),sha(b[name])
        equal=ah==bh
        if not equal and a[name].suffix.lower() in ('.tif','.tiff'):
            equal=image_signature(a[name])==image_signature(b[name])
        checks.append(dict(file=name,equal=equal,before_sha256=ah,after_sha256=bh))
    first=json.loads((before/'m23_completed_feedback/profile/run.json').read_text())
    last=json.loads((after/'m23_completed_feedback/profile/run.json').read_text())
    def stages(report):return [{k:v for k,v in row.items() if k!='seconds'} for row in report['stages']]
    boundaries=stages(first)==stages(last)
    return dict(status='pass' if checks and all(r['equal'] for r in checks) and boundaries else 'fail',
        file_count=len(checks),boundary_count=len(first['stages']),all_boundary_hashes_counts_and_order_equal=boundaries,
        checks=checks,deferred_native_review_files=deferred,
        scope='All available native/correction TIFF pixels, CSV tables, correction JSON tables and114 ordered correction audit records; excludes runtime and producer metadata. Explicitly listed native display files may be deferred in bounded mode.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--before',type=Path,required=True)
    p.add_argument('--after',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--deferred-native-review',action='store_true')
    p.add_argument('--native-before-history',action='store_true')
    p.add_argument('--before-run',default='profile');p.add_argument('--after-run',default='profile')
    args=p.parse_args()
    if args.native_before_history:
        stage_names=('m0_inputs','m1_motion','m2_anchor','m3_track','m4_events','m5_reconcile',
            'm7_persistence','m8_merge','m9_alias','m12_identity_reservation','m19_layered_lineage','m20_oscillatory_lineage')
        checks=[];timings=[]
        for stage in stage_names:
            before=args.before/stage/args.before_run;after=args.after/stage/args.after_run
            old,new=[json.loads((folder/'run.json').read_text()) for folder in (before,after)]
            assert old['status']=='done' and new['status']=='done',(stage,old['status'],new['status'])
            timings.append(dict(stage=stage,before_seconds=old['elapsed_seconds'],after_seconds=new['elapsed_seconds']))
            def science(folder):return {str(f.relative_to(folder)):f for f in folder.rglob('*')
                if f.is_file() and f.suffix.lower() in ('.tif','.tiff','.csv')}
            a,b=science(before),science(after)
            for name in sorted(set(a)|set(b)):
                exists=name in a and name in b
                ah,bh=(sha(a[name]),sha(b[name])) if exists else (None,None)
                equal=exists and (ah==bh or (Path(name).suffix.lower() in ('.tif','.tiff')
                    and image_signature(a[name])==image_signature(b[name])))
                checks.append(dict(file=stage+'/'+name,equal=equal,before_sha256=ah,after_sha256=bh))
        report=dict(status='pass' if checks and all(c['equal'] for c in checks) else 'fail',
            file_count=len(checks),complete_group=True,group='earlier_tracking',stages=timings,
            before_seconds=sum(t['before_seconds'] for t in timings),
            after_seconds=sum(t['after_seconds'] for t in timings),checks=checks,
            scope='All twelve native stages before accepted-history processing; sum of complete sequential stage runtimes, including their native reviews. Candidate has profiling enabled; shared desktop; later adapter revisions excluded.')
    else:report=compare(args.before,args.after,deferred_native_review=args.deferred_native_review)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='checks'}))
    if report['status']!='pass':raise SystemExit(1)


if __name__=='__main__':main()
