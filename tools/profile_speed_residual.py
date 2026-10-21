"""Diagnose the predeclared small rendering residual; never use profiled timings as benchmarks."""
import argparse
import cProfile
import json
import os
from pathlib import Path
import pstats
import shutil
import sys

from benchmark_pipeline_execution import ROOT,configure_ledger,digest,source_identity


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['before','after'])
    args=parser.parse_args()
    folder=ROOT/'development/speed-pass-20261011/complete'/('residual-profile-'+args.mode)
    folder.mkdir(exist_ok=False)
    os.environ['PYMICROGLIA_INDEX']=str(folder/'index')
    os.environ['PYMICROGLIA_STORE']=str(folder/'store')
    os.environ['MPLBACKEND']='Agg'
    sys.executable=shutil.which('python') or sys.executable
    from pymicroglia.run import run_action
    treatment=configure_ledger(args.mode=='before')
    before=source_identity()
    profiler=cProfile.Profile()
    result=profiler.runcall(run_action,'tissue_coverage_gaps',
        run=ROOT/'tests/fixtures/motion_parity/run',stem='parity_A1',output_dir=folder/'output',
        claim='Controlled synthetic performance diagnosis; no biological finding.')
    assert result['figures'] and all(Path(p).is_file() for p in result['figures'])
    profiler.dump_stats(str(folder/'profile.pstats'))
    stats=pstats.Stats(profiler)
    rows=[{'file':file,'line':line,'function':name,'calls':calls,'own_seconds':own,'cumulative_seconds':total}
          for (file,line,name),(_,calls,own,total,_) in stats.stats.items()]
    result={'kind':'profile diagnostic','profiled':True,'action':'tissue_coverage_gaps',
            'mode':args.mode,'treatment':treatment,'source_unchanged':before==source_identity(),
            'profile_sha256':digest(folder/'profile.pstats'),
            'largest':sorted(rows,key=lambda r:r['cumulative_seconds'],reverse=True)[:25],
            'targeted':[row for row in rows if row['function'] in ['entry_for','deepcopy','_analysis_scope','fingerprint']
                        or '_parsed_lookup' in row['file']],
            'interpretation':'Instrumented diagnostic only; excluded from measured execution coverage and speed ratios.'}
    (folder/'diagnostic.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'mode':args.mode,'source_unchanged':result['source_unchanged'],'targeted':result['targeted']}))


if __name__=='__main__':main()
