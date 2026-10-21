"""Compare old/new audit memory in isolated processes on accepted stage files.

Example: python tools/benchmark_correction_memory.py --recipe <a162_settings.json>
--output validation/completed_base_20261012/memory/audit.json --movie 95_A1
No tracking execution or synthetic accuracy estimate is claimed.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import time
import tracemalloc

import numpy as np
import psutil
import tifffile

from pymicroglia.tracking.correction_audit import CorrectionAudit, array_sha256


def execute(recipe, movie, mode):
    data = json.loads(Path(recipe).read_text())['movies'][movie]
    expected = data['expected']
    first = next(iter(next(iter(expected.values())).values()))
    labels = tifffile.imread(first['labels'])
    process = psutil.Process()
    initial_rss = process.memory_info().rss
    tracemalloc.start()
    started = time.perf_counter()
    previous = labels.copy() if mode == 'old' else None
    audit = CorrectionAudit(labels) if mode == 'new' else None
    inferred = np.zeros(labels.shape, bool) if mode == 'old' else audit.inferred
    records = []
    peak_rss = initial_rss
    for family, stages in expected.items():
        for stage, paths in stages.items():
            labels = tifffile.imread(paths['labels'])
            ledger = tifffile.imread(paths['ledger'])
            if mode == 'old':
                delta = labels != previous
                inferred[:] |= delta
                result = dict(changed_pixels=int(delta.sum()),
                    label_array_sha256=hashlib.sha256(labels.tobytes()).hexdigest(),
                    ledger_array_sha256=hashlib.sha256(ledger.tobytes()).hexdigest())
                previous = labels.copy()
            else:
                result = audit.record(labels, ledger)
            records.append(dict(family=family, stage=stage, **result))
            peak_rss = max(peak_rss, process.memory_info().rss)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return dict(mode=mode, movie=movie, shape=list(labels.shape), seconds=time.perf_counter()-started,
        python_numpy_peak_bytes=peak, python_numpy_retained_bytes=current,
        process_peak_working_set_bytes=getattr(process.memory_info(), 'peak_wset', peak_rss),
        initial_rss_bytes=initial_rss, inferred_sha256=array_sha256(inferred), stages=records)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--recipe', required=True)
    parser.add_argument('--movie', default='95_A1')
    parser.add_argument('--output', required=True)
    parser.add_argument('--worker', choices=['old','new'])
    args=parser.parse_args()
    if args.worker:
        print(json.dumps(execute(args.recipe,args.movie,args.worker)))
        return
    trials=[]
    # Reverse order on the second pair to reduce filesystem-cache bias.
    for mode in ['old','new','new','old']:
        command=[sys.executable,'-X','utf8',str(Path(__file__).resolve()),'--recipe',args.recipe,
                 '--movie',args.movie,'--output',args.output,'--worker',mode]
        result=subprocess.run(command,check=True,capture_output=True,text=True)
        trials.append(json.loads(result.stdout))
    first=trials[0]
    assert all(row['stages']==first['stages'] and row['inferred_sha256']==first['inferred_sha256'] for row in trials)
    report=dict(status='pass',scope='Audit overhead over all saved correction stages; not full tracking process peak',
        recipe_sha256=hashlib.sha256(Path(args.recipe).read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        exact_stage_count=len(first['stages']),trials=trials)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='trials'}))
    for r in trials: print(r['mode'],r['seconds'],r['python_numpy_peak_bytes'],r['process_peak_working_set_bytes'])


if __name__=='__main__': main()
