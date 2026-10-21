"""Paired live-byte ledger lookup using a real completed pipeline ledger."""
import argparse
import gc
import importlib.util
import json
from pathlib import Path
import statistics
import time
from benchmark_pipeline_execution import ROOT, digest


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    from auto_organotypic.store import ledger
    import psutil
    path=ROOT/'development/speed-pass-20261011/baseline/store_ledger.py'
    spec=importlib.util.spec_from_file_location('auto_organotypic.store._before_ledger',path)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    folder=ROOT/'development/speed-pass-20261011/complete/before2/rhythm/outputs'
    source=max(folder.rglob('artefacts.json'),key=lambda p:p.stat().st_size)
    data=json.loads(source.read_text(encoding='utf-8'))
    targets=[source.parent.parent/name for name in data['entries']]
    expected=[old.entry_for(p) for p in targets]
    assert expected==[ledger.entry_for(p) for p in targets]
    trials={'before':[],'after':[]};rss=[]
    for repeat in range(6):
        for mode,implementation in ([('before',old),('after',ledger)] if repeat%2==0 else [('after',ledger),('before',old)]):
            gc.collect();start=time.perf_counter()
            for _ in range(5):
                result=[implementation.entry_for(p) for p in targets]
                assert result==expected
            trials[mode].append(time.perf_counter()-start)
            rss.append(psutil.Process().memory_info().rss)
    output={'operation':'verified single artifact lookup','fixture':str(source),'input_sha256':digest(source),
        'ledger_bytes':source.stat().st_size,'lookups_per_trial':5*len(targets),'trials':trials,'parity':True,
        'peak_sampled_rss_bytes':max(rss),'cache_input_bytes':ledger._ENTRY_LOOKUP.byte_count,
        'parsed_lookup_code':digest(Path(ledger.__file__).with_name('_parsed_lookup.py')),
        'before_code':digest(path),'after_code':digest(Path(ledger.__file__)),
        'speedup':statistics.median(trials['before'])/statistics.median(trials['after'])}
    args.output.write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
