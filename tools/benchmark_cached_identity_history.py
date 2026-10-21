"""Run the existing exact stage replay with immutable component reuse enabled."""
import argparse
from contextlib import nullcontext
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import platform
import sys


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--cache-adapter',type=Path,required=True)
    parser.add_argument('--reuse-terminal-features',action='store_true')
    parser.add_argument('--lazy-application',action='store_true')
    args, remaining=parser.parse_known_args()
    def value(flag):
        return remaining[remaining.index(flag)+1]
    if value('--operation')!='flip':
        raise ValueError('Only the audited immutable identity-switch discovery is supported')
    engine=Path(value('--engine'))
    output=Path(value('--output'))
    platform.platform=lambda *a,**k:'Windows-cached-identity-comparison'
    sys.path[:0]=[str((engine/'code').resolve()),str(args.cache_adapter.resolve())]
    module=importlib.import_module('complete_flip_recovery')
    adapter=importlib.import_module('frame_component_reuse')
    features=importlib.import_module('terminal_feature_reuse') if args.reuse_terminal_features else None
    lazy=importlib.import_module('lazy_flip_components') if args.lazy_application else None
    spec=importlib.util.spec_from_file_location('current_identity_benchmark',
        Path(__file__).with_name('benchmark_duplicate_history.py'))
    benchmark=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    source_files=(Path(adapter.__file__).resolve(),Path(__file__).resolve())
    if features is not None:
        source_files+= (Path(features.__file__).resolve(),)
    if lazy is not None:
        source_files+= (Path(lazy.__file__).resolve(),)
    measured_sources={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
    sys.argv=[sys.argv[0],*remaining]
    with adapter.install(module, output/'component_cache') as counters, \
            (features.install(module) if features else nullcontext({})) as feature_counters, \
            (lazy.install(module,output/'application_cache') if lazy else nullcontext({})) as lazy_counters:
        benchmark.main()
    report_path=output/'report.json'
    report=json.loads(report_path.read_text())
    report['component_cache']=counters
    report['terminal_features']=feature_counters
    report['application_cache']=lazy_counters
    report['sources'].update(measured_sources)
    report['adapter_sources_unchanged']=all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
        for path,digest in measured_sources.items())
    if not report['adapter_sources_unchanged']:
        report['status']='fail: adapter source changed during measurement'
    report_path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(component_cache=counters,report=str(report_path))),flush=True)
    if report['status']!='pass':
        raise SystemExit(1)


if __name__=='__main__':
    main()
