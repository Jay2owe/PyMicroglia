"""Run focused checks against an explicit frozen tracker and storage dependency."""
import argparse
import importlib
import platform
from pathlib import Path
import sys

from benchmark_working_storage import load


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--basetemp',type=Path,required=True)
    parser.add_argument('--native-engine',type=Path)
    parser.add_argument('--report',type=Path,help='Optional compact JUnit result receipt')
    parser.add_argument('tests',nargs='+')
    args=parser.parse_args()
    if args.native_engine and not (args.native_engine/"code/common.py").is_file():
        parser.error("--native-engine must contain code/common.py")
    source=args.source.resolve()
    platform.platform=lambda *a,**k:'Windows-frozen-tracking-tests'
    sys.path.insert(0,str(source))
    load(source/'array_storage.py')
    # Import before repository conftest files prepend the editable checkout.
    # A green suite against another source tree is not candidate verification.
    for name in ('engine','engine_bounded','frozen_io','checkpoints',
                 'completed_evidence','completed_feedback','correction_storage',
                 'prepare_stream','prepare','motion_preparation','display_scale'):
        module=importlib.import_module('pymicroglia.tracking.'+name)
        if not Path(module.__file__).resolve().is_relative_to(source):
            raise RuntimeError('Test imported a different source: '+module.__file__)
    import pytest
    class ExistingNative:
        def pytest_collection_modifyitems(self,items):
            if args.native_engine:
                for item in items:
                    if item.module.__name__.endswith('test_frozen_io'):
                        item.module._engine=lambda folder:args.native_engine.resolve()
    print('Testing frozen source:',source,flush=True)
    report=[] if args.report is None else ['--junitxml',str(args.report)]
    raise SystemExit(pytest.main([*args.tests,'-q','--tb=short','--basetemp',str(args.basetemp),*report],
                                plugins=[ExistingNative()]))


if __name__=='__main__':main()
