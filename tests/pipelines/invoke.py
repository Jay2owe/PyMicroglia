"""Exercise public packaged actions with the original demonstrations' arguments."""
from pymicroglia._results import read_document
import argparse
import importlib
import json
from pathlib import Path


def invoke(arguments, **unused):
    arguments = list(map(str, arguments))
    if '-m' in arguments:
        arguments = arguments[arguments.index('-m') + 2:]
    parser = argparse.ArgumentParser()
    parser.add_argument('operation')
    parser.add_argument('run', type=Path)
    parser.add_argument('--request', type=Path)
    parser.add_argument('--presentation', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--step', action='append')
    parser.add_argument('--choices', type=Path)
    parser.add_argument('--override-reason')
    args = parser.parse_args(arguments)
    if args.operation == 'pipeline-export':
        from pymicroglia.pipelines.audit.profiles import export_profile
        return export_profile(args.run, read_document(args.choices),
                              override_reason=args.override_reason)
    assert args.operation == 'pipeline', args.operation
    from pymicroglia.pipelines._requests import FAMILIES
    request = read_document(args.request)
    action = next(key for key, row in FAMILIES.items() if row[0] == request['pipeline'])
    from pymicroglia.run import run_action
    presentation = read_document(args.presentation) if args.presentation else None
    result = run_action(action, run=args.run, pipeline_request=request, output_dir=args.out, only=args.step,
                       presentation=presentation, if_exists='skip',
                       claim='Verify the packaged workflow against controlled synthetic inputs; no biological finding.')
    assert result.successful, {key: value.outcome.reason for key, value in result.results.items()}
    return result


def source_identity(run):
    from pymicroglia.pipelines._requests import inputs
    return inputs(run)[1]


def write_json(path, value):
    """Write a user-owned demo input or verification report, outside pipeline caches."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,sort_keys=True,separators=(",",":")),encoding="utf-8")
    return path
