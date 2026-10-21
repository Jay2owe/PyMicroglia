"""Root import stays light and exported objects remain the owning objects."""
import importlib
import json
from pathlib import Path
import shutil
import subprocess


def test_parameter_access_does_not_load_scientific_actions():
    result=subprocess.check_output([shutil.which('python'),'-c',
        "import sys,json,pymicroglia as p; from pymicroglia import ParamBlock; "
        "assert ParamBlock is p.params.ParamBlock; "
        "print(json.dumps(sorted(set(sys.modules)&{'numpy','matplotlib','scipy','pymicroglia.registry'})))"],
        cwd=Path(__file__).resolve().parents[1],text=True,timeout=30)
    assert json.loads(result)==[]


def test_all_public_names_resolve_to_original_module_objects():
    import pymicroglia as package
    for name in package.__all__:
        value=getattr(package,name)
        if name in package._PUBLIC_OBJECTS:
            assert value is getattr(importlib.import_module('pymicroglia.'+package._PUBLIC_OBJECTS[name]),name)
        elif name in package._PUBLIC_MODULES:
            assert value is importlib.import_module('pymicroglia.'+name)
        assert name in dir(package)
    try:
        package.not_an_export
    except AttributeError:
        pass
    else:
        raise AssertionError('Unknown exports must raise AttributeError')
