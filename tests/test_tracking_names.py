"""Shipped engine files and corrections are named for what they do (docs/tracking-naming-plan.md)."""
import ast
import json
import re
from pathlib import Path

from pymicroglia.tracking import shipped_engines

ENGINE = Path(__file__).parents[1] / 'src/pymicroglia/tracking/data/motion_engine'
BUG_NAME = re.compile(r'issue\d|well_[a-h]\d', re.I)


def test_engine_paths_and_functions_carry_no_bug_report_names():
    names = shipped_engines.names(ENGINE)
    assert not [n for n in names if BUG_NAME.search(n) or 'accepted_history_ops' in n]
    assert any(n.startswith('code/corrections/ownership_quarantine/') for n in names)
    named = []
    for name in names:
        if name.endswith('.py'):
            tree = ast.parse((ENGINE / name).read_bytes())
            named += ['%s:%s' % (name, node.name) for node in ast.walk(tree)
                      if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and BUG_NAME.search(node.name)]
    assert named == []


def test_renamed_engine_records_its_source():
    manifest = json.loads((ENGINE / 'source_hashes.json').read_bytes())
    assert manifest['renamed_from']['tool'] == 'tools/rename_motion_engine.py'
    assert len(manifest['renamed_from']['archive_sha256']) == 64
