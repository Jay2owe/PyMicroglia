"""Inventory for the tracking file renaming (docs/tracking-naming-plan.md, step 1).

Lists every Python file in the two shipped engine folders and in
pymicroglia/tracking, with its first docstring line, top-level functions and
the functions still named after a bug report or recording well. Each tracking
module is linked to the engine files it patches through the SHA-256 digests it
checks before patching. Read-only; writes one JSON file.

    python tools/tracking_rename_inventory.py --out <file.json>
"""
import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

from pymicroglia.tracking import shipped_engines

TRACKING = Path(__file__).resolve().parents[1] / 'src' / 'pymicroglia' / 'tracking'
ENGINES = ('motion_engine', 'completed_feedback_engine')
BUG_NAME = re.compile(r'issue\d|well_[a-h]\d', re.I)
DIGEST = re.compile(r'\b[0-9a-f]{64}\b')


def describe(source):
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return dict(doc='', functions=[], parse_error=True)
    doc = (ast.get_docstring(tree) or '').strip().splitlines()
    functions = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    return dict(doc=doc[0] if doc else '', functions=functions,
                bug_named=[f for f in functions if BUG_NAME.search(f)],
                lines=source.count('\n') + 1)


def function_sources(source):
    """Each top-level function's text as inspect.getsource returns it, decorators included.

    inspect.getsource reads files with universal newlines, so Windows line endings become LF.
    """
    source = source.replace('\r\n', '\n')
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    lines = source.splitlines(keepends=True)
    found = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            found.append((node.name, ''.join(lines[start - 1:node.end_lineno])))
    return found


def engine_files():
    files, by_digest = [], {}
    for name in ENGINES:
        root = TRACKING / 'data' / name
        for member in shipped_engines.names(root):
            if not member.endswith('.py'):
                continue
            data = (root / member).read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            entry = dict(engine=name, path=member, sha256=digest,
                         bug_named_path=bool(BUG_NAME.search(member)),
                         **describe(data.decode('utf-8', 'replace')))
            files.append(entry)
            by_digest.setdefault(digest, []).append('%s:%s' % (name, member))
            # Patches that check one function hash inspect.getsource(function).
            for function, source in function_sources(data.decode('utf-8', 'replace')):
                function_digest = hashlib.sha256(source.encode()).hexdigest()
                by_digest.setdefault(function_digest, []).append('%s:%s:%s' % (name, member, function))
    return files, by_digest


def tracking_files(by_digest):
    files = []
    for path in sorted(TRACKING.glob('*.py')):
        source = path.read_text(encoding='utf-8')
        digests = set(DIGEST.findall(source))
        files.append(dict(path=path.name, has_install='def install(' in source,
                          patches=sorted({t for d in digests for t in by_digest.get(d, [])}),
                          unmatched_digests=len([d for d in digests if d not in by_digest]),
                          **describe(source)))
    return files


# Proposed names where the old name says nothing about the work (Jamie approves).
BARE_FUNCTIONS = {
    '_issue005': ('_confidence_layered_retracking', 'Re-track by confidence layer, then optimise lineage, host-merge and territory graphs'),
    '_issue006': ('_contact_tenure', 'Keep a static territory and its mobile neighbour through contact'),
    '_issue007': ('_host_capture_correction', 'Undo forced host carries that compete with independent destinations'),
    '_issue008': ('_ownership_quarantine', 'Global and tenure quarantine, merge bridging and duplicate cleanup'),
    '_issue009': ('_hierarchy_reconcile', 'Hierarchy-ordered quarantine reconciled with movie-wide identity authority'),
    '_install_issue009_component_extractor': ('_install_cropped_component_extractor', 'Swap in the cropped connected-component extractor'),
    '_issue010': ('_short_lived_reclaim', 'Drop short-lived identities and reclaim their gaps for persistent lineages'),
    '_issue011': ('_merge_split_reconcile', 'Reassign identities across bracketed merge-and-separate events'),
    '_issue012': ('_domain_ownership_reconcile', 'Domain ownership rules, then restore names after dormant releases'),
    '_issue023_tail': ('_generalised_repairs', 'Territory memory, unclaimed bodies, local and reciprocal lineages, stationary gaps'),
}
OP_FOLDERS = {
    'issue006': 'contact_tenure', 'issue007': 'host_capture', 'issue008': 'ownership_quarantine',
    'issue009': 'hierarchy', 'issue010': 'short_lived', 'issue011': 'merge_splits',
    'issue012': 'domain_ownership', 'issue013': 'seat_identity', 'issue015': 'blob_residency',
    'issue017': 'continuity',
}
# Step files whose name without its number would be too generic in the engine's flat import namespace.
OP_FILES = {
    'issue007/m1_detect_hypotheses.py': 'detect_host_capture.py',
    'issue007/m2_reconcile.py': 'reconcile_host_capture.py',
    'issue017/m1_census.py': 'continuity_census.py',
    'issue017/m2_classify_evidence.py': 'classify_continuity_evidence.py',
    'issue017/m3_repair.py': 'continuity_repair.py',
}
STEP_PREFIX = re.compile(r'^[ms]\d+_')
NUMBER_PREFIX = re.compile(r'^_(well_([a-h]\d_)?)?issue\d+_')


def table(inventory):
    """Rows of (kind, old, new, note) for the naming table."""
    rows = []
    for entry in inventory['engine']:
        if entry['engine'] != ENGINES[0]:
            continue
        for name in entry['bug_named']:
            if name in BARE_FUNCTIONS:
                new, note = BARE_FUNCTIONS[name]
                rows.append(('function', '%s:%s' % (entry['path'], name), new, 'proposed: ' + note))
            else:
                rows.append(('function', '%s:%s' % (entry['path'], name), NUMBER_PREFIX.sub('_', name), 'number dropped'))
        parts = entry['path'].split('/')
        if entry['bug_named_path'] and len(parts) == 4:
            folder = OP_FOLDERS[parts[2]]
            stem = OP_FILES.get('/'.join(parts[2:]), STEP_PREFIX.sub('', parts[3]))
            rows.append(('file', entry['path'], 'corrections/%s/%s' % (folder, stem), 'step number dropped'))
    paths = {e['path'] for e in inventory['engine'] if e['engine'] == ENGINES[0]}
    for path in sorted(paths):
        if path.endswith('_integration.py') and path.replace('_integration.py', '.py') in paths:
            rows.append(('pair', path, path.replace('_integration.py', '.py'), 'merged into one module'))
    for entry in inventory['tracking']:
        if entry['has_install']:
            kind = 'behaviour' if entry['path'] in BEHAVIOUR else 'speed'
            targets = ', '.join(entry['patches']) or 'target found at fold time (no static fingerprint match)'
            rows.append((kind, 'tracking/' + entry['path'], 'folded into ' + targets, entry['doc']))
    return rows


# Patching modules that change what the engine decides (safety nets, evidence,
# accepted rule changes); every other patching module is an exact speed or memory change.
BEHAVIOUR = {'stage_fallbacks.py', 'pipeline_fallbacks.py', 'residency_frames.py',
             'repair_composition.py', 'lineage_overlaps.py', 'owner_consensus_conflicts.py',
             'reciprocal_overlaps.py', 'owner_identity_accounting.py', 'process_proposals.py',
             'boundary_frame_checks.py', 'merge_foreground_scope.py', 'tracking_pass_record.py'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--table', help='also write the naming table as CSV')
    args = parser.parse_args()
    engine, by_digest = engine_files()
    tracking = tracking_files(by_digest)
    inventory = dict(engine=engine, tracking=tracking)
    Path(args.out).write_text(json.dumps(inventory, indent=1), encoding='utf-8')
    if args.table:
        import csv
        with open(args.table, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(('kind', 'old', 'new', 'note'))
            writer.writerows(table(inventory))
    print('engine files', len(engine), 'bug-named functions', sum(len(e['bug_named']) for e in engine),
          'bug-named paths', sum(e['bug_named_path'] for e in engine))
    print('tracking files', len(tracking), 'with install()', sum(t['has_install'] for t in tracking),
          'linked to engine files', sum(bool(t['patches']) for t in tracking))


if __name__ == '__main__':
    main()
