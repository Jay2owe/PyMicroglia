"""Package the accepted full correction sequence, without study images or selectors.

Build-time inputs are the recorded accepted replay recipe and its source closure.
Runtime consumes freshly generated arrays, never the recipe's per-movie paths.
"""
from pathlib import Path
import ast
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MOTION = ROOT.parent / 'Motion'
TUNING = MOTION / 'issues/ISSUE-009-completed-feedback-rule-optimization/analysis/completed_feedback_tuning'
RECIPE = TUNING / 'rounds/R223-speed-and-normalisation/a162_settings.json'
OUTPUT = ROOT / 'src/pymicroglia/tracking/data/completed_feedback_engine.zip'


def rebuild_sequence(verified_archive):
    """Reuse verified frozen operators when only orchestration lifetime changes.

An explicit source archive pins every unchanged entry. This route does not
silently substitute current development sources for historical operators.
"""
    source = Path(verified_archive)
    with zipfile.ZipFile(source) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(entries['source_hashes.json'])
    for name, digest in manifest['files_sha256'].items():
        assert hashlib.sha256(entries[name]).hexdigest() == digest, name
    key = 'code/completed_feedback_sequence.py'
    data = (MOTION / manifest['source_origins'][key]).read_bytes()
    entries[key] = data
    manifest['files_sha256'][key] = hashlib.sha256(data).hexdigest()
    manifest['storage_revision'] = dict(
        parent_archive_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        changed_source=key,
        reason='Release consumed discovery snapshots; all frozen scientific operators and settings reused exactly')
    entries['source_hashes.json'] = (json.dumps(manifest,indent=2)+'\n').encode()
    with zipfile.ZipFile(OUTPUT,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,data in sorted(entries.items()):
            archive.writestr(zipfile.ZipInfo(name,(2026,10,12,0,0,0)),data,
                             compress_type=zipfile.ZIP_DEFLATED)
    print(json.dumps(dict(archive=str(OUTPUT),entries=len(entries),changed_source=key)))
    return OUTPUT


def build():
    recipe = json.loads(RECIPE.read_text())
    directories = [TUNING/'code', MOTION/'code', TUNING/'scripts']
    sources = {}
    for folder in directories:
        for path in folder.glob('*.py'):
            sources.setdefault(path.stem, path)
    entries = {}; origins = {}; seen = set(); adaptations = {}

    def collect(name):
        if name in seen or name not in sources:
            return
        seen.add(name)
        path = sources[name]
        data = path.read_bytes()
        key = 'code/'+name+'.py'
        if name == 'recover_persistent_duplicate_soma_sites':
            # This unused CLI root lookup otherwise prevents importing the
            # array functions outside Motion. Scientific functions are intact.
            old = b'ROOT = next(p for p in HERE.parents if (p / "WORKFLOW_CONTRACT.md").exists())'
            new = b'ROOT = next((p for p in HERE.parents if (p / "WORKFLOW_CONTRACT.md").exists()), HERE.parent)'
            assert data.count(old) == 1
            adaptations[key] = dict(original_sha256=hashlib.sha256(data).hexdigest(),
                change='Allow import outside the development project; unused CLI root fallback only')
            data = data.replace(old,new)
        entries[key] = data
        origins[key] = path.relative_to(MOTION).as_posix()
        tree = ast.parse(data.decode('utf-8-sig'))
        # CLI-only producers are not dependencies of the array functions.
        tree.body = [node for node in tree.body if not (
            isinstance(node, ast.If) and '__name__' in ast.unparse(node.test))]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for item in node.names:
                    collect(item.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                collect(node.module.split('.')[0])
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                # Several accepted sequences select their method by a module name.
                if node.value.isidentifier():
                    collect(node.value)

    collect('completed_feedback_sequence')
    settings = {key: recipe[key] for key in ('density_parameters', 'revise_isolated_prior_reassignment', 'operations')}
    for family, operations in settings['operations'].items():
        for spec in operations:
            folder = Path(spec['code_dir'])
            destination = 'frozen/'+family+'/'+spec['stage']
            for name, digest in spec['code_sha256'].items():
                path = folder/(name+'.py')
                data = path.read_bytes()
                assert hashlib.sha256(data).hexdigest() == digest, path
                key = destination+'/'+path.name
                entries[key] = data
                origins[key] = path.relative_to(MOTION).as_posix()
                for node in ast.walk(ast.parse(data.decode('utf-8-sig'))):
                    if isinstance(node, ast.Import):
                        for item in node.names:
                            if item.name not in spec['code_sha256']:
                                collect(item.name.split('.')[0])
                    elif isinstance(node, ast.ImportFrom) and node.module and node.module not in spec['code_sha256']:
                        collect(node.module.split('.')[0])
            spec['code_dir'] = destination
    entries['settings.json'] = (json.dumps(settings, indent=2)+'\n').encode()
    boundaries = [sorted((family,stage) for family,stages in movie['expected'].items() for stage in stages)
                  for movie in recipe['movies'].values()]
    assert all(value == boundaries[0] for value in boundaries)
    manifest = dict(schema='pymicroglia.accepted-completed-feedback.v1',
        accepted_attempt='R218-A004', saved_feedback=dict(satisfied=1798,total=2348),
        required_boundaries=boundaries[0], required_inventories=sorted(recipe['inventories']),
        metric='Saved feedback relationships; not dense pixel accuracy or a prediction for new data',
        original_recipe_sha256=hashlib.sha256(RECIPE.read_bytes()).hexdigest(),
        files_sha256={name:hashlib.sha256(data).hexdigest() for name,data in entries.items()},
        source_origins=origins, packaging_adaptations=adaptations)
    entries['source_hashes.json'] = (json.dumps(manifest,indent=2)+'\n').encode()
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(OUTPUT,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,data in sorted(entries.items()):
            info=zipfile.ZipInfo(name,(2026,10,12,0,0,0))
            archive.writestr(info,data,compress_type=zipfile.ZIP_DEFLATED)
    print(json.dumps(dict(archive=str(OUTPUT),modules=len(seen),entries=len(entries))))
    return OUTPUT


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--reuse-archive',type=Path)
    arguments=parser.parse_args()
    if arguments.reuse_archive:
        rebuild_sequence(arguments.reuse_archive)
    else:
        build()
