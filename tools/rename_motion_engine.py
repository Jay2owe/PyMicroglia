"""Apply the approved tracking names (docs/tracking-naming-plan.md) to the Motion engine.

One deterministic rewrite, driven by docs/tracking-naming-table.csv:
functions named after a bug report or recording well get their descriptive
names, the numbered correction folders become corrections/<what it repairs>/,
and their step files lose their numbers. Settings keys ("issue005") and run
output names ("01_issue005.tif") are unchanged.

The same rewrite is applied to the PyMicroglia modules, tests and tools that
name engine code, and every SHA-256 they check against an engine file or
function is moved to the renamed file or function, so each exact patch still
recognises the code it patches.

    python tools/rename_motion_engine.py --engine <old engine folder or zip> --output <new folder> [--apply]

Without --apply only the engine is written and the repository rewrite is reported.
"""
import argparse
import ast
import csv
import hashlib
import json
import re
from pathlib import Path

from pymicroglia.tracking import shipped_engines

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / 'docs/tracking-naming-table.csv'
# Files that describe the rename itself keep the old names.
SKIP = {'tools/rename_motion_engine.py', 'tools/tracking_rename_inventory.py',
        'tools/build_motion_vendor.py'}  # builds from the Motion project's own, old layout
WORD = r'(?<![A-Za-z0-9_])%s(?![A-Za-z0-9_])'
DIGEST = re.compile(r'\b[0-9a-f]{64}\b')


def renames(table=TABLE):
    """(functions, modules, folders) maps of old to new names, from the naming table."""
    rows = list(csv.DictReader(table.open(encoding='utf-8')))
    functions = {r['old'].split(':')[1]: r['new'] for r in rows if r['kind'] == 'function'}
    modules, folders = {}, {}
    for r in rows:
        if r['kind'] != 'file':
            continue
        old_folder, old_file = r['old'].split('/')[2:]
        new_folder, new_file = r['new'].split('/')[1:]
        folders[old_folder] = new_folder
        if old_file != 'project_paths.py':
            if modules.get(old_file[:-3], new_file[:-3]) != new_file[:-3]:
                raise ValueError('one step module would get two names: ' + old_file)
            modules[old_file[:-3]] = new_file[:-3]
    return functions, modules, folders


class Rewrite:
    def __init__(self, functions, modules, folders):
        self.functions, self.modules, self.folders = functions, modules, folders
        constants = {'I%s' % old[5:]: '%s_OPS' % new.upper() for old, new in folders.items()}
        words = {**functions, **modules, **constants}
        self.words = words
        self.word = re.compile(WORD % '(%s)' % '|'.join(map(re.escape, sorted(words, key=len, reverse=True))))
        names = '|'.join(map(re.escape, sorted(folders, key=len, reverse=True)))
        # accepted_history_ops/issue008, 'accepted_history_ops' / 'issue009', PRODUCTION_OPS / "issue006"
        self.joined = re.compile(r"accepted_history_ops/(%s)(?![0-9])" % names)
        self.split = re.compile(r"""(['"])accepted_history_ops\1(\s*/\s*)(['"])(%s)\3""" % names)
        self.production = re.compile(r"""(PRODUCTION_OPS\s*/\s*)(['"])(%s)\2""" % names)
        self.home = re.compile(r"""with_name\((['"])accepted_history_ops\1\)""")
        self.glob = re.compile(r"""(['"/])accepted_history_ops(['"])""")

    def path(self, name):
        parts = name.split('/')
        if len(parts) == 4 and parts[1] == 'accepted_history_ops':
            stem = parts[3][:-3]
            return '/'.join(('code', 'corrections', self.folders[parts[2]], self.modules.get(stem, stem) + '.py'))
        return name

    def text(self, source):
        source = self.joined.sub(lambda m: 'corrections/' + self.folders[m.group(1)], source)
        source = self.split.sub(lambda m: '%scorrections%s%s%s%s%s' % (
            m.group(1), m.group(1), m.group(2), m.group(3), self.folders[m.group(4)], m.group(3)), source)
        source = self.production.sub(lambda m: m.group(1) + m.group(2) + self.folders[m.group(3)] + m.group(2), source)
        source = self.home.sub(lambda m: 'with_name(%scorrections%s)' % (m.group(1), m.group(1)), source)
        source = self.glob.sub(lambda m: m.group(1) + 'corrections' + m.group(2), source)
        return self.word.sub(lambda m: self.words[m.group(1)], source)


def functions_of(source):
    """Each top-level function's text as inspect.getsource returns it (universal newlines, so LF)."""
    source = source.replace('\r\n', '\n')
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    lines = source.splitlines(keepends=True)
    found = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            found[node.name] = ''.join(lines[start - 1:node.end_lineno])
    return found


def sha(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode()).hexdigest()


def rename_engine(engine, output, rewrite):
    """Write the renamed engine folder; return the old-to-new digest map of files and functions."""
    if Path(output).exists():
        raise FileExistsError(output)
    entries = shipped_engines.files(engine)
    renamed, digests = {}, {}
    for name, data in entries.items():
        if name == 'source_hashes.json':
            continue
        new_name = rewrite.path(name)
        new_data = data
        if name.endswith('.py'):
            text = data.decode('utf-8')
            new_text = rewrite.text(text)
            new_data = new_text.encode('utf-8')
            old_functions, new_functions = functions_of(text), functions_of(new_text)
            for function, body in old_functions.items():
                new_function = rewrite.functions.get(function, function)
                digests[sha(body)] = sha(new_functions[new_function])
        if new_name in renamed:
            raise ValueError('two engine files would share ' + new_name)
        renamed[new_name] = new_data
        digests[sha(data)] = sha(new_data)
        # Patches also pin files with Windows line endings normalised to LF.
        digests[sha(data.replace(b'\r\n', b'\n'))] = sha(new_data.replace(b'\r\n', b'\n'))
    manifest = json.loads(entries['source_hashes.json'])
    manifest['files_sha256'] = {name: sha(data) for name, data in sorted(renamed.items())}
    # archive_sha256 names the engine the rename started from (shipped_engines.digest).
    manifest['renamed_from'] = dict(archive_sha256=shipped_engines.digest(engine),
                                    naming_table_sha256=sha(TABLE.read_bytes()),
                                    tool='tools/rename_motion_engine.py')
    renamed['source_hashes.json'] = (json.dumps(manifest, indent=2) + '\n').encode()
    shipped_engines.write(output, renamed)
    digests = {old: new for old, new in digests.items() if old != new}
    return digests


def repository_files():
    for folder in ('src/pymicroglia/tracking', 'tests', 'tools'):
        for path in sorted((ROOT / folder).glob('*.py')):
            if path.relative_to(ROOT).as_posix() not in SKIP:
                yield path


def rewrite_repository(rewrite, digests, apply):
    changed = {}
    for path in repository_files():
        text = path.read_bytes().decode('utf-8')  # keeps the file's own line endings
        new = DIGEST.sub(lambda m: digests.get(m.group(0), m.group(0)), rewrite.text(text))
        if new != text:
            changed[path.relative_to(ROOT).as_posix()] = (
                sum(1 for m in DIGEST.finditer(text) if m.group(0) in digests))
            if apply:
                path.write_bytes(new.encode('utf-8'))
    return changed


def rename_snapshot(source, target, engine, rewrite, digests):
    """Copy an A1 replay snapshot with only this rename applied, byte for byte otherwise."""
    import shutil
    shutil.copytree(source, target, ignore=shutil.ignore_patterns(
        '__pycache__', 'composition.json', 'motion_engine.zip', 'motion_engine'))
    changed = []
    for path in sorted(target.rglob('*.py')):
        data = path.read_bytes()
        text = data.decode('utf-8')
        new = DIGEST.sub(lambda m: digests.get(m.group(0), m.group(0)), rewrite.text(text))
        if new != text:
            path.write_bytes(new.encode('utf-8'))
            changed.append(path.relative_to(target).as_posix())
    shutil.copytree(engine, target / 'pymicroglia/tracking/data/motion_engine')
    (target / 'composition.json').write_text(json.dumps(dict(
        status='renamed_pending_validation', base=source.name, transform='tools/rename_motion_engine.py',
        naming_table_sha256=sha(TABLE.read_bytes()), motion_engine_sha256=shipped_engines.digest(engine),
        rewritten=changed), indent=2) + '\n')
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--engine', type=Path, required=True, help='engine folder, or a zip from before release 1b')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--apply', action='store_true', help='also rewrite the repository files')
    parser.add_argument('--snapshot', type=Path, nargs=2, metavar=('SOURCE', 'TARGET'),
                        help='also write a renamed copy of an A1 replay snapshot')
    args = parser.parse_args()
    rewrite = Rewrite(*renames())
    digests = rename_engine(args.engine, args.output, rewrite)
    if args.snapshot:
        print('snapshot files rewritten:', len(rename_snapshot(*args.snapshot, args.output, rewrite, digests)))
    changed = rewrite_repository(rewrite, digests, args.apply)
    print(json.dumps(dict(engine=str(args.output), engine_sha256=shipped_engines.digest(args.output),
                          moved_digests=len(digests), repository_files=len(changed),
                          digests_moved_in=sum(changed.values()), applied=args.apply), indent=1))
    for name, count in sorted(changed.items()):
        print('  %s%s' % (name, ' (%d digests)' % count if count else ''))


if __name__ == '__main__':
    main()
