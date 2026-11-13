"""Pack the A1 full-replay inputs into one hashed zip, and run that replay from the unpacked bundle elsewhere.

pack (here, from the PyMicroglia folder):
    python tools/full_replay_bundle.py pack --snapshot <combined snapshot> [--snapshot ...] --reference-evidence <evidence_hashes.json> --output <bundle.zip>
run (on the machine that unpacks it; its own Python):
    python full_replay_bundle.py run --bundle <bundle.zip> [--bundle <snapshots-only bundle> ...] --root <folder> --tag <snapshot tag> [--tag ...] [--history]
A later bundle made with pack --snapshots-only adds snapshots to the same root.

The zip keeps PyMicroglia-relative paths: the A1 tracking engine, the A1
motion config and inputs, the replay tools, each snapshot and this machine's
auto_organotypic source (run first on the path, so both machines run the same
code), plus a manifest of every file's SHA-256. `run` refuses a root that already holds other files,
unpacks, checks every hash, then runs benchmark_completed_evidence (evidence
and every correction, 476 frames, compared with the recorded A1 output and the
reference evidence fingerprints) and optionally the whole identity-check stage
for each tag in turn, writing run_summary.json with exit codes, timings and each
step's peak committed memory (all its processes together, and its largest one).
Inputs are read-only; every output goes under <root>/outputs.
"""
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

A1 = 'development/cd68_full_red_green/rigid_analysis/A1/tracking/engine'
INPUTS = 'development/cd68_full_red_green/rigid_batch/.j/A1/inputs'
CONFIG = INPUTS + '/motion_config.json'
TOOLS = ('tools/benchmark_completed_evidence.py', 'tools/benchmark_working_storage.py',
         'tools/benchmark_identity_reservation.py', 'tools/compare_tracking_memory.py', 'tools/full_replay_bundle.py')
REFERENCE = 'reference/evidence_hashes.json'
# The packing machine's auto_organotypic source, imported ahead of any installed copy.
SITE = 'site'
ONLINE_ONLY = 0x1000 | 0x40000 | 0x400000


def _sha(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _files(folder):
    for path in sorted(Path(folder).rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts:
            yield path


def _fingerprint(folder):
    """pymicroglia.tracking.shipped_engines.fingerprint, repeated because this runner travels alone."""
    digest = hashlib.sha256()
    for path in sorted(_files(folder), key=lambda p: p.relative_to(folder).as_posix()):
        if path.suffix != '.pyc':
            digest.update(('%s\0%s\n' % (path.relative_to(folder).as_posix(), _sha(path))).encode())
    return digest.hexdigest()


class _Limits(ctypes.Structure):
    """JOBOBJECT_EXTENDED_LIMIT_INFORMATION; only the two peaks at the end are read."""
    _fields_ = [('PerProcessUserTimeLimit', ctypes.c_int64), ('PerJobUserTimeLimit', ctypes.c_int64),
                ('LimitFlags', wintypes.DWORD), ('MinimumWorkingSetSize', ctypes.c_size_t),
                ('MaximumWorkingSetSize', ctypes.c_size_t), ('ActiveProcessLimit', wintypes.DWORD),
                ('Affinity', ctypes.c_size_t), ('PriorityClass', wintypes.DWORD), ('SchedulingClass', wintypes.DWORD),
                ('IoCounters', ctypes.c_ulonglong * 6), ('ProcessMemoryLimit', ctypes.c_size_t),
                ('JobMemoryLimit', ctypes.c_size_t), ('PeakProcessMemoryUsed', ctypes.c_size_t),
                ('PeakJobMemoryUsed', ctypes.c_size_t)]


def _memory_job(pid):
    """A Windows job object holding the process and every process it starts later, or None where unavailable."""
    if os.name != 'nt':
        return None
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateJobObjectW.restype = kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    kernel.QueryInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
                                                 ctypes.c_void_p)
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    job = kernel.CreateJobObjectW(None, None)
    process = kernel.OpenProcess(0x0100 | 0x0001, False, pid)  # set quota, terminate: what assignment needs
    joined = bool(job and process and kernel.AssignProcessToJobObject(job, process))
    if process:
        kernel.CloseHandle(process)
    if not joined:
        if job:
            kernel.CloseHandle(job)
        return None
    return kernel, job


def _peak_memory(handle):
    """Peak committed memory of the whole process tree and of its largest process, in GiB, then the job is closed."""
    if handle is None:
        return dict(peak_gb=None, peak_process_gb=None)
    kernel, job = handle
    limits = _Limits()
    read = kernel.QueryInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits), None)
    kernel.CloseHandle(job)
    if not read:
        return dict(peak_gb=None, peak_process_gb=None)
    return dict(peak_gb=round(limits.PeakJobMemoryUsed / 1024 ** 3, 2),
                peak_process_gb=round(limits.PeakProcessMemoryUsed / 1024 ** 3, 2))


def snapshot_engine(engine, snapshot, outputs, name):
    """The A1 run folder with the engine code the snapshot ships, when that differs from A1's own copy.

    The copy keeps A1's stage outputs and run config; only code/, source_hashes.json
    and archive.sha256 come from the snapshot's motion_engine folder (a zip in
    snapshots made before the engines shipped as folders).
    """
    shipped = snapshot / 'pymicroglia/tracking/data/motion_engine'
    archive = shipped.with_suffix('.zip')
    digest = _fingerprint(shipped) if shipped.is_dir() else _sha(archive)
    if (engine / 'archive.sha256').read_text().strip() == digest:
        return engine
    target = outputs / ('engine_' + name)
    marker = target / 'archive.sha256'
    if marker.is_file() and marker.read_text().strip() == digest:
        return target
    if target.exists():
        raise SystemExit('%s holds another engine; choose another --label' % target)
    shutil.copytree(engine, target, ignore=shutil.ignore_patterns(
        'code', 'source_hashes.json', 'archive.sha256', 'tracking.lock', '__pycache__'))
    if shipped.is_dir():
        shutil.copytree(shipped / 'code', target / 'code', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        shutil.copyfile(shipped / 'source_hashes.json', target / 'source_hashes.json')
    else:
        with zipfile.ZipFile(archive) as source:
            for member in source.namelist():
                if member.startswith('code/') or member == 'source_hashes.json':
                    source.extract(member, target)
    marker.write_text(digest + '\n')
    return target


def compare_history(reference, candidate):
    """Two identity-check stage outputs of the same machine: every saved table and image, byte for byte."""
    def outputs(root):
        return {p.relative_to(root / 'engine').as_posix(): p for p in (root / 'engine').rglob('*')
                if p.is_file() and p.suffix in ('.tif', '.tiff', '.csv')}
    left, right = outputs(reference), outputs(candidate)
    differing = sorted(name for name in left.keys() & right.keys() if _sha(left[name]) != _sha(right[name]))
    return dict(status='pass' if left and not differing and left.keys() == right.keys() else 'fail', files=len(left),
                differing=differing[:40], only_reference=sorted(left.keys() - right.keys())[:20],
                only_candidate=sorted(right.keys() - left.keys())[:20])


def _compare_row(outputs, kind, baseline, name):
    folder = 'evidence_full_' if kind == 'evidence' else 'history_'
    same = (compare if kind == 'evidence' else compare_history)(outputs / (folder + baseline), outputs / (folder + name))
    (outputs / ('compare_%s%s_to_%s.json' % ('' if kind == 'evidence' else 'history_', name, baseline))).write_text(
        json.dumps(same, indent=1) + '\n')
    return dict(same_as_baseline=same['status'], baseline=baseline, differing=same['differing'][:5])


def compare(reference, candidate):
    """Two replay outputs of the same machine, byte for byte: tables, boundaries, evidence, final hashes."""
    def tables(root):
        return {p.relative_to(root / 'tables').as_posix() for p in (root / 'tables').rglob('*') if p.is_file()}
    left, right = tables(reference), tables(candidate)
    differing = sorted(name for name in left & right
                       if (reference / 'tables' / name).read_bytes() != (candidate / 'tables' / name).read_bytes())
    old, new = (json.loads((root / 'report.json').read_text()) for root in (reference, candidate))
    checks = dict(tables=not differing and left == right,
                  boundaries=json.loads((reference / 'boundaries.json').read_text())
                  == json.loads((candidate / 'boundaries.json').read_text()),
                  evidence=json.loads((reference / 'evidence_hashes.json').read_text())
                  == json.loads((candidate / 'evidence_hashes.json').read_text()),
                  final_labels=old.get('final_label_sha256') == new.get('final_label_sha256'),
                  final_ledger=old.get('final_ledger_sha256') == new.get('final_ledger_sha256'))
    return dict(status='pass' if all(checks.values()) else 'fail', checks=checks, tables=len(left),
                differing=differing[:40], only_reference=sorted(left - right)[:20], only_candidate=sorted(right - left)[:20])


def pack(args):
    if not args.snapshots_only and args.reference_evidence is None:
        raise SystemExit('--reference-evidence is needed unless --snapshots-only')
    root = Path.cwd()
    # A snapshots-only bundle carries the replay tools too, but not this runner,
    # which travels beside the bundles and may be newer than an unpacked copy.
    chosen = [(root / name, name) for name in TOOLS if not (args.snapshots_only and name.endswith('full_replay_bundle.py'))]
    if not args.snapshots_only:
        chosen += [(path, path.relative_to(root).as_posix()) for folder in (A1, INPUTS) for path in _files(root / folder)]
    for snapshot in args.snapshot:
        chosen += [(path, path.relative_to(root).as_posix()) for path in _files(snapshot.resolve())]
    if not args.snapshots_only:
        chosen.append((args.reference_evidence.resolve(), REFERENCE))
        package = Path(__import__('auto_organotypic').__file__).parent
        chosen += [(path, SITE + '/auto_organotypic/' + path.relative_to(package).as_posix()) for path in _files(package)]
    offline = [name for path, name in chosen if getattr(os.stat(path), 'st_file_attributes', 0) & ONLINE_ONLY]
    if offline:
        raise SystemExit('Online-only inputs; make them local first: ' + ', '.join(offline[:5]))
    rows = []
    with zipfile.ZipFile(args.output, 'x', zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
        for path, name in chosen:
            bundle.write(path, name)
            rows.append(dict(name=name, bytes=path.stat().st_size, sha256=_sha(path)))
        manifest = dict(files=rows, snapshots=[s.resolve().relative_to(root).as_posix() for s in args.snapshot])
        if not args.snapshots_only:
            manifest.update(engine=A1, config=CONFIG, reference=REFERENCE)
        bundle.writestr('bundle_manifest.json', json.dumps(manifest, indent=1))
    print(json.dumps(dict(files=len(rows), bytes=sum(r['bytes'] for r in rows), bundle=str(args.output),
                          sha256=_sha(args.output))))


def run(args):
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    marker = root / 'bundle_manifest.json'
    if not marker.exists() and any(root.iterdir()):
        raise SystemExit('Root holds other files; choose a new folder')
    manifest = json.loads(marker.read_text()) if marker.exists() else dict(files=[], snapshots=[])
    known = {r['name']: r['sha256'] for r in manifest['files']}
    for path in args.bundle:
        with zipfile.ZipFile(path) as bundle:
            part = json.loads(bundle.read('bundle_manifest.json'))
            for r in part['files']:
                if known.get(r['name'], r['sha256']) != r['sha256']:
                    raise SystemExit('Bundles disagree on ' + r['name'])
                if r['name'] not in known:
                    bundle.extract(r['name'], root)
                    known[r['name']] = r['sha256']
                    manifest['files'].append(r)
        manifest['snapshots'] += [s for s in part['snapshots'] if s not in manifest['snapshots']]
        manifest.update({k: part[k] for k in ('engine', 'config', 'reference') if k in part})
        marker.write_text(json.dumps(manifest, indent=1))
    bad = [r['name'] for r in manifest['files'] if _sha(root / r['name']) != r['sha256']]
    if bad:
        raise SystemExit('Hash mismatch: ' + ', '.join(bad[:5]))
    outputs = root / 'outputs'
    outputs.mkdir(exist_ok=True)
    summary = dict(machine=os.environ.get('COMPUTERNAME'), python=sys.executable, verified_files=len(manifest['files']),
                   runs=[])
    engine, config = root / manifest['engine'], root / manifest['config']
    environment = dict(os.environ, PYTHONPATH=str(root / SITE))
    summary_path = outputs / ('run_summary%s.json' % args.label)
    # Earlier runs on this machine, compared with the baseline before anything runs.
    for name in args.compare_only:
        for kind, folder, check in (('evidence', 'evidence_full_', 'report.json'), ('history', 'history_', 'report.json')):
            if not ((outputs / (folder + name) / check).exists() and (outputs / (folder + args.baseline) / check).exists()):
                continue
            row = dict(tag=name, step='compare_' + kind, **_compare_row(outputs, kind, args.baseline, name))
            row['exit_code'] = 0 if row['same_as_baseline'] == 'pass' else 1
            summary['runs'].append(row)
            summary_path.write_text(json.dumps(summary, indent=1) + '\n')
            print(json.dumps(row), flush=True)
    for tag, history_only in [(t, False) for t in args.tag] + [(t, True) for t in args.history_only]:
        snapshot = next(root / s for s in manifest['snapshots'] if s.endswith('combined_%s_candidate' % tag))
        name = tag + args.label
        code = snapshot_engine(engine, snapshot, outputs, name) if args.engine_from_snapshot else engine
        steps = [] if history_only else [('evidence_and_corrections', [
            sys.executable, str(root / 'tools/benchmark_completed_evidence.py'), '--engine', str(code),
            '--snapshot', str(snapshot), '--config', str(config), '--output', str(outputs / ('evidence_full_' + name)),
            '--run', 'pymicroglia_95_full_A1', '--stem', '95_full_A1',
            '--reference', str(engine / 'm23_completed_feedback/pymicroglia_95_full_A1'),
            '--reference-evidence', str(root / manifest['reference'])] + (['--profile-corrections'] if args.profile else []))]
        if args.history or history_only:
            steps.append(('identity_check_stage', [
                sys.executable, str(root / 'tools/benchmark_identity_reservation.py'), '--engine', str(code),
                '--config', str(config), '--snapshot', str(snapshot), '--adapter', str(snapshot / 'pymicroglia/tracking'),
                '--output', str(outputs / ('history_' + name)), '--run', 'pymicroglia_95_full_A1', '--stem', '95_full_A1',
                '--native-history', '--stage', 'history'] + (['--profile'] if args.profile else [])))
        for label, command in steps:
            started = time.time()
            with open(outputs / ('%s_%s.log' % (label, name)), 'ab') as log:
                process = subprocess.Popen(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT)
                job = _memory_job(process.pid)
                code = process.wait()
            row = dict(tag=tag, step=label, exit_code=code, seconds=round(time.time() - started, 1), **_peak_memory(job))
            report = outputs / ('evidence_full_' + name) / 'report.json'
            if label == 'evidence_and_corrections' and report.exists():
                data = json.loads(report.read_text())
                row.update(status=data.get('status'), timings=data.get('timings'),
                           tables='%d/%d' % (sum(data['checks'].values()), len(data['checks'])),
                           final_labels=data.get('final_label_sha256'), final_ledger=data.get('final_ledger_sha256'))
            kind, folder = (('evidence', 'evidence_full_') if label == 'evidence_and_corrections' else ('history', 'history_'))
            if (args.baseline and args.baseline != name and (outputs / (folder + name) / 'report.json').exists()
                    and (outputs / (folder + args.baseline) / 'report.json').exists()):
                row.update(_compare_row(outputs, kind, args.baseline, name))
            summary['runs'].append(row)
            summary_path.write_text(json.dumps(summary, indent=1) + '\n')
            print(json.dumps(row), flush=True)
    return 0 if all(r['exit_code'] == 0 for r in summary['runs']) else 1


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('pack')
    q.add_argument('--snapshot', type=Path, action='append', required=True)
    q.add_argument('--reference-evidence', type=Path)
    q.add_argument('--output', type=Path, required=True)
    q.add_argument('--snapshots-only', action='store_true')
    q = sub.add_parser('run')
    q.add_argument('--bundle', type=Path, action='append', default=[])
    q.add_argument('--root', type=Path, required=True)
    q.add_argument('--tag', action='append', default=[])
    q.add_argument('--history', action='store_true')
    q.add_argument('--profile', action='store_true', help='cProfile the corrections and identity-check stage (slower; for scouting, not timing)')
    q.add_argument('--history-only', action='append', default=[], help='tag whose identity-check stage alone runs')
    q.add_argument('--compare-only', action='append', default=[], help='existing run name compared with --baseline')
    q.add_argument('--baseline', help='earlier run name on this machine to compare each new run with, byte for byte')
    q.add_argument('--label', default='', help='suffix for output folders, logs and summary of this run')
    q.add_argument('--engine-from-snapshot', action='store_true',
                   help="run the engine code each snapshot ships on a copy of the A1 run folder")
    a = p.parse_args()
    sys.exit(pack(a) if a.command == 'pack' else run(a))


if __name__ == '__main__':
    main()
