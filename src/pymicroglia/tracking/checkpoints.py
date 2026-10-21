"""Verified reuse of Motion's completed stage files and accepted-history work.

Array checkpoints reference existing TIFFs: no duplicate movie cache. Small
table sidecars preserve exact pandas values and types across interruption.
Resume only one's own run directory; table sidecars are trusted local pickles.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd
import scipy
import skimage
import tifffile

METHOD_VERSION = '2026-10-14-motion-stage-resume-1'
STAGES = {
    'stage_m0': 'm0_inputs', 'stage_m1': 'm1_motion', 'stage_m2': 'm2_anchor',
    'stage_m3': 'm3_track', 'stage_m4': 'm4_events', 'stage_m5': 'm5_reconcile',
    'stage_m7_accepted': 'm7_persistence', 'stage_m8_accepted': 'm8_merge',
    'stage_m9_accepted': 'm9_alias', 'stage_m12_accepted': 'm12_identity_reservation',
    'stage_m19_accepted': 'm19_layered_lineage', 'stage_m20_accepted': 'm20_oscillatory_lineage',
    'stage_m22_accepted_history': 'm22_accepted_history',
}


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf8')
    deadline = time.monotonic() + 10
    while True:
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(.2)


@contextmanager
def exclusive(path):
    """Do not let two processes resume or preserve the same tracking stage."""
    with Path(path).open('a+b') as handle:
        if handle.seek(0, 2) == 0:
            handle.write(b'0'); handle.flush()
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise FileExistsError('Another process is using this tracking run') from error
        try:
            yield
        finally:
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


class Stages:
    def __init__(self, pipeline, cfg, stem, run_name, loader):
        self.pipeline, self.cfg, self.stem = pipeline, cfg, stem
        self.root, self.run_name, self.loader = Path(pipeline.ROOT), run_name, loader
        self.root.mkdir(parents=True, exist_ok=True)
        self.reused = []
        # Verify real source bytes, rather than trusting paths or saved pins.
        pipeline._verify_pins(cfg, stem)
        code = {str(p.relative_to(self.root)): sha(p)
                for p in sorted((self.root / 'code').rglob('*.py'))}
        package = {p.name: sha(p) for p in (
            Path(__file__), *(Path(__file__).with_name(name) for name in (
                'engine.py', 'engine_bounded.py', 'frozen_io.py', 'detector_acceleration.py',
                'completed_feedback.py', 'completed_evidence.py')))}
        from importlib import resources
        correction = resources.files('pymicroglia.tracking').joinpath('data/completed_feedback_engine.zip')
        package['completed_feedback_engine.zip'] = sha(correction)
        import inspect
        from auto_organotypic.array_storage import Pool
        package['array_storage.py'] = sha(inspect.getfile(Pool))
        runtime = {}
        for name in ('r12_runtime', 'r13_cache_runtime', 'r13_copy_runtime'):
            source = getattr(sys.modules.get(name), '__file__', None)
            if source:
                runtime[name] = sha(source)
        self.signature = dict(method=METHOD_VERSION, stem=stem, config_sha256=sha(cfg.path),
            code=code, adapter=package, runtime=runtime, software=dict(python=platform.python_version(),
            numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__,
            skimage=skimage.__version__, tifffile=tifffile.__version__))
        self.state_path = self.root / 'tracking_resume.json'
        if self.state_path.is_file():
            previous = json.loads(self.state_path.read_text())
            if previous['signature'] != self.signature:
                raise ValueError('Tracking checkpoint inputs, settings, software or code changed; use a fresh folder')
        self.state = dict(signature=self.signature, run=run_name, reused=[], status='running')
        write(self.state_path, self.state)

    def _file(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root.resolve()) or any(
                parent.is_symlink() for parent in (self.root / relative, *(self.root / relative).parents)
                if parent.is_relative_to(self.root)):
            raise ValueError('Checkpoint file lies outside the owned engine directory')
        return path

    def _verify(self, files):
        for relative, expected in files.items():
            path = self._file(relative)
            if not path.is_file() or sha(path) != expected:
                raise ValueError('Tracking checkpoint file changed or is incomplete: ' + relative)

    def _load(self, value):
        if isinstance(value, list):
            return tuple(self._load(v) for v in value)
        if not isinstance(value, dict):
            return value
        if 'array' in value:
            array = self.loader(self._file(value['array']))
            if list(array.shape) != value['shape']:
                raise ValueError('Checkpoint array shape changed')
            return array.astype(np.dtype(value['dtype']), copy=False)
        if 'table' in value:
            # Verified hash and owned path are checked before unpickling.
            return pd.read_pickle(self._file(value['table']))
        if 'set' in value:
            return set(value['set'])
        if 'path' in value:
            return Path(value['path'])
        return {k: self._load(v) for k, v in value.items()}

    def _capture(self, directory, manifest, result, history, method):
        files = {}
        for output in manifest['outputs'].values():
            path = Path(output['path']).resolve()
            relative = str(path.relative_to(self.root.resolve()))
            files[relative] = output['sha256']
        tables = 0

        def array(value, role):
            if value is None:
                return None
            path = Path(manifest['outputs'][role]['path']).resolve()
            return dict(array=str(path.relative_to(self.root.resolve())),
                        dtype=str(value.dtype), shape=list(value.shape))

        def small(value):
            nonlocal tables
            if isinstance(value, pd.DataFrame):
                path = directory / f'checkpoint_table_{tables}.pkl'
                tables += 1
                value.to_pickle(path)
                relative = str(path.relative_to(self.root))
                files[relative] = sha(path)
                return dict(table=relative)
            if isinstance(value, Path):
                return dict(path=str(value))
            if isinstance(value, set):
                return dict(set=sorted(value))
            if isinstance(value, dict):
                return {k: small(v) for k, v in value.items()}
            if isinstance(value, tuple):
                return [small(v) for v in value]
            return value

        if method in ('stage_m0', 'stage_m1', 'stage_m5'):
            saved = small(result)
        elif method == 'stage_m2':
            # The bounded caller never consumes gained/lost full movies.
            saved = [array(result[0], 'observations'), small(result[1]), int(result[2]), None, None]
        elif method == 'stage_m3':
            saved = [array(result[0], 'initial_identities'), small(result[1])]
        elif method == 'stage_m4':
            saved = [array(result[0], 'event_identities'), small(result[1]),
                     array(result[2], 'inferred_pixels'), array(result[3], 'unresolved_pixels')]
        elif method == 'stage_m7_accepted':
            saved = [array(result[0], 'labels'), small(result[1])]
        elif method == 'stage_m22_accepted_history':
            saved = [array(result[0], 'labels'), array(result[1], 'unclaimed'),
                     array(result[2], 'inferred'), small(result[3])]
        else:
            saved = [array(result[0], 'labels'), array(result[1], 'inferred'), small(result[2])]
        attributes = {}
        if history is not None:
            # Save only this stage's side effects. Earlier evidence is restored
            # by its own checkpoint before the next stage consumes it.
            if method == 'stage_m12_accepted':
                attributes = dict(reservation_labels=array(history.reservation_labels, 'labels'),
                    pre_reservation_labels=array(history.pre_reservation_labels, 'pre_reservation_labels'),
                    host_audit=small(history.host_audit), motion_reserved=small(history.motion_reserved),
                    motion_reservation_events=small(history.motion_reservation_events))
            elif method == 'stage_m19_accepted':
                attributes = dict(physical_somas=array(history.physical_somas, 'physical_somas'))
        self._verify(files)
        receipt = dict(signature=self.signature, result=saved, history=attributes, files=files)
        write(directory / 'checkpoint.json', receipt)
        return receipt

    def call(self, method, *args, **kwargs):
        directory = self.root / STAGES[method] / self.run_name
        marker = directory / 'checkpoint.json'
        history = kwargs.get('history_inputs')
        if marker.is_file():
            receipt = json.loads(marker.read_text())
            if receipt['signature'] != self.signature:
                raise ValueError('Tracking stage checkpoint has different inputs, settings or code')
            self._verify(receipt['files'])
            for name, value in receipt['history'].items():
                setattr(history, name, self._load(value))
            self.reused.append(STAGES[method])
            self.state.update(reused=self.reused, current_stage=STAGES[method], status='resuming')
            write(self.state_path, self.state)
            print('CHECKPOINT reused', STAGES[method], flush=True)
            return self._load(receipt['result'])
        if directory.exists():
            try:
                old = json.loads((directory / 'run.json').read_text())
            except (OSError, ValueError):
                old = {}  # A hard stop may interrupt Motion's own metadata write.
            # Metadata-only stages from an earlier adapter need no array/table
            # reconstruction. The same source pins have just been verified.
            if method in ('stage_m0', 'stage_m1') and old.get('status') == 'done':
                sources = self.pipeline._pinned_paths(self.cfg, self.stem)
                source_record = self.root / 'm0_inputs' / self.run_name / 'out' / f'{self.stem}_sources.json'
                if json.loads(source_record.read_text())['config_sha256'] != self.signature['config_sha256']:
                    raise ValueError('Earlier Motion input settings differ')
                result = sources if method == 'stage_m0' else None
                self._capture(directory, old, result, None, method)
                return self.call(method, *args, **kwargs)
            archived = directory.with_name(directory.name + '_interrupted_' + str(time.time_ns()))
            directory.rename(archived)
            if method == 'stage_m22_accepted_history' and (archived / 'mid/accepted_history').is_dir():
                kwargs['checkpoint_run_name'] = archived.name
        self.state.update(current_stage=STAGES[method], status='running')
        write(self.state_path, self.state)
        result = getattr(self.pipeline, method)(*args, **kwargs)
        manifest = json.loads((directory / 'run.json').read_text())
        if manifest.get('status') != 'done':
            raise RuntimeError('Motion stage did not finish; no checkpoint published')
        self._capture(directory, manifest, result, history, method)
        return result
