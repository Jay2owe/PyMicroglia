"""Recorded Issue 005 tracking passes: recorded by the accepted history, reused by correction evidence, exactly.

Runs Issue 005 as run_accepted_history does (inputs decoded from the m12 and
m19 history checkpoints, raw and lag from the pinned sources, on the calling
thread under the native adapters) with the recorder active. Issue 005's result
must equal the reference 01_issue005.tif. The correction-evidence side then
builds its gap-reference inputs as completed_evidence does, and the recorded
passes must load (identical input digest) with the fingerprints the reference
evidence holds for r213's prior traces. A changed input, a changed setting and
a changed pass file must each be refused.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import time


def main():
    p = argparse.ArgumentParser()
    for name in ('engine', 'snapshot', 'config', 'output', 'reference_evidence'):
        p.add_argument('--' + name.replace('_', '-'), dest=name, type=Path, required=True)
    p.add_argument('--run', required=True)
    p.add_argument('--stem', required=True)
    p.add_argument('--keep', type=Path, help='copy the verified pass folder here')
    a = p.parse_args()
    for name in ('engine', 'snapshot', 'config', 'output', 'reference_evidence'):
        setattr(a, name, getattr(a, name).resolve())
    platform.platform = lambda *x, **k: 'Windows-tracking-pass-record'
    import numpy as np
    import pandas as pd
    import tifffile
    a.output.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(a.snapshot), str(a.engine / 'code')]
    from pymicroglia.tracking import frozen_io, tracking_pass_record as record
    from pymicroglia.tracking.correction_audit import array_sha256
    from pymicroglia.tracking.checkpoints import Stages
    from benchmark_working_storage import load
    import accepted_history as ah
    import common, optimized_pipeline, pipeline
    storage = load(a.snapshot / 'array_storage.py')
    cfg = common.Config.load(a.config)
    reference = json.loads(a.reference_evidence.read_text())
    history_root = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
    work = a.output / 'output_root'
    work.mkdir()
    report = dict(status='fail')

    def execute():
        pool = storage.Pool(a.output / 'scratch2', minimum_bytes=1024**2, temporary_ram_bytes=32 * 1024**2)
        with storage.install(pool), frozen_io.install(common, a.engine / 'code', a.output / 'scratch3', array_pool=pool):
            # The native adapters installed the recorder around Issue 005.
            assert 'tracking_pass_record' in ah._issue005.__code__.co_filename
            sources = pipeline._pinned_paths(cfg, a.stem)

            def read(path):
                target = a.output / 'scratch2' / ('input_' + hashlib.sha1(str(path).encode()).hexdigest()[:10] + '.dat')
                with tifffile.TiffFile(path) as opened:
                    value = pool.adopt(opened.series[0].asarray(out=str(target)), target)
                value.flags.writeable = False
                return value

            # Earlier group: Issue 005 as run_accepted_history calls it.
            decoder = Stages.__new__(Stages); decoder.root = a.engine; decoder.loader = tifffile.imread
            inputs = ah.AcceptedHistoryInputs()
            for prior in ('m12_identity_reservation', 'm19_layered_lineage'):
                manifest = json.loads((a.engine / prior / a.run / 'checkpoint.json').read_text())
                decoder._verify(manifest['files'])
                for name, value in manifest['history'].items():
                    setattr(inputs, name, decoder._load(value))
            physical, reservation, _, _, motion_reserved = inputs.required()
            observations, observation_table, anchor_t = pipeline._load_anchor_inputs(a.stem, a.run)
            raw_full, lag_full = read(sources['registered_raw']), read(sources['lag_float'])
            token = record._output_root.set(work)
            tick = time.perf_counter()
            try:
                labels = ah._issue005(cfg, physical, reservation, raw_full, lag_full, observations,
                                      observation_table, anchor_t, motion_reserved)
            finally:
                record._output_root.reset(token)
            report['issue005_seconds'] = time.perf_counter() - tick
            report['issue005_equal'] = array_sha256(labels) == array_sha256(tifffile.imread(history_root / '01_issue005.tif'))
            folder = work / record.FOLDER
            report['recorded'] = (folder / 'manifest.json').is_file()
            report['recorded_bytes'] = sum(f.stat().st_size for f in folder.iterdir())

            # Correction evidence: the gap-reference inputs as completed_evidence builds them.
            read_plain = tifffile.imread
            lineage = a.engine / 'm19_layered_lineage' / a.run
            e_physical = read_plain(lineage / 'mid' / f'{a.stem}_physical_somas.tif')
            e_reservation = read_plain(a.engine / 'm12_identity_reservation' / a.run / 'out' / f'{a.stem}.tif')
            e_raw = read_plain(sources['registered_raw']); e_lag = read_plain(sources['lag_float'])
            e_observations, e_table, e_anchor = pipeline._load_anchor_inputs(a.stem, a.run)
            e_reserved = ah._field_motion_reservations(a.run)
            args = (cfg, e_physical, e_reservation, e_raw, e_lag, e_observations, e_table, e_anchor, e_reserved)
            tick = time.perf_counter()
            loaded = record.load(work, ah, *args)
            report['load_seconds'] = time.perf_counter() - tick
            report['loaded'] = loaded is not None
            if loaded is not None:
                report['pass_fingerprints'] = [
                    array_sha256(movie) == reference[f'evidence/final_evidence/r213_a010/traces/{i}/0']['sha256']
                    and movie.dtype.str == reference[f'evidence/final_evidence/r213_a010/traces/{i}/0']['dtype']
                    for i, (movie, _) in enumerate(loaded[0])]
                report['pass_types'] = [type(movie).__name__ for movie, _ in loaded[0]]
            refused = {}
            refused['reserved_name'] = record.load(work, ah, *args[:8], set(e_reserved) | {65000}) is None
            settings = common.Config.load(a.config)
            settings.values['accepted_postprocessing'] = json.loads(json.dumps(settings.values['accepted_postprocessing']))
            settings.values['accepted_postprocessing']['_changed'] = 1
            refused['setting'] = record.load(work, ah, settings, *args[1:]) is None
            changed_physical = np.array(e_physical); changed_physical.flat[12345] ^= 1
            refused['physical_pixel'] = record.load(work, ah, cfg, changed_physical, *args[2:]) is None
            tampered = a.output / 'tampered'
            shutil.copytree(folder, tampered / record.FOLDER)
            movie = tifffile.imread(tampered / record.FOLDER / 'pass2_labels.tif'); movie[7, 3, 3] ^= 1
            tifffile.imwrite(tampered / record.FOLDER / 'pass2_labels.tif', movie, compression='zlib')
            refused['pass_file'] = record.load(tampered, ah, *args) is None
            report['refused'] = refused
            report['stats'] = dict(record.stats)

    optimized_pipeline.main(['--no-evidence-cache'], runner=execute)
    ok = (report.get('issue005_equal') and report.get('recorded') and report.get('loaded')
          and all(report.get('pass_fingerprints', [False])) and len(report.get('pass_fingerprints', [])) == 2
          and all(report['refused'].values()))
    report['status'] = 'pass' if ok else 'fail'
    report['sources'] = {str(f): hashlib.sha256(Path(f).read_bytes()).hexdigest()
                         for f in (Path(__file__), Path(record.__file__))}
    if ok and a.keep is not None:
        target = a.keep / record.FOLDER
        if target.exists():
            raise FileExistsError(target)
        shutil.copytree(work / record.FOLDER, target)
        report['kept'] = str(target)
    (a.output / 'check.json').write_text(json.dumps(report, indent=1, default=str) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'sources'}, default=str))
    if not ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
