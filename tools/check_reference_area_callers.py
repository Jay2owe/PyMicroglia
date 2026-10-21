"""Compare the three direct _reference_area callers' correct() outputs, exactly.

recover_flanked_large_contact_interruptions, recover_terminal_name_switch_contacts_a002
and recover_persistent_resident_contacts each compute reference areas through
recover_stable_dual_soma_contacts._reference_area. On a real accepted label
movie, raw movie and tracker points (first frames only), each caller's complete
correct() result (labels and tables) under reference_area_boxes.install_loaded
must equal the original's, and the input labels must be left unchanged.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

from check_reference_area_boxes import same

CALLERS = ('recover_flanked_large_contact_interruptions',
           'recover_terminal_name_switch_contacts_a002',
           'recover_persistent_resident_contacts')


def main():
    p = argparse.ArgumentParser()
    for name in ('engine', 'adapter', 'code', 'config', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--stem', default='95_full_A1')
    p.add_argument('--run', default='pymicroglia_95_full_A1')
    p.add_argument('--frames', type=int, default=400)
    a = p.parse_args()
    import numpy as np
    import pandas as pd
    import tifffile
    sys.path[:0] = [str((a.engine / 'code').resolve())]
    import common
    import pipeline
    cfg = common.Config.load(a.config)
    raw = tifffile.imread(pipeline._pinned_paths(cfg, a.stem)['registered_raw'])[:a.frames]
    folder = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
    labels = tifffile.imread(folder / '99_well_b3_issue013_anchored_projection_owner_relay.tif')[:a.frames]
    points = pd.read_csv(folder / '26_latent_body_tracks/out/latent_track_points.csv', low_memory=False)
    points = points[points.frame < a.frames].reset_index(drop=True)
    protected = np.zeros(labels.shape, bool)
    sys.path[:0] = [str(a.code.resolve()), str(a.adapter.resolve())]
    import reference_area_boxes as adapter
    modules = {name: importlib.import_module(name) for name in CALLERS}
    digest = hashlib.sha256(labels.tobytes()).hexdigest()
    report = dict(status='pass', frames=a.frames, callers={})
    for name, module in modules.items():
        stats = dict(calls=0, labelled=0, reused=0, fallback=0)
        tick = time.perf_counter()
        expected = module.correct(labels, raw, points, protected)
        middle = time.perf_counter()
        with adapter.install_loaded(stats):
            assert module.correct.__wrapped__
            actual = module.correct(labels, raw, points, protected)
        end = time.perf_counter()
        assert not hasattr(module.correct, '__wrapped__')
        assert stats['calls'] == 1 and stats['fallback'] == 0, stats
        assert same(list(expected), list(actual)), name
        assert hashlib.sha256(labels.tobytes()).hexdigest() == digest, name
        report['callers'][name] = dict(outputs=len(expected), changed_pixels=int(np.count_nonzero(
            expected[0] != labels)), seconds=dict(original=middle - tick, adapted=end - middle), stats=stats)
        print(name, report['callers'][name], flush=True)
    a.output.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
