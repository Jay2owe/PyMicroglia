"""Compare indexed tenure tracklets with the frozen all-tracklet scan, exactly."""
import argparse
import json
from pathlib import Path
import platform
import sys
import time


def signature(result):
    frames, tracklets = result
    components = [(c.frame, c.observed_identity, c.local_component, c.area, c.x, c.y,
                   c.tracklet, c.local_age, c.pixels.tobytes()) for frame in frames for c in frame]
    tracks = [(number, track.number, track.observed_identity,
               [(c.frame, c.local_component) for c in track.components])
              for number, track in tracklets.items()]
    return components, tracks


def main():
    p = argparse.ArgumentParser()
    for name in ('engine', 'adapter', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--run', default='pymicroglia_95_full_A1')
    p.add_argument('--stacks', nargs='+', default=['03_issue007.tif', '05_issue009.tif', '12_issue023_generalized.tif'])
    a = p.parse_args()
    platform.platform = lambda *x, **k: 'Windows-tenure-tracklet-check'
    import numpy as np
    import tifffile
    sys.path[:0] = [str(a.adapter.resolve()), str((a.engine / 'code').resolve())]
    import accepted_history as history
    import tenure_geometry_reuse as reuse
    history._issue_module(history.I008, 'm16_persistent_quarantine')
    helper = sys.modules['m12_component_tenure_audit']
    frozen = helper.build_tracklets
    reference = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
    results = {}
    for name in a.stacks:
        movie = tifffile.imread(reference / name)
        with reuse.geometry(helper) as stats:
            assert stats['supported'] and helper.build_tracklets is not frozen
            tick = time.perf_counter(); expected = frozen(movie); before = time.perf_counter() - tick
            tick = time.perf_counter(); actual = helper.build_tracklets(movie); after = time.perf_counter() - tick
            # A tighter gap and step exercise the pruning and ties.
            narrow = (frozen(movie[:120], 9.0, 1), helper.build_tracklets(movie[:120], 9.0, 1))
        assert signature(expected) == signature(actual), name
        assert signature(narrow[0]) == signature(narrow[1]), name + ' narrow'
        results[name] = dict(before_seconds=before, after_seconds=after, ties=stats.get('tracklet_ties', 0),
                             tracklets=len(expected[1]), exact=True)
        print(name, results[name], flush=True)
    a.output.write_text(json.dumps(dict(status='pass', cases=results), indent=2) + '\n')


if __name__ == '__main__':
    main()
