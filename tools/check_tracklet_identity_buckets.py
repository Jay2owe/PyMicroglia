"""Compare per-identity tracklet building with Issue 009's build_tracklets, exactly.

Components of real accepted-history movies (the Issue 009 input and a later
stack) are built with the module's own frame_components; tracklets are built
with the frozen function and with the adapter (overlap reuse installed in both
runs, as in the history), at the movie-authority and hierarchy settings and
wider gaps and radii. Every tracklet's number, identity and node sequence must
match.
"""
import argparse
import json
import sys
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    for name in ('engine', 'adapter', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--run', default='pymicroglia_95_full_A1')
    p.add_argument('--frames', type=int, default=160)
    a = p.parse_args()
    import tifffile
    sys.path[:0] = [str(a.adapter.resolve()), str((a.engine / 'code' / 'corrections' / 'hierarchy').resolve()),
                    str((a.engine / 'code').resolve())]
    import lineage_hierarchy as geometry
    import hierarchy_overlap_reuse
    import tracklet_identity_buckets as adapter
    folder = a.engine / 'm22_accepted_history' / a.run / 'mid/accepted_history'
    settings = [dict(maximum_tracklet_gap_frames=2, same_identity_tracklet_radius_px=30.0),
                dict(maximum_tracklet_gap_frames=0, same_identity_tracklet_radius_px=5.0),
                dict(maximum_tracklet_gap_frames=8, same_identity_tracklet_radius_px=80.0)]
    report = dict(cases=[], seconds=dict(original=0.0, adapted=0.0))
    failures = []

    def shape(tracklets):
        return [(t.number, t.identity, [(n.frame, n.identity, n.number) for n in t.nodes]) for t in tracklets]

    with hierarchy_overlap_reuse.install(geometry) as overlap_stats:
        assert overlap_stats['supported']
        for stack_name, minimum in (('04_issue008.tif', 20), ('11_issue017_rebase.tif', 1)):
            stack = tifffile.imread(folder / stack_name)[:a.frames]
            frames = [geometry.frame_components(stack[t], t, minimum) for t in range(len(stack))]
            for cfg in settings:
                tick = time.perf_counter()
                expected = shape(geometry.build_tracklets(frames, cfg))
                middle = time.perf_counter()
                with adapter.install(geometry) as stats:
                    assert stats['supported']
                    actual = shape(geometry.build_tracklets(frames, cfg))
                    assert stats['calls'] == 1
                end = time.perf_counter()
                report['seconds']['original'] += middle - tick
                report['seconds']['adapted'] += end - middle
                report['cases'].append(dict(stack=stack_name, cfg=cfg, tracklets=len(expected),
                                            nodes=sum(len(t[2]) for t in expected)))
                if expected != actual:
                    failures.append(dict(stack=stack_name, cfg=cfg))
    report['status'] = 'pass' if not failures else 'fail'
    report['failures'] = failures
    a.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(report))
    return 0 if not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
