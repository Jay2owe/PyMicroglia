"""Compare ephemeral speckle alias discovery with once-per-call evidence rows.

discover and apply run on the accepted A1 input of stage 100 (accepted and
loosened settings; plain and memory-pool label movies) behind the presence
adapter the pipeline installs first, with and without speckle_evidence_rows;
audits, internal rows, attached points, labels, applications, details and
warnings must be identical. `_evidence_rows` outside a discover call must be
the original's answer.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'development/cd68_full_red_green/rigid_analysis/A1/tracking/engine'
HISTORY = ENGINE / 'm22_accepted_history/pymicroglia_95_full_A1/mid/accepted_history'
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_projection_chain_rows import same, outcome  # noqa: E402

LOOSE = dict(minimum_durable_support_movie_fraction=0.3, minimum_durable_owner_purity=0.8,
             maximum_alias_span_movie_fraction=0.5, maximum_alias_physical_support_movie_fraction=0.2,
             minimum_fragmented_owner_frame_fraction=0.1,
             maximum_component_destination_gap_equivalent_radii=6.0,
             maximum_component_destination_area_fraction=0.9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    a = parser.parse_args()
    sys.path[:0] = [str(ENGINE / 'code'), str(a.adapters.resolve())]
    import array_storage
    import ephemeral_speckle_alias_retirement as retirement
    import ephemeral_speckle_alias_retirement_integration as integration
    import speckle_alias_presence
    import speckle_evidence_rows as adapter
    config = json.loads((ENGINE / 'config.json').read_text(encoding='utf-8'))
    _, params = integration._configured(config)
    loose = {**params, **LOOSE}
    a.scratch.mkdir(parents=True, exist_ok=True)
    pool = array_storage.Pool(a.scratch / 'pool')
    kept = []

    def pooled(stack):
        path = a.scratch / f'stack_{len(kept)}.dat'
        mapped = np.memmap(path, dtype=stack.dtype, mode='w+', shape=stack.shape)
        mapped[:] = stack
        mapped.flush()
        working = pool.adopt(mapped, path)
        kept.append(working)
        return working

    labels = tifffile.imread(HISTORY / '99_well_b3_issue013_anchored_projection_owner_relay.tif')
    points = pd.read_csv(HISTORY / '26_latent_body_tracks/out/latent_track_points.csv')
    report = dict(failures=[], stages={}, questions=0)

    def stage():
        audit, internal, attached = retirement.discover(labels_in, points.copy(), settings)
        return audit, internal, attached, retirement.apply(labels_in, internal)

    for name, settings in (('accepted', params), ('loose', loose)):
        for form in ('plain', 'pool'):
            labels_in = labels.copy() if form == 'plain' else pooled(labels)
            with speckle_alias_presence.install(retirement) as front:
                assert front['supported'], front
                t0 = time.perf_counter()
                expected = outcome(stage)
                t1 = time.perf_counter()
                with adapter.install(retirement) as stats:
                    assert stats['supported'], stats
                    actual = outcome(stage)
                t2 = time.perf_counter()
            problem = same(expected, actual)
            audit = expected[1][0] if expected[0] == 'ok' else None
            key = f'{name}:{form}'
            report['stages'][key] = dict(
                status=expected[0], audit_rows=None if audit is None else len(audit),
                eligible=None if audit is None or 'eligible' not in audit else int(audit.eligible.astype(bool).sum()),
                warnings=len(expected[2]), original_seconds=round(t1 - t0, 2),
                adapted_seconds=round(t2 - t1, 2), stats=dict(stats))
            print(key, report['stages'][key], problem, flush=True)
            if problem:
                report['failures'].append(dict(stage=key, problem=problem))

    attached = retirement.attach_owners(points, labels).sort_values(['track_id', 'frame']).reset_index(drop=True)
    groups = [group for _, group in attached.groupby('track_id', sort=True)][:300]
    with adapter.install(retirement) as stats:
        for group in groups:
            report['questions'] += 1
            problem = same(outcome(retirement._evidence_rows.__wrapped__, group),
                           outcome(retirement._evidence_rows, group))
            if problem:
                report['failures'].append(dict(outside_scope=True, problem=problem))
        outside = dict(stats)
    report['outside_stats'] = outside
    restored = not hasattr(retirement._evidence_rows, '__wrapped__')
    report['restored'] = restored
    report['status'] = 'pass' if not report['failures'] and restored and outside['filtered'] == 0 else 'fail'
    report['failures'] = report['failures'][:20]
    a.output.write_text(json.dumps(report, indent=1, default=str), encoding='utf-8')
    print(json.dumps(dict(status=report['status'], questions=report['questions'], failures=report['failures'])))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
