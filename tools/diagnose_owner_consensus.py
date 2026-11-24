"""Replay saved owner-consensus inputs without rerunning physical detection."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import tifffile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('engine', type=Path)
    parser.add_argument('config', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--repair', action='store_true')
    args = parser.parse_args()
    root = args.engine.resolve()
    sys.path.insert(0, str(root/'code'))
    import owner_consensus as module
    from component_accounting import component_excess
    stage = next((root/'m22_accepted_history').glob('*/mid/accepted_history'))
    labels_path = stage/'15_issue030_moderate_gap_recovery.tif'
    points_path = stage/'26_latent_body_tracks/out/latent_track_points.csv'
    labels = tifffile.imread(labels_path)
    params = json.loads(args.config.read_text())['accepted_postprocessing']['field_wide_owner_consensus']['owner_consensus']
    attached = module.attach_owners(pd.read_csv(points_path), labels, params)
    proposals = module.discover_proposals(attached, params)
    if args.repair:
        from pymicroglia.tracking.owner_consensus_conflicts import repaired_function
        _, patched = repaired_function(Path(module.__file__).read_text())
        exec(patched, module.__dict__)
    candidate, audit = module.apply_dependency_ordered_proposals(labels, attached, proposals, params)
    duplicates = []
    for frame in range(len(labels)):
        changed = labels[frame] != candidate[frame]
        owners = np.unique(np.concatenate([labels[frame][changed], candidate[frame][changed]]))
        for owner in owners[owners > 0]:
            before, after = component_excess(labels[frame], owner), component_excess(candidate[frame], owner)
            if after <= before: continue
            relevant = audit[(audit.outcome == 'applied') & (audit.first_frame <= frame) & (audit.last_frame >= frame)]
            duplicates.append(dict(frame=int(frame), owner=int(owner), before=before, after=after, applied=relevant.to_dict('records')))
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    report = dict(inputs={str(p):sha(p) for p in (labels_path,points_path,args.config,Path(module.__file__))}, repaired=args.repair,
        duplicates=duplicates, audit=audit.to_dict('records'), changed_pixels=int(np.count_nonzero(candidate != labels)),
        foreground_unchanged=bool(np.array_equal(labels>0,candidate>0)))
    args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(duplicates=duplicates,changed_pixels=report['changed_pixels'],outcomes=audit.outcome.value_counts().to_dict())))


if __name__ == '__main__': main()
