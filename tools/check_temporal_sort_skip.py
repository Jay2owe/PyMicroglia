"""temporally_supported_points with the in-order sort skip against the original, on shuffled and reversed tables."""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'development/cd68_full_red_green/rigid_analysis/A1/tracking/engine'
FROZEN = ROOT / 'validation/completed_base_20261012/runtime_v2/completed_feedback_engine/frozen'
sys.path[:0] = [str(ENGINE / 'code'), str(ROOT / 'validation/completed_base_20261012/memory/snapshots/identity_candidate')]
import common
import pipeline
import raw_support_memo as m

FRAMES = 20
cfg = common.Config.load(ROOT / 'development/cd68_full_red_green/rigid_batch/.j/A1/inputs/motion_config.json')
raw = tifffile.imread(pipeline._pinned_paths(cfg, '95_full_A1')['registered_raw'])[:FRAMES]
points = pd.read_csv(ENGINE / 'm22_accepted_history/pymicroglia_95_full_A1/mid/accepted_history/26_latent_body_tracks/out/latent_track_points.csv', low_memory=False)
points = points[points.frame < FRAMES].reset_index(drop=True)
rng = np.random.default_rng(5)
tables = dict(recorded=points, shuffled=points.sample(frac=1, random_state=3),
              reversed=points.iloc[::-1], float_frames=points.assign(frame=points.frame.astype(float)))
report = {}
for version in ('recovery/r194_a003', 'recovery/r193_a002', 'recovery/r189_a001'):
    sys.path.insert(0, str(FROZEN / version))
    for name in ('recovery_temporal_soma_support', 'recovery_qualified_soma', 'recovery_destination_soma'):
        sys.modules.pop(name, None)
    module = importlib.import_module('recovery_temporal_soma_support')
    sys.path.pop(0)
    memo = m.Memo()
    rewritten = m._temporal_text(module, module.temporally_supported_points, memo)
    assert rewritten is not None, version
    for name, table in tables.items():
        for kwargs in ({}, dict(centre_connected=True)):
            if kwargs and 'centre_connected' not in module.temporally_supported_points.__code__.co_varnames:
                continue
            outcome = []
            for function in (module.temporally_supported_points, rewritten):
                try:
                    outcome.append(function(raw, table, **kwargs))
                except Exception as error:
                    outcome.append(('error', type(error).__name__, str(error)))
            a, b = outcome
            same = (isinstance(a, tuple) and a == b) or (
                isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame) and a.index.equals(b.index)
                and a.equals(b) and list(a.dtypes) == list(b.dtypes))
            report[f'{version} {name} {kwargs}'] = bool(same)
    report[f'{version} stats'] = {k: v for k, v in memo.stats.items() if k.startswith('temporal')}
out = ROOT / "validation/completed_base_20261012/memory/temporal_sort_skip_checks.json"
report["status"] = "pass" if all(v for k, v in report.items() if not k.endswith("stats")) else "fail"
out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
print(json.dumps(report, indent=1))
assert all(v for k, v in report.items() if not k.endswith('stats'))
