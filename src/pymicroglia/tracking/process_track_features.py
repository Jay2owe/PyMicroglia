"""Process inventory track features from one pass over the scored table.

process_inventory_tracked._track_features grouped the scored seats by track
and, per track, filtered visible rows, cast owners to int and ran
value_counts. The same numbers come from whole-table arrays: groups are the
sorted non-missing track ids (pd.factorize(sort=True), as groupby keys them),
casts are the same element-wise casts on the same rows, and per-track owner
counts come from one np.unique over (track, owner) keys. A track whose top
owner count is tied keeps value_counts on its owners in row order, so the
dominant owner is chosen exactly as before. Track ids that are not numeric,
casts that fail and very large owner values keep the original function.
"""
from contextlib import contextmanager

import numpy as np
import pandas as pd

try:
    from .module_scan import file_sha256 as _file_sha256
except ImportError:
    from module_scan import file_sha256 as _file_sha256

SOURCE = '6864e0175fb3450ef76d97d5fd0e915399587b0b132f8713eae0f3b211f21b36'


def _features(scored, stats):
    track = scored['track_id']
    if track.dtype.kind not in 'iuf':
        return None
    codes, keys = pd.factorize(track, sort=True)
    groups = len(keys)
    kept = codes >= 0
    rows = scored if kept.all() else scored[kept]
    codes = codes[kept]
    visible_mask = rows.physically_visible.astype(bool)
    visible = visible_mask.to_numpy(dtype=bool)
    owners = rows[visible_mask].accepted_owner.astype(int).to_numpy()
    strong = rows.strong.astype(bool).to_numpy(dtype=bool)
    positive = owners > 0
    owner_codes = codes[visible][positive]
    owners = owners[positive]
    frames = np.bincount(codes, minlength=groups)
    strong_frames = np.bincount(codes[strong], minlength=groups)
    total = np.bincount(owner_codes, minlength=groups)
    distinct = np.zeros(groups, np.int64)
    best = np.zeros(groups, np.int64)
    leaders = np.zeros(groups, np.int64)
    dominant = np.zeros(groups, np.int64)
    if len(owners):
        if int(owners.min()) < 1 or int(owners.max()) >= 2**31 or groups >= 2**31:
            return None
        width = int(owners.max()) + 1
        pairs, counts = np.unique(owner_codes.astype(np.int64) * width + owners.astype(np.int64),
                                  return_counts=True)
        pair_codes, pair_owners = pairs // width, pairs % width
        distinct = np.bincount(pair_codes, minlength=groups)
        np.maximum.at(best, pair_codes, counts)
        top = counts == best[pair_codes]
        leaders = np.bincount(pair_codes[top], minlength=groups)
        dominant[pair_codes[top]] = pair_owners[top]
    result = {}
    for g in range(groups):
        if not distinct[g]:
            owner, purity = 0, 0.0
        elif leaders[g] == 1:
            owner = int(dominant[g])
            purity = float(np.int64(best[g]) / int(total[g]))
            stats['direct'] += 1
        else:
            # Tied top counts: value_counts on this track's owners, in row order.
            these = pd.Series(owners[owner_codes == g])
            ranked = these.value_counts()
            owner = int(ranked.index[0])
            purity = float(ranked.iloc[0] / len(these))
            stats['tied'] += 1
        result[int(keys[g])] = {
            "track_frames": int(frames[g]),
            "strong_frames": int(strong_frames[g]),
            "positive_owner_count": int(distinct[g]),
            "dominant_owner": owner,
            "dominant_owner_fraction": purity,
        }
    return result


def track_features(original, stats):
    def _track_features(scored):
        try:
            result = _features(scored, stats)
        except Exception:
            result = None
        if result is None:
            stats['fallback'] += 1
            return original(scored)
        stats['tables'] += 1
        return result
    _track_features.__wrapped__ = original
    return _track_features


@contextmanager
def install(module, stats=None):
    """Replace the verified module's _track_features for the duration; restore afterwards."""
    stats = stats if stats is not None else {}
    for key in ('versions', 'tables', 'direct', 'tied', 'fallback'):
        stats.setdefault(key, 0)
    original = getattr(module, '_track_features', None) if module is not None else None
    try:
        supported = (original is not None and getattr(original, '__wrapped__', None) is None
                     and _file_sha256(module.__file__) == SOURCE)
    except (AttributeError, OSError, TypeError):
        supported = False
    if not supported:
        yield stats
        return
    module._track_features = track_features(original, stats)
    stats['versions'] += 1
    try:
        yield stats
    finally:
        module._track_features = original
