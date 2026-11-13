"""Conservative complete-movie lineage union from independent track evidence.

The producer is field-wide and target-free. Feedback identities and locations are
not inputs. The accepted parent remains unchanged unless a pair passes every
predeclared evidence gate, and ambiguous pair graphs are refused.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from resident_takeover import attach_owners
import recurrent_cyclic_alias

@dataclass(frozen=True)
class Params:
    minimum_each_direction_mask_handoffs: int = 1
    minimum_each_direction_track_handoffs: int = 2
    minimum_bidirectional_shared_tracks: int = 4
    minimum_mean_shared_track_pair_purity: float = 0.90
    minimum_union_coverage: float = 1.0
    maximum_cooccurrence_movie_fraction: float = 0.65

_LAX = {
    "minimum_overlap_px": 1,
    "minimum_overlap_fraction": 0.0,
    "minimum_area_similarity": 0.0,
    "minimum_shape_correlation": 0.0,
    "maximum_step_sum_radii": 100.0,
    "minimum_cross_handoffs": 1,
    "minimum_each_direction_handoffs": 0,
    "minimum_onset_lead_movie_fraction": 0.0,
    "minimum_union_coverage": 0.0,
    "maximum_cooccurrence_movie_fraction": 1.0,
    "minimum_terminal_alias_movie_fraction": 0.0,
    "maximum_track_handoff_gap_movie_fraction": 0.20,
    "minimum_track_handoffs": 0,
    "minimum_each_direction_track_handoffs": 0,
}

def _track_evidence(attached: pd.DataFrame, first: int, second: int) -> dict:
    first_rows = attached[attached.candidate_owner.astype(int).eq(first)]
    second_rows = attached[attached.candidate_owner.astype(int).eq(second)]
    shared = set(first_rows.track_id.astype(int)) & set(second_rows.track_id.astype(int))
    purities: list[float] = []
    bidirectional = 0
    for _, group in attached[attached.track_id.astype(int).isin(shared)].groupby("track_id"):
        owners = group.candidate_owner.astype(int)
        owner_set = set(owners)
        if first in owner_set and second in owner_set:
            bidirectional += 1
        purities.append(float(owners.isin((first, second)).mean()))
    return {
        "shared_tracks": len(shared),
        "bidirectional_shared_tracks": bidirectional,
        "mean_shared_track_pair_purity": float(np.mean(purities)) if purities else 0.0,
        "minimum_shared_track_pair_purity": float(np.min(purities)) if purities else 0.0,
    }

def discover(labels: np.ndarray, points: pd.DataFrame, params: Params = Params()):
    proposals, mask_handoffs, track_handoffs, _ = recurrent_cyclic_alias.discover(
        labels, points, _LAX)
    attached = attach_owners(points, labels)
    attached = attached[
        attached.physically_visible.astype(bool)
        & attached.candidate_owner.astype(int).gt(0)
    ]
    rows: list[dict] = []
    for proposal in proposals.itertuples(index=False):
        if pd.isna(proposal.canonical_identity) or pd.isna(proposal.alias_identity):
            continue
        canonical = int(proposal.canonical_identity)
        alias = int(proposal.alias_identity)
        extra = _track_evidence(attached, canonical, alias)
        mask_bidir = min(int(proposal.canonical_to_alias_handoffs),
                         int(proposal.alias_to_canonical_handoffs))
        track_bidir = min(int(proposal.canonical_to_alias_track_handoffs),
                          int(proposal.alias_to_canonical_track_handoffs))
        failures = []
        if mask_bidir < params.minimum_each_direction_mask_handoffs:
            failures.append("insufficient_bidirectional_mask_handoffs")
        if track_bidir < params.minimum_each_direction_track_handoffs:
            failures.append("insufficient_bidirectional_track_handoffs")
        if extra["bidirectional_shared_tracks"] < params.minimum_bidirectional_shared_tracks:
            failures.append("insufficient_bidirectional_shared_tracks")
        if extra["mean_shared_track_pair_purity"] < params.minimum_mean_shared_track_pair_purity:
            failures.append("shared_track_pair_purity_too_low")
        if float(proposal.union_coverage) < params.minimum_union_coverage:
            failures.append("temporal_union_incomplete")
        if float(proposal.cooccurrence_fraction) > params.maximum_cooccurrence_movie_fraction:
            failures.append("cooccurrence_too_high")
        rows.append({
            "canonical_identity": canonical,
            "alias_identity": alias,
            "mask_handoffs": int(proposal.mask_handoffs),
            "each_direction_mask_handoffs": mask_bidir,
            "track_handoffs": int(proposal.track_handoffs),
            "each_direction_track_handoffs": track_bidir,
            **extra,
            "union_coverage": float(proposal.union_coverage),
            "cooccurrence_fraction": float(proposal.cooccurrence_fraction),
            "eligible": not failures,
            "reason": "eligible_complete_lineage_union" if not failures else "|".join(failures),
        })
    audit = pd.DataFrame(rows)
    if len(audit):
        eligible = audit[audit.eligible.astype(bool)]
        counts: dict[int, int] = {}
        for row in eligible.itertuples(index=False):
            for identity in (int(row.canonical_identity), int(row.alias_identity)):
                counts[identity] = counts.get(identity, 0) + 1
        ambiguous = [
            bool(row.eligible and (
                counts.get(int(row.canonical_identity), 0) > 1
                or counts.get(int(row.alias_identity), 0) > 1))
            for row in audit.itertuples(index=False)
        ]
        audit["ambiguous_pair_graph"] = ambiguous
        audit.loc[audit.ambiguous_pair_graph, "eligible"] = False
        audit.loc[audit.ambiguous_pair_graph, "reason"] = "ambiguous_pair_graph"
    else:
        audit["ambiguous_pair_graph"] = pd.Series(dtype=bool)
    return audit, mask_handoffs, track_handoffs

def correct(labels: np.ndarray, points: pd.DataFrame, params: Params = Params()):
    audit, mask_handoffs, track_handoffs = discover(labels, points, params)
    candidate = np.asarray(labels).copy()
    applications = []
    eligible = audit[audit.eligible.astype(bool)] if len(audit) else audit
    for row in eligible.itertuples(index=False):
        canonical, alias = int(row.canonical_identity), int(row.alias_identity)
        mask = candidate == alias
        candidate[mask] = canonical
        applications.append({
            "canonical_identity": canonical,
            "alias_identity": alias,
            "changed_pixels": int(mask.sum()),
            "changed_frames": int(np.any(mask, axis=(1, 2)).sum()),
        })
    changed = candidate != labels
    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("lineage union changed foreground")
    summary = {
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "tracks", "frames", "coordinates", "events",
            "regions", "review_cases")},
        "pairs_audited": int(len(audit)),
        "eligible_pairs": int(len(eligible)),
        "applied_pairs": int(len(applications)),
        "changed_pixels": int(changed.sum()),
        "changed_frames": int(np.any(changed, axis=(1, 2)).sum()),
        "foreground_changed_pixels": int(np.count_nonzero(
            (candidate > 0) != (labels > 0))),
        "parameters": asdict(params),
    }
    return candidate, audit, mask_handoffs, track_handoffs, pd.DataFrame(applications), summary
