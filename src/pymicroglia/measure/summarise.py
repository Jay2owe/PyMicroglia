"""Roll-ups.

Three levels, and every figure downstream is drawn from one of them:

* ``cell_frame`` - one row per cell per frame, the raw material;
* ``cell_summary`` - one row per cell, for comparing cells;
* ``frame_summary`` - one row per frame, for the population over time.

Nothing here re-measures anything. If a number is wrong the fault is in a
measurement module, not in this file. Ported from Motion's
``analysis/summarise.py`` on 2026-09-21 with the arithmetic unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .context import MeasurementContext
from .declare import Output, declared_tables

__all__ = ["ROLLUPS", "SUMMARY_METRICS", "side_column_names", "join_side",
           "build_cell_summary", "build_frame_summary", "build_movie_summary"]

#: The three tables this file builds, declared the same way a module declares
#: its own. A module's table folds into whichever roll-up shares its grain, so
#: the roll-ups have to say what their grain is.
ROLLUPS: tuple[Output, ...] = (
    Output("cell_frame", grain=("identity", "frame_index")),
    Output("cell_summary", grain=("identity",)),
    Output("frame_summary", grain=("frame_index",)),
)

# Columns worth a per-cell median and spread. Anything absent is skipped, so a
# run with modules switched off still produces a valid summary.
SUMMARY_METRICS = [
    "area_px",
    "perimeter_px",
    "circularity",
    "solidity",
    "ramification_index",
    "aspect_ratio",
    "skeleton_px",
    "skeleton_endpoints",
    "skeleton_branches",
    "signal_mean",
    "corrected_mean",
    "integrated_density",
    "corrected_integrated_density",
    "signal_to_background",
    "punctate_fraction",
    "punctateness",
    "signal_cv",
    "dff",
    "background_median",
    "turnover_index",
    "jaccard",
    "extension_bias",
    "step_px_gapless",
    "soma_step_px_gapless",
    "evidence_motion_fraction",
    "inferred_fraction",
    # Four of the 28 Sholl summaries, chosen by what they add rather than by how
    # stable they look; the rest are cell size again.
    "sholl_regression_coefficient_scale_global",
    "sholl_occupancy_outer_scale_cell",
    "sholl_occupancy_ratio_scale_cell",
    "sholl_max_intersections_scale_global",
    # Neighbour geometry. Both distances, because they are not substitutes.
    "nearest_neighbour_px",
    "nearest_edge_px",
    "local_density",
    "domain_occupancy",
    # Branch geometry.
    "longest_branch_px",
    "branch_count",
    # How recently anything happened where a cell is standing.
    "trail_age_frames_mean",
]


def side_column_names(context: MeasurementContext) -> set[str]:
    """Every column the user's own spreadsheets contribute, under their prefixes.

    The keys are excluded because they are not the user's numbers: the loader
    renamed the user's key column to the key it joins on.
    """
    names: set[str] = set()
    for table in context.side.values():
        names.update(c for c in table.columns if c not in ("identity", "frame_index"))
    return names


def join_side(table: pd.DataFrame, context: MeasurementContext, key: str) -> pd.DataFrame:
    """Attach every side table keyed on ``key`` to ``table``.

    A left join, always. A frame the user's log does not mention, or a cell
    their spreadsheet has no row for, keeps its row and gets a blank.
    """
    for name in sorted(context.side):
        side = context.side[name]
        if side.empty or key not in side.columns:
            continue
        extra = [column for column in side.columns
                 if column != key and column not in table.columns]
        if not extra:
            continue
        table = table.merge(side[[key, *extra]], on=key, how="left")
    return table


def _spread(series: pd.Series) -> float:
    valid = series.dropna()
    if valid.empty:
        return np.nan
    return float(np.percentile(valid, 75) - np.percentile(valid, 25))


#: Every metric is summarised four ways, not one. Which statistic is used is a
#: judgement, and a table that offers only the median makes that judgement for
#: the reader invisibly. Median pairs with the inter-quartile range and mean
#: with the standard deviation.
_STATISTICS = {
    "median": lambda grouped, metric: grouped[metric].median(),
    "iqr": lambda grouped, metric: grouped[metric].apply(_spread),
    "mean": lambda grouped, metric: grouped[metric].mean(),
    "sd": lambda grouped, metric: grouped[metric].std(),
}


def build_cell_summary(
    cell_frame: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    context: MeasurementContext,
) -> pd.DataFrame:
    metrics = [column for column in SUMMARY_METRICS if column in cell_frame.columns]
    grouped = cell_frame.groupby("identity", sort=True)

    summary = pd.DataFrame({"identity": sorted(cell_frame["identity"].unique())}).set_index("identity")
    summary["observed_frames"] = grouped["frame_index"].count()
    summary["first_frame_index"] = grouped["frame_index"].min()
    summary["last_frame_index"] = grouped["frame_index"].max()
    summary["span_frames"] = summary["last_frame_index"] - summary["first_frame_index"] + 1
    summary["gap_frames"] = summary["span_frames"] - summary["observed_frames"]
    summary["coverage"] = summary["observed_frames"] / context.n_frames
    summary["first_hour"] = context.scale.hours(summary["first_frame_index"] + context.source_frame_offset)
    summary["last_hour"] = context.scale.hours(summary["last_frame_index"] + context.source_frame_offset)

    if "touches_border" in cell_frame.columns:
        summary["border_frames"] = grouped["touches_border"].sum().astype(int)
        summary["ever_touches_border"] = summary["border_frames"] > 0

    for metric in metrics:
        for statistic, reduce in _STATISTICS.items():
            summary[f"{metric}_{statistic}"] = reduce(grouped, metric)

    summary = summary.reset_index()

    # Any module that declared a per-cell table as a fold has its columns
    # merged in here rather than written to a file of its own. A column the
    # summary already has is dropped rather than merged.
    for name, output in sorted(declared_tables().items()):
        if not output.fold or output.grain != ("identity",):
            continue
        folded = tables.get(name)
        if folded is None or folded.empty or "identity" not in folded.columns:
            continue
        columns = [c for c in folded.columns
                   if c == "identity" or c not in summary.columns]
        summary = summary.merge(folded[columns], on="identity", how="left")

    return join_side(summary, context, "identity")


def build_frame_summary(
    cell_frame: pd.DataFrame,
    context: MeasurementContext,
    tables: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    frames = context.frame_table()
    labels = context.labels

    assigned = np.array([int((labels[i] > 0).sum()) for i in range(context.n_frames)])
    field_px = int(labels.shape[1] * labels.shape[2])
    frames["valid_px"] = [context.valid_px(i) for i in range(context.n_frames)]
    frames["valid_share"] = frames["valid_px"] / float(field_px)
    frames["identities_present"] = [
        int(len([v for v in np.unique(labels[i]) if v])) for i in range(context.n_frames)
    ]
    frames["assigned_px"] = assigned
    if context.unclaimed is not None:
        frames["unclaimed_px"] = [
            int((context.unclaimed[i] > 0).sum()) for i in range(context.n_frames)
        ]
        frames["unclaimed_fraction"] = frames["unclaimed_px"] / (
            frames["unclaimed_px"] + frames["assigned_px"]
        ).replace(0, np.nan)

    metrics = [column for column in SUMMARY_METRICS if column in cell_frame.columns]
    if metrics:
        grouped = cell_frame.groupby("frame_index")[metrics]
        aggregate = pd.concat(
            [grouped.median().add_suffix("_median"),
             grouped.mean().add_suffix("_mean")], axis=1)
        frames = frames.merge(aggregate.reset_index(), on="frame_index", how="left")

    for name, output in sorted(declared_tables().items()):
        if not output.fold or output.grain != ("frame_index",):
            continue
        folded = (tables or {}).get(name)
        if folded is None or folded.empty or "frame_index" not in folded.columns:
            continue
        columns = [c for c in folded.columns
                   if c == "frame_index" or c not in frames.columns]
        frames = frames.merge(folded[columns], on="frame_index", how="left")

    return join_side(frames, context, "frame_index")


def build_movie_summary(
    cell_frame: pd.DataFrame,
    cell_summary: pd.DataFrame,
    frame_summary: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    context: MeasurementContext,
) -> dict:
    scale = context.scale
    summary: dict = {
        "stem": context.stem,
        "frames": context.n_frames,
        "hours_covered": round(scale.hours(context.n_frames - 1), 3),
        "minutes_per_frame": scale.minutes_per_frame,
        "identities": len(context.identities),
        "cell_frame_rows": int(len(cell_frame)),
        "scale": scale.describe(),
        "coverage": {
            "median_identities_per_frame": float(frame_summary["identities_present"].median()),
            "cells_present_every_frame": int(
                (cell_summary["observed_frames"] == context.n_frames).sum()
            ) if "observed_frames" in cell_summary else 0,
            "total_gap_frames": int(cell_summary["gap_frames"].sum())
            if "gap_frames" in cell_summary else 0,
        },
    }

    if "area_px_median" in cell_summary:
        summary["morphology"] = {
            "median_cell_area_px": float(cell_summary["area_px_median"].median()),
            "median_solidity": float(cell_summary["solidity_median"].median())
            if "solidity_median" in cell_summary
            else None,
            "median_skeleton_px": float(cell_summary["skeleton_px_median"].median())
            if "skeleton_px_median" in cell_summary
            else None,
        }

    if "corrected_mean_median" in cell_summary:
        zero_background_fraction = (
            float(cell_frame["background_is_zero"].mean())
            if "background_is_zero" in cell_frame
            else None
        )
        background_is_zero = (
            zero_background_fraction is not None and zero_background_fraction > 0.9
        )
        summary["intensity"] = {
            "median_corrected_mean": float(cell_summary["corrected_mean_median"].median()),
            "input_already_background_subtracted": background_is_zero,
            "zero_background_fraction": zero_background_fraction,
            "median_punctateness_p90_over_median": float(
                cell_summary["punctateness_median"].median()
            )
            if "punctateness_median" in cell_summary
            else None,
            "median_signal_to_background": None
            if background_is_zero
            else float(cell_summary["signal_to_background_median"].median()),
        }

    if "turnover_index" in cell_frame:
        summary["surveillance"] = {
            "median_turnover_index": float(cell_frame["turnover_index"].median()),
            "median_jaccard": float(cell_frame["jaccard"].median())
            if "jaccard" in cell_frame
            else None,
        }

    provenance = tables.get("provenance")
    if provenance is not None and not provenance.empty:
        inferred_px = int(provenance["inferred_px"].sum())
        footprint = inferred_px + int(provenance["observed_px"].sum())
        per_frame = tables.get("provenance_frame")
        summary["provenance"] = {
            "labelled_px": footprint,
            "inferred_px": inferred_px,
            "inferred_fraction_of_footprint": (
                inferred_px / footprint if footprint else None
            ),
            "renamed_px": int(provenance["renamed_px"].sum())
            if "renamed_px" in provenance
            else None,
            "added_px": int(provenance["added_px"].sum())
            if "added_px" in provenance
            else None,
            "added_fraction_of_footprint": (
                int(provenance["added_px"].sum()) / footprint
                if footprint and "added_px" in provenance
                else None
            ),
            "cell_frames": int(len(provenance)),
            "cell_frames_touched": int((provenance["inferred_px"] > 0).sum()),
            "cell_frames_with_added_px": int((provenance["added_px"] > 0).sum())
            if "added_px" in provenance
            else None,
            "cell_frames_entirely_supplied": int(
                ((provenance["added_px"] > 0)
                 & (provenance["observed_px"] == 0)
                 & (provenance["renamed_px"] == 0)).sum()
            )
            if "added_px" in provenance
            else None,
            "identities": int(provenance["identity"].nunique()),
            "identities_touched": int(
                provenance.loc[provenance["inferred_px"] > 0, "identity"].nunique()
            ),
            "unresolved_px_outside_any_cell": (
                int(per_frame["unresolved_px_unowned"].sum())
                if per_frame is not None and "unresolved_px_unowned" in per_frame
                else None
            ),
        }

    tracks = tables.get("motility_tracks")
    if tracks is not None and not tracks.empty:
        summary["motility"] = {
            "tracks": int(len(tracks)),
            "median_step_px": float(tracks["median_step_px"].median()),
            "median_straightness": float(tracks["straightness"].median()),
            "median_msd_alpha": float(tracks["msd_alpha"].median()),
            "median_speed_" + scale.speed_unit: float(tracks["mean_speed"].median()),
        }

    rhythms = tables.get("rhythms")
    if rhythms is not None and not rhythms.empty:
        per_metric = {}
        for metric, group in rhythms.groupby("metric"):
            per_metric[metric] = {
                "cells_tested": int(len(group)),
                "primary_rhythm_test": str(group["primary_rhythm_test"].iloc[0]),
                "period_estimation_method": str(
                    group.get("period_estimation_method", group["primary_rhythm_test"]).iloc[0]
                ),
                "rhythmic_by_primary_test": int(
                    group["rhythmic"].fillna(False).astype(bool).sum()),
                "rhythmic_by_cosinor": (
                    int(group["rhythmic_cosinor"].sum())
                    if group["rhythmic_cosinor"].notna().any() else None
                ),
                "rhythmic_by_both_tests": (
                    int(group["rhythmic_both"].sum())
                    if group["rhythmic_both"].notna().any() else None
                ),
                "median_best_period_hours": float(group["best_period_hours"].median())
                if "best_period_hours" in group
                else None,
                "median_free_period_hours": float(group["best_period_hours"].median())
                if "best_period_hours" in group else None,
                "median_peak_hour": float(
                    (group["best_phase_hours"] if "best_phase_hours" in group
                     else group["cosinor_peak_hour"]).median()
                ),
            }
        population = tables.get("rhythms_population")
        if population is not None and not population.empty:
            for _, row in population[population["population"] == "all_tested"].iterrows():
                if row["metric"] in per_metric:
                    per_metric[row["metric"]].update(
                        {
                            "phase_vector_length": float(row["vector_length"]),
                            "phase_rayleigh_p": float(row["rayleigh_p_value"]),
                            "population_mean_peak_hour": float(row["mean_peak_hour"]),
                        }
                    )

        null = tables.get("rhythms_null")
        if null is not None and not null.empty:
            for _, row in null.iterrows():
                if row["metric"] in per_metric:
                    per_metric[row["metric"]].update(
                        {
                            "null_false_positive_rate_primary": float(
                                row["false_positive_rate_primary"]),
                            "excess_over_null_primary": float(
                                row["excess_over_null_primary"]),
                            "null_false_positive_rate_both": float(row["false_positive_rate_both"]),
                            "excess_over_null": float(row["excess_over_null_both"]),
                        }
                    )
        summary["rhythms"] = {
            "cycles_covered": float(rhythms["cycles_covered"].max()),
            "period_underdetermined": bool(rhythms["period_underdetermined"].all()),
            "null_model": str(null["null_model"].iloc[0]) if null is not None and not null.empty else None,
            "per_metric": per_metric,
        }

    return summary
