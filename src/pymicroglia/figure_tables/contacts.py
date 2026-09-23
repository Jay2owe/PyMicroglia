"""Prepare original contact-network permutations and drawing tables."""
import textwrap
import numpy as np
import pandas as pd
from .. import workbench
from ..measure.modules.coupling import contact_metric_permutation_test
from ..visualisation.labels import describe
from .prepared import PreparedViews
from .rhythm_eligibility import common_period

def _benjamini_hochberg(values):
    return workbench.adjust_pvalues(values,'bh')

DEFAULT_METRICS = [
    "rhythmic",
    "best_period_hours",
    "best_phase_fraction",
    "area_px",
    "corrected_mean",
    "mean_speed",
    "turnover_index",
    "ramification_index",
]

LABELS = {
    "rhythmic": "Rhythm status",
    "best_period_hours": "Detected period",
    "best_phase_fraction": "Peak position",
    "area_px": "Cell area",
    "corrected_mean": "Reporter intensity",
    "mean_speed": "Movement speed",
    "turnover_index": "Footprint turnover",
    "ramification_index": "Ramification",
}

MATRIX_LABELS = {
    "rhythmic": "Rhythm\nstatus",
    "best_period_hours": "Detected\nperiod",
    "best_phase_fraction": "Peak\nposition",
    "area_px": "Cell\narea",
    "corrected_mean": "Reporter\nintensity",
    "mean_speed": "Movement\nspeed",
    "turnover_index": "Footprint\nturnover",
    "ramification_index": "Ramification",
}

RHYTHM_COLUMNS = {"rhythmic", "best_period_hours", "best_phase_fraction"}

RHYTHM_METHOD_LABELS = {
    "lomb": "Lomb-Scargle periodogram",
    "chi_square": "Enright-Sokolove periodogram",
    "f": "F periodogram",
    "jtk": "JTK_CYCLE",
    "ejtk": "empirical JTK_CYCLE",
}

RANDOM_SEED = 24_051_986

ALPHA = 0.05

def _compatible_rhythms(rhythms: pd.DataFrame) -> pd.DataFrame:
    """Give older rhythm tables the primary-test columns used by current runs."""
    fits = rhythms.copy()
    fallbacks = {
        "rhythmic": "rhythmic_lombscargle",
        "best_period_hours": "lombscargle_period_hours",
        "best_phase_hours": "free_cosinor_peak_hour",
        "best_p_value": "lombscargle_false_alarm",
    }
    for current, legacy in fallbacks.items():
        if current not in fits and legacy in fits:
            fits[current] = fits[legacy]
    if "best_phase_fraction" not in fits:
        period = pd.to_numeric(fits.get("best_period_hours"), errors="coerce")
        phase = pd.to_numeric(fits.get("best_phase_hours"), errors="coerce")
        fits["best_phase_fraction"] = workbench.cycle_fraction(phase, period)
    if "primary_rhythm_test" not in fits:
        fits["primary_rhythm_test"] = "lomb"
    if "period_estimation_method" not in fits:
        fits["period_estimation_method"] = "lomb"
    return fits

def _rhythm_label(fits: pd.DataFrame) -> str:
    method = (
        str(fits["primary_rhythm_test"].dropna().mode().iloc[0])
        if "primary_rhythm_test" in fits and fits["primary_rhythm_test"].notna().any()
        else "lomb"
    )
    return RHYTHM_METHOD_LABELS.get(method, method)

def _estimator_label(fits: pd.DataFrame) -> str:
    if "best_method_label" in fits and fits["best_method_label"].notna().any():
        return str(fits["best_method_label"].dropna().mode().iloc[0])
    method = (
        str(fits["period_estimation_method"].dropna().mode().iloc[0])
        if "period_estimation_method" in fits
        and fits["period_estimation_method"].notna().any()
        else "lomb"
    )
    return RHYTHM_METHOD_LABELS.get(method, method)

def _metric_values(
    requested: str,
    summary: pd.DataFrame,
    fits: pd.DataFrame | None,
    interval_minutes: float,
) -> tuple[pd.Series, dict[str, str | float | None]]:
    """Resolve one user metric to per-cell values and explicit comparison rules."""
    if requested in RHYTHM_COLUMNS:
        if fits is None or requested not in fits:
            raise ValueError(
                f"--metrics {requested} needs rhythms.csv with the current or legacy primary-test columns"
            )
        values = fits.set_index("identity")[requested]
        if requested in {"best_period_hours", "best_phase_fraction"}:
            rhythmic = fits.set_index("identity")["rhythmic"].fillna(False).astype(bool)
            values = values.where(rhythmic)
        circular_period = 1.0 if requested == "best_phase_fraction" else None
        units = {
            "rhythmic": "mismatch: 0 same, 1 different",
            "best_period_hours": "h",
            "best_phase_fraction": "fraction of own cycle",
        }[requested]
        return pd.to_numeric(values, errors="coerce"), {
            "metric": requested,
            "source_column": requested,
            "label": LABELS[requested],
            "unit": units,
            "circular_period": circular_period,
            "rhythmic_pairs_only": requested in {"best_period_hours", "best_phase_fraction"},
        }

    candidates = [requested, f"{requested}_median", f"{requested}_mean"]
    resolved = next((column for column in candidates if column in summary), None)
    source = summary
    if resolved is None and fits is not None and requested in fits:
        resolved = requested
        source = fits
    if resolved is None:
        available = sorted(set(summary.columns) | (set() if fits is None else set(fits.columns)))
        preview = ", ".join(available[:24])
        raise ValueError(
            f"--metrics {requested} is unavailable; the first available columns are: {preview}"
        )
    metric = describe(resolved)
    label = LABELS.get(requested, metric.label)
    unit = metric.unit_text(interval_minutes)
    values = source.set_index("identity")[resolved]
    return pd.to_numeric(values, errors="coerce"), {
        "metric": requested,
        "source_column": resolved,
        "label": label,
        "unit": unit,
        "circular_period": 1.0 if requested.endswith("phase_fraction") else None,
        "rhythmic_pairs_only": False,
    }

def _q_label(value: float) -> str:
    if not np.isfinite(value):
        return "q unavailable"
    if value < 0.001:
        return "q < 0.001"
    return f"q = {value:.3g}"

def _forest_labels(summary: pd.DataFrame, q_column: str) -> list[str]:
    return [
        f"{row.metric_label}\n{int(row.pairs)} pairs; {_q_label(float(getattr(row, q_column)))}"
        for row in summary.itertuples()
    ]

def _statistics_rows(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for item in summary.to_dict("records"):
        shared = {
            "metric": item["metric"],
            "metric_label": item["metric_label"],
            "source_column": item["source_column"],
            "unit": item["unit"],
            "cells": item["cells_available"],
            "pairs": item["pairs"],
            "shuffles": item["shuffles"],
            "random_state": item["random_state"],
            "alternative": "two-sided",
            "correction": "Benjamini-Hochberg false-discovery rate within analysis family",
            "alpha": ALPHA,
            "status": item["status"],
        }
        rows.append({
            **shared,
            "analysis_family": "overall contact-pair similarity",
            "test": "cell-label permutation test of mean absolute pair difference",
            "estimate_name": "observed mean difference / shuffled mean difference",
            "estimate": item.get("difference_ratio", np.nan),
            "observed_raw": item.get("observed_mean_difference", np.nan),
            "null_mean": item.get("random_mean_difference", np.nan),
            "null_interval_low": item.get("similarity_null_lo", np.nan),
            "null_interval_high": item.get("similarity_null_hi", np.nan),
            "p_value": item.get("similarity_p_value", np.nan),
            "q_value": item.get("similarity_q_value", np.nan),
        })
        rows.append({
            **shared,
            "analysis_family": "contact duration and pair similarity",
            "test": "cell-label permutation test of Spearman rank correlation",
            "estimate_name": "Spearman rho: contact hours versus pair difference",
            "estimate": item.get("duration_spearman_rho", np.nan),
            "observed_raw": item.get("duration_spearman_rho", np.nan),
            "null_mean": 0.0,
            "null_interval_low": item.get("duration_null_lo", np.nan),
            "null_interval_high": item.get("duration_null_hi", np.nan),
            "p_value": item.get("duration_p_value", np.nan),
            "q_value": item.get("duration_q_value", np.nan),
        })
    return pd.DataFrame(rows)

def prepare(source,options):
    requested_metrics = list(options.get("metrics"))
    if not requested_metrics:
        raise ValueError("--metrics needs at least one cell-level measurement")
    contacts_all = source.table("contacts.csv")
    summary = source.table("cell_summary.csv")
    rhythms = source.table("rhythms.csv", optional=True)
    rhythm_metric = str(options.get("rhythm_metric"))
    fits = None
    if rhythms is not None:
        selected = rhythms[rhythms["metric"] == rhythm_metric].copy()
        if not selected.empty:
            fits = _compatible_rhythms(selected)

    available_radii = sorted(int(value) for value in contacts_all["dilation_px"].unique())
    if not available_radii: raise ValueError("No measured contact radii are available")
    requested_radius = options.get("dilation") if options.get("dilation") is not None else [float(available_radii[0])]
    if np.isscalar(requested_radius): requested_radius = [requested_radius]
    if len(requested_radius) != 1:
        raise ValueError("--dilation accepts one measured radius on the contact-similarity ledger")
    radius = int(requested_radius[0])
    if radius not in available_radii:
        raise ValueError(
            f"--dilation {radius} was not measured; available radii: {available_radii}"
        )
    minimum_hours = float(options.get("min_hours"))
    if minimum_hours < 0:
        raise ValueError("--min-hours cannot be negative")
    contacts = contacts_all[
        contacts_all["dilation_px"].eq(radius)
        & contacts_all["hours_in_contact"].ge(minimum_hours)
    ].copy()
    handoffs = contacts.get("handoff_coincident", pd.Series(False, index=contacts.index))
    excluded_handoffs = int(handoffs.fillna(False).astype(bool).sum())
    contacts = contacts[~handoffs.fillna(False).astype(bool)].copy()
    contacts["identity_a"] = contacts["identity_a"].astype(int)
    contacts["identity_b"] = contacts["identity_b"].astype(int)
    contacts["pair"] = (
        contacts["identity_a"].astype(str) + "-" + contacts["identity_b"].astype(str)
    )
    if contacts.empty:
        raise ValueError(
            f"no non-handoff pairs contacted for at least {minimum_hours:g} h at radius {radius} px"
        )

    shuffles = int(options.get("shuffles"))
    pair_tables: list[pd.DataFrame] = []
    result_rows: list[dict] = []
    metric_labels: dict[str, str] = {}
    matrix_labels: dict[str, str] = {}
    for index, requested in enumerate(requested_metrics):
        values, info = _metric_values(
            requested, summary, fits, float(source.interval),
        )
        if info["circular_period"] is not None:
            if fits is None: raise ValueError("Phase comparisons require saved supported periods")
            common_period(fits.loc[fits.identity.isin(values.dropna().index)])
        pair_data, result = contact_metric_permutation_test(
            contacts, values,
            circular_period=info["circular_period"],
            shuffles=shuffles,
            random_state=RANDOM_SEED + index,
        )
        label = str(info["label"])
        metric_labels[requested] = label
        matrix_labels[requested] = MATRIX_LABELS.get(
            requested, textwrap.fill(label, width=13)
        )
        if not pair_data.empty:
            pair_data.insert(0, "metric", requested)
            pair_data.insert(1, "metric_label", label)
            pair_data.insert(2, "source_column", str(info["source_column"]))
            pair_data.insert(3, "unit", str(info["unit"]))
            pair_data["pair"] = (
                pair_data["identity_a"].astype(int).astype(str)
                + "-" + pair_data["identity_b"].astype(int).astype(str)
            )
            pair_tables.append(pair_data)
        result_rows.append({
            "metric": requested,
            "metric_label": label,
            "source_column": str(info["source_column"]),
            "unit": str(info["unit"]),
            "rhythmic_pairs_only": bool(info["rhythmic_pairs_only"]),
            **result,
        })

    all_pair_metrics = (
        pd.concat(pair_tables, ignore_index=True) if pair_tables else pd.DataFrame()
    )
    tests = pd.DataFrame(result_rows)
    missing_values = pd.Series(np.nan, index=tests.index, dtype=float)
    tests["similarity_q_value"] = _benjamini_hochberg(
        pd.to_numeric(
            tests.get("similarity_p_value", missing_values), errors="coerce",
        ).to_numpy()
    )
    tests["duration_q_value"] = _benjamini_hochberg(
        pd.to_numeric(
            tests.get("duration_p_value", missing_values), errors="coerce",
        ).to_numpy()
    )
    tests["similarity_significant"] = tests["similarity_q_value"] <= ALPHA
    tests["duration_significant"] = tests["duration_q_value"] <= ALPHA
    for column in ('difference_ratio','similarity_null_lo','similarity_null_hi','duration_spearman_rho','duration_null_lo','duration_null_hi'):
        if column not in tests: tests[column] = np.nan
    statistics = _statistics_rows(tests)

    pair_order = contacts.sort_values(
        ["hours_in_contact", "identity_a", "identity_b"],
        ascending=[False, True, True], kind="mergesort",
    )["pair"].tolist()
    maximum_pairs = int(options.get("max_pairs"))
    if maximum_pairs < 0:
        raise ValueError("--max-pairs cannot be negative")
    shown_pairs = pair_order if maximum_pairs == 0 else pair_order[:maximum_pairs]
    figure_data = all_pair_metrics[all_pair_metrics["pair"].isin(shown_pairs)].copy() if not all_pair_metrics.empty else all_pair_metrics.assign(pair=pd.Series(dtype=str),metric=pd.Series(dtype=str),difference_ratio=pd.Series(dtype=float),hours_in_contact=pd.Series(dtype=float))

    pivot=figure_data.pivot(index='pair',columns='metric',values='difference_ratio').reindex(index=shown_pairs,columns=requested_metrics)
    durations=contacts.set_index('pair').hours_in_contact
    ledger=dict(table=figure_data,matrix=pivot.to_numpy(float),rows=[f'{pair} | {durations[pair]:g} h' for pair in shown_pairs],columns=[matrix_labels[m] for m in requested_metrics])
    forests={}
    for key,prefix,estimate,baseline in [('overall','similarity','difference_ratio',1.),('duration','duration','duration_spearman_rho',0.)]:
        forests[key]=dict(table=tests,labels=_forest_labels(tests,prefix+'_q_value'),estimate=estimate,low=prefix+'_null_lo',high=prefix+'_null_hi',significant=prefix+'_significant',baseline=baseline)
    return PreparedViews({'pairs':ledger,**forests},auxiliary={'all_pair_metrics.csv':all_pair_metrics,'metric_tests.csv':tests,'statistics.csv':statistics,'contact_pairs.csv':contacts},wording=dict(title='Contact-pair measurement similarity',subtitle=f'{len(contacts)} pairs at radius {radius:g} px for at least {minimum_hours:g} h; {len(shown_pairs)} longest pairs shown.',footnote=f'Measurements shuffled {shuffles:,} times over the unchanged network. Whiskers are central 95% shuffle intervals, not confidence intervals. Benjamini-Hochberg correction within each test family; {excluded_handoffs} handoff-coincident pairs excluded.')),None
