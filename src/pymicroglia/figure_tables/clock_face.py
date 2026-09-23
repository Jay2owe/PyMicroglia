"""Prepare saved detected-period evidence; no fresh fitting occurs."""

import numpy as np
import pandas as pd

METHOD_LABELS = {
    "lomb": "Lomb-Scargle periodogram",
    "chi_square": "Enright-Sokolove periodogram",
    "f": "F periodogram",
    "jtk": "JTK_CYCLE",
    "ejtk": "empirical JTK_CYCLE",
}

def _fallback_rows(fits: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Recover the primary Lomb-Scargle result from runs made before method rows."""
    source = fits.copy()
    period = source.get("best_period_hours", source.get("lombscargle_period_hours"))
    phase = source.get("best_phase_hours", source.get("free_cosinor_peak_hour"))
    p_value = source.get("best_p_value", source.get("lombscargle_false_alarm"))
    rhythmic = source.get("rhythmic", source.get("rhythmic_lombscargle", False))
    amplitude = source.get("free_cosinor_amplitude", source.get("cosinor_amplitude"))
    if period is None or phase is None or amplitude is None:
        raise SystemExit(
            "rhythms.csv has no detected-period phase and amplitude for the clock face"
        )

    period = pd.to_numeric(period, errors="coerce")
    phase = pd.to_numeric(phase, errors="coerce")
    rhythmic = pd.Series(rhythmic, index=source.index).fillna(False).astype(bool)
    estimator = str(params.get(
        "period_estimation_method", params.get("primary_rhythm_test", "lomb")))
    significance = str(params.get("primary_rhythm_test", "lomb"))
    default_label = METHOD_LABELS.get(estimator, estimator)
    method_label = source.get("best_method_label", pd.Series(default_label, index=source.index))
    alpha = source.get("best_alpha", pd.Series(float(params.get("alpha", 0.05)), index=source.index))
    status = source.get(
        "rhythm_status",
        pd.Series(np.where(rhythmic, "rhythmic", "arrhythmic"), index=source.index),
    )
    return pd.DataFrame({
        "identity": source["identity"],
        "metric": source["metric"],
        "method": estimator,
        "method_label": method_label,
        "significance_method": significance,
        "significance_method_label": METHOD_LABELS.get(significance, significance),
        "rhythm_status": status,
        "rhythmic": rhythmic,
        "period_hours": period,
        "phase_hours": phase,
        "phase_fraction": np.mod(phase, period) / period,
        "amplitude": pd.to_numeric(amplitude, errors="coerce"),
        "p_value": pd.to_numeric(p_value, errors="coerce") if p_value is not None else np.nan,
        "alpha": pd.to_numeric(alpha, errors="coerce"),
    })

def _clock_rows(
    fits: pd.DataFrame,
    methods: pd.DataFrame | None,
    metric: str,
    params: dict,
) -> pd.DataFrame:
    """One explicit primary-test row per cell, with phase in hours and cycles."""
    if methods is None:
        return _fallback_rows(fits, params)
    estimator_flag = methods.get(
        "period_estimator", pd.Series(False, index=methods.index)).fillna(False)
    estimated = methods[(methods["metric"] == metric) & estimator_flag].copy()
    primary = methods[(methods["metric"] == metric) & methods["primary"].fillna(False)].copy()
    if estimated.empty or primary.empty:
        return _fallback_rows(fits, params)
    verdict = primary[[
        "identity", "metric", "method", "method_label", "method_rhythm_status",
        "method_rhythmic", "p_value", "alpha",
    ]].rename(columns={
        "method": "significance_method",
        "method_label": "significance_method_label",
        "method_rhythm_status": "rhythm_status",
        "method_rhythmic": "rhythmic",
        "p_value": "significance_p_value",
        "alpha": "significance_alpha",
    })
    selected = estimated[[
        "identity", "metric", "method", "method_label", "period_hours",
        "phase_hours", "amplitude",
    ]].merge(verdict, on=["identity", "metric"], how="inner", validate="one_to_one")
    period = pd.to_numeric(selected["period_hours"], errors="coerce")
    phase = pd.to_numeric(selected["phase_hours"], errors="coerce")
    selected["phase_fraction"] = np.mod(phase, period) / period
    selected["p_value"] = selected.pop("significance_p_value")
    selected["alpha"] = selected.pop("significance_alpha")
    return selected

def prepare(source, options):
    metric = options["metrics"]
    rhythms = source.table("rhythms")
    methods = source.table("rhythm_methods", optional=True)
    fits = rhythms.loc[rhythms.metric.eq(metric)].copy()
    if fits.empty:
        raise ValueError(f"No saved rhythm result for {metric}")
    data = _clock_rows(fits, methods, metric, source.module_params("rhythms"))
    for column in ("period_hours", "phase_hours", "phase_fraction", "amplitude", "p_value", "alpha"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["rhythmic"] = data.rhythmic.fillna(False).astype(bool)
    data["included_in_plot"] = data.rhythmic & np.isfinite(data.phase_fraction) & np.isfinite(data.amplitude)
    selected = data.loc[data.included_in_plot]
    bins = options["bins"]
    if isinstance(bins,bool) or not isinstance(bins,int) or bins < 1:
        raise ValueError("bins must be a positive integer")
    counts, edges = np.histogram(selected.phase_fraction, bins=np.linspace(0,1,bins+1))
    rose = pd.DataFrame({"bin_left":edges[:-1],"bin_right":edges[1:],"count":counts})
    counts, edges = np.histogram(selected.amplitude, bins=bins)
    histogram = pd.DataFrame({"bin_left":edges[:-1],"bin_right":edges[1:],"count":counts})
    return {"dial":data,"rose":rose,"histogram":histogram}, data
