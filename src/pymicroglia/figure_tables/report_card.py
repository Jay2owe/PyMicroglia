"""Prepare cell images, traces and explicitly requested Workbench model curves."""
import numpy as np
import pandas as pd
import tifffile
from .. import workbench
from ..measure.modules.rhythms import DEFAULTS as RHYTHM_DEFAULTS
from ..visualisation.labels import describe
from . import images as intensity
from .prepared import PreparedViews

def commas(value):
    return [part.strip() for part in str(value).split(',') if part.strip()]

def prepare(source,options):
    identity = options.get("identity")
    hour_ticks = options.get("hour_ticks")
    metrics = options.get("metrics")
    n_tiles = int(options.get("images"))
    if n_tiles < 0: raise ValueError("images cannot be negative")
    tile_hours = options.get("image_hours")
    cell_lut = options.get("cell_lut")
    image_filter = options.get("image_filter")
    display_black = options.get("display_black_percentile")
    display_white = options.get("display_white_percentile")
    display_range_scope = str(options.get("display_range_scope")).lower()
    display_gamma = options.get("display_gamma")
    display_gain = options.get("display_gain")
    display_spatial_sigma = options.get("display_spatial_sigma")
    display_pool_px = options.get("display_pool_px")
    display_sharpness = options.get("display_sharpness")
    display_noise_multiple = options.get("display_noise_multiple")
    display_pad_frames = options.get("display_pad_frames")
    trace_luts = options.get("trace_luts")
    trace_layout = str(options.get("trace_layout")).lower()
    trace_view = str(options.get("trace_view")).lower()
    outline = options.get("outline")
    fit = options.get("fit")
    inherited = {**RHYTHM_DEFAULTS, **source.module_params("rhythms")}
    resolved = workbench.resolve_analysis_options(inherited, options.get)
    rhythm_params = resolved["params"]
    detrending = {
        name: resolved[name] for name in workbench.DETREND_DEFAULTS
    }
    detrend = str(detrending["detrend"])
    fit_method = resolved["method"]
    significance_method = resolved["significance_method"]

    if trace_layout not in ("stack", "overlay"):
        raise ValueError("--trace-layout must be stack or overlay")
    if trace_view not in ("raw", "detrended"):
        raise ValueError("--trace-view must be raw or detrended")
    if detrend not in workbench.DETREND_METHODS:
        raise ValueError(
            f"--detrend {detrend!r} is unknown; choose "
            + ", ".join(workbench.DETREND_METHODS))
    if display_range_scope not in ("stack", "displayed"):
        raise ValueError("--display-range-scope must be stack or displayed")
    if fit_method not in workbench.PERIOD_METHODS:
        raise ValueError(
            f"--fit-method {fit_method!r} is unknown; choose "
            + ", ".join(workbench.PERIOD_METHODS))

    if not metrics:
        raise ValueError(
            "--metrics needs at least one column; pass --images 0 to drop the tiles")

    cell_frame = source.table("cell_frame.csv")
    rhythms = source.table("rhythms.csv")

    unknown = [column for column in metrics if column not in cell_frame.columns]
    if unknown:
        traceable = ", ".join(
            c for c in cell_frame.columns
            if c not in ("stem", "condition", "subject", "identity")
            and cell_frame[c].dtype.kind in "fi"
        )
        raise ValueError(
            f"--metrics names {', '.join(unknown)}, which cell_frame.csv does not "
            f"have.\nColumns available: {traceable}"
        )

    cell = cell_frame[
        cell_frame["identity"] == identity].sort_values("frame_index").copy()
    if cell.empty:
        present = sorted(cell_frame["identity"].unique())
        raise ValueError(
            f"identity {identity} is not in this run; it holds {len(present)} "
            f"identities, {present[0]} to {present[-1]}"
        )
    stored_fits = rhythms[rhythms["identity"] == identity].copy()

    # Which traces carry the model curve. The period is re-estimated through
    # Circadian Workbench at the selected detrending, so this control is not
    # tied to whichever method produced the run's summary table.
    if not str(fit).strip():
        fitted_metrics = []
    elif fit.strip().lower() == "all":
        fitted_metrics = list(metrics)
    else:
        fitted_metrics = commas(fit)
        missing_fit = [column for column in fitted_metrics if column not in metrics]
        if missing_fit:
            raise ValueError(
                "--fit can only name traces selected by --metrics; missing: "
                + ", ".join(missing_fit))

    hours = cell["hours"].to_numpy(float)

    detrended_by_metric: dict[str, np.ndarray] = {}
    baseline_by_metric: dict[str, np.ndarray] = {}
    fit_evidence: dict[str, dict] = {}
    for column in metrics:
        raw_values = cell[column].to_numpy(float)
        usable = np.isfinite(hours) & np.isfinite(raw_values)
        detrended_values = np.full(raw_values.shape, np.nan, dtype=float)
        baseline_values = np.full(raw_values.shape, np.nan, dtype=float)
        if usable.sum() >= 2:
            result = workbench.detrend_trace(
                hours[usable], raw_values[usable], rhythm_params, method=detrend)
            detrended_values[usable] = np.asarray(result["values"], dtype=float)
            baseline_values[usable] = np.asarray(result["baseline"], dtype=float)
        detrended_by_metric[column] = detrended_values
        baseline_by_metric[column] = baseline_values
        if column in fitted_metrics and usable.sum() >= resolved["min_observations"]:
            try:
                fit_evidence[column] = workbench.estimate_one(
                    hours[usable], raw_values[usable], rhythm_params,
                    fit_method, detrend=detrend,
                )
                result = fit_evidence[column]
                evidence = result if significance_method == fit_method else workbench.estimate_one(
                    hours[usable], raw_values[usable], rhythm_params,
                    significance_method, detrend=detrend)
                result.update(
                    significance_method=significance_method,
                    significance_p_value=evidence.get("p_value"),
                    significance_period_hours=evidence.get("period_hours"),
                    significance_status=evidence.get("status", "failed"))
                period = result.get("period_hours")
                span = float(hours[usable].max() - hours[usable].min())
                cycles = (
                    span / float(period)
                    if period is not None and np.isfinite(period) and float(period) > 0
                    else np.nan
                )
                result.update(
                    cycles_observed=cycles,
                    period_underdetermined=(
                        not np.isfinite(cycles) or cycles < resolved["min_cycles"]
                    ),
                )
            except ValueError as error:
                fit_evidence[column] = {
                    "method": fit_method,
                    "method_label": workbench.PERIOD_METHODS[fit_method]["label"],
                    "status": "failed",
                    "rhythm_status": "unknown",
                    "period_hours": np.nan,
                    "p_value": np.nan,
                    "message": str(error),
                    "workbench_version": workbench.WORKBENCH_VERSION,
                }
        elif column in fitted_metrics:
            fit_evidence[column] = {
                "method": fit_method,
                "method_label": workbench.PERIOD_METHODS[fit_method]["label"],
                "status": "not_tested",
                "rhythm_status": "not tested",
                "period_hours": np.nan,
                "p_value": np.nan,
                "message": "too_few_observations",
                "workbench_version": workbench.WORKBENCH_VERSION,
            }

    fitted_columns = list(fit_evidence)
    adjusted = workbench.adjust_pvalues(
        [fit_evidence[column].get("significance_p_value", np.nan)
         for column in fitted_columns],
        resolved["multiple_testing"],
    )
    for column, q_value in zip(fitted_columns, adjusted):
        result = fit_evidence[column]
        result.update(
            q_value=q_value,
            correction=resolved["multiple_testing"],
            alpha=resolved["rhythmic_alpha"],
        )
        tested = result.get("significance_status") == "ok" and np.isfinite(q_value)
        result["rhythm_status"] = (
            "rhythmic" if tested and q_value < resolved["rhythmic_alpha"]
            else "not rhythmic" if tested else "not tested"
        )

    # Label frame k is source frame k + offset; the tables carry both, so the offset
    # is read off them rather than repeated here.
    offsets = cell["source_imagej_frame"]-cell["imagej_frame"]
    if offsets.nunique()!=1: raise ValueError("Raw and label frame offsets must be constant")
    offset = int(offsets.iloc[0])
    frames = cell["frame_index"].to_numpy(int)

    tile_frames: list[int] = []
    if 'tiles' in getattr(source,'requested_views',('tiles','traces')) and (n_tiles > 0 or tile_hours):
        if tile_hours:
            # Nearest observed frame to each requested hour: an hour the cell was
            # missing for still gets a tile, and the tile says which hour it is.
            tile_frames = [int(frames[int(np.abs(hours - wanted).argmin())])
                           for wanted in tile_hours]
        else:
            picks = np.linspace(0, len(frames) - 1, min(n_tiles, len(frames)))
            tile_frames = [int(frames[i]) for i in picks.round().astype(int)]

    crops: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    display_settings: dict = {}
    crop_box = (0, 0, 0, 0)
    if tile_frames:
        labels = tifffile.imread(source.input_path("labels"))
        raw_full = tifffile.imread(source.input_path("raw"))
        raw = raw_full[offset: offset + labels.shape[0]]
        if raw.shape != labels.shape: raise ValueError("Raw and label stacks do not align")

        # One crop for the whole strip, big enough for the cell at its largest, so
        # a tile that looks bigger is a bigger cell and not a tighter crop.
        centre_y = int(round(cell["centroid_y"].median()))
        centre_x = int(round(cell["centroid_x"].median()))
        half = 0
        for frame_index in frames:
            ys, xs = np.nonzero(labels[frame_index] == identity)
            if ys.size:
                half = max(half, int(np.abs(ys - centre_y).max()),
                           int(np.abs(xs - centre_x).max()))
        half = min(half + 5, 40)
        top = max(centre_y - half, 0)
        left_edge = max(centre_x - half, 0)
        bottom = min(centre_y + half + 1, labels.shape[1])
        right = min(centre_x + half + 1, labels.shape[2])
        crop_box = (top, bottom, left_edge, right)

        display_stack, display_settings = intensity.presentation_stack(
            raw[:, top:bottom, left_edge:right],
            image_filter=image_filter,
            black_percentile=display_black,
            white_percentile=display_white,
            gamma=display_gamma,
            time_gain=display_gain,
            spatial_sigma_px=display_spatial_sigma,
            pool_px=display_pool_px,
            sharpness=display_sharpness,
            noise_multiple=display_noise_multiple,
            pad_frames=display_pad_frames,
            range_frames=tile_frames if display_range_scope == "displayed" else None,
        )
        crops = [display_stack[f] for f in tile_frames]
        masks = [labels[f][top:bottom, left_edge:right] == identity for f in tile_frames]

    shown_by_metric={column:cell[column].to_numpy(float) if trace_view=='raw' else detrended_by_metric[column] for column in metrics}
    plotted_by_metric={}
    for column,values in shown_by_metric.items():
        if trace_layout=='overlay':
            spread=float(np.nanstd(values));values=(values-float(np.nanmean(values)))/spread if np.isfinite(spread) and spread>0 else np.zeros_like(values)
        plotted_by_metric[column]=values
    fitted_by_metric={column:np.full(hours.shape,np.nan) for column in metrics}
    for column in fitted_metrics:
        result=fit_evidence.get(column,{});period=result.get('period_hours')
        if period is None or not np.isfinite(period):continue
        fit_values=plotted_by_metric[column];fit_baseline=None
        if trace_view=='raw':
            fit_values=detrended_by_metric[column];fit_baseline=baseline_by_metric[column]
            if trace_layout=='overlay':
                raw_values=cell[column].to_numpy(float);centre=float(np.nanmean(raw_values));spread=float(np.nanstd(raw_values))
                if np.isfinite(spread) and spread>0:fit_values=fit_values/spread;fit_baseline=(fit_baseline-centre)/spread
                else:fit_values=np.zeros_like(fit_values);fit_baseline=np.zeros_like(fit_baseline)
        if fit_method=='fft_nlls':
            fitted=workbench.fft_nlls_fitted_values(hours,result)
            if trace_view=='raw':fitted=fitted+baseline_by_metric[column]
            if trace_layout=='overlay':
                reference=shown_by_metric[column];centre=float(np.nanmean(reference));spread=float(np.nanstd(reference));fitted=(fitted-centre)/spread if spread>0 else np.zeros_like(fitted)
        elif options.get('descriptive_cosinor',False):
            fitted=workbench.descriptive_cosinor_fitted_values(hours,fit_values,float(period))
            if fit_baseline is not None:fitted=fitted+fit_baseline
        else:
            fitted=np.full(hours.shape,np.nan)
        fitted_by_metric[column]=fitted
    vmin,vmax=0.,1.
    rows = []
    identifiers = ["frame_index", "imagej_frame", "source_imagej_frame"]
    for position, source_row in cell.reset_index(drop=True).iterrows():
        for column in metrics:
            evidence = fit_evidence.get(column, {})
            rows.append({
                "identity": int(identity),
                **{name: source_row[name] for name in identifiers},
                "hours": float(hours[position]),
                "metric": column,
                "raw_value": float(source_row[column]),
                "detrended_value": float(detrended_by_metric[column][position]),
                "baseline_value": float(baseline_by_metric[column][position]),
                "plotted_value": float(plotted_by_metric[column][position]),
                "fitted_value": float(fitted_by_metric[column][position]),
                "trace_layout": trace_layout,
                "trace_view": trace_view,
                "detrend": detrend,
                "detrend_window_hours": detrending["detrend_window_hours"],
                "fit_method": fit_method if column in fitted_metrics else "",
                "fit_period_hours": evidence.get("period_hours", np.nan),
                "fit_p_value": evidence.get("p_value", np.nan),
                "significance_method": evidence.get("significance_method", ""),
                "significance_p_value": evidence.get("significance_p_value", np.nan),
                "significance_q_value": evidence.get("q_value", np.nan),
                "significance_correction": evidence.get("correction", ""),
                "significance_period_hours": evidence.get("significance_period_hours", np.nan),
                "fit_relative_amplitude_error": evidence.get("rae", np.nan),
                "rhythm_status": evidence.get("rhythm_status", ""),
            })
    figure_data = pd.DataFrame(rows)

    evidence_rows = []
    for column, result in fit_evidence.items():
        evidence_rows.append({
            "metric": column,
            "method": result.get("method", fit_method),
            "method_label": result.get("method_label", fit_method),
            "detrend": detrend,
            "detrend_window_hours": detrending["detrend_window_hours"],
            "status": result.get("status", "unknown"),
            "rhythm_status": result.get("rhythm_status", "unknown"),
            "period_hours": result.get("period_hours", np.nan),
            "period_error_hours": result.get("period_error_hours", np.nan),
            "significance_method": result.get("significance_method"),
            "significance_p_value": result.get("significance_p_value"),
            "q_value": result.get("q_value"),
            "correction": result.get("correction"),
            "significance_period_hours": result.get("significance_period_hours"),
            "cycles_observed": result.get("cycles_observed"),
            "period_underdetermined": result.get("period_underdetermined"),
            **{key: result.get(key) for key in ("phase_hours", "phase_error_hours",
                "amplitude", "amplitude_error", "rae", "components", "diagnostics")},
            "p_value": result.get("p_value", np.nan),
            "alpha": result.get("alpha", np.nan),
            "goodness_of_fit": result.get("goodness_of_fit", np.nan),
            "workbench_version": result.get(
                "workbench_version", workbench.WORKBENCH_VERSION),
            "message": result.get("message", ""),
        })
    auxiliary = {
        "stored_rhythm_fits_this_cell.csv": stored_fits,
        "selected_fit_evidence.csv": pd.DataFrame(evidence_rows),
    }
    statistical_rows = []
    for row in evidence_rows:
        p_value, alpha = row["significance_p_value"], row["alpha"]
        if (p_value is None or alpha is None
                or not np.isfinite(p_value) or not np.isfinite(alpha)):
            continue
        statistical_rows.append({
            "identity": int(identity),
            "metric": row["metric"],
            "test_name": workbench.PERIOD_METHODS[significance_method]["label"],
            "estimate_name": "best period",
            "estimate": row["period_hours"],
            "estimate_units": "h",
            "effect_size_name": "periodogram goodness of fit",
            "effect_size": row["goodness_of_fit"],
            "p_value": p_value,
            "q_value": row["q_value"],
            "correction_method": row["correction"],
            "alpha": alpha,
            "significant": bool(float(row["q_value"]) < float(alpha)),
            "n": int(cell[row["metric"]].notna().sum()),
            "detrend": detrend,
            "detrend_window_hours": detrending["detrend_window_hours"],
            "period_search_min_hours": float(rhythm_params["period_search_hours"][0]),
            "period_search_max_hours": float(rhythm_params["period_search_hours"][1]),
            "workbench_version": row["workbench_version"],
        })
    if statistical_rows:
        auxiliary["statistics.csv"] = pd.DataFrame(statistical_rows)
    if tile_frames:
        auxiliary["display_settings.csv"] = pd.DataFrame([display_settings])
        auxiliary["tile_index.csv"] = pd.DataFrame({
            "position": range(len(tile_frames)),
            "frame_index": tile_frames,
            "hours": [float(cell.loc[cell.frame_index == f, "hours"].iloc[0])
                      for f in tile_frames],
            "area_px": [float(cell.loc[cell.frame_index == f, "area_px"].iloc[0])
                        if "area_px" in cell.columns else np.nan for f in tile_frames],
            "crop_top": crop_box[0], "crop_bottom": crop_box[1],
            "crop_left": crop_box[2], "crop_right": crop_box[3],
            "display_vmin": vmin, "display_vmax": vmax,
            "lut": cell_lut or "gray",
            "display_only": True,
            "image_filter": display_settings["image_filter"],
            "display_gain": display_settings["time_gain"],
            "display_gamma": display_settings["gamma"],
        })

    fit_notes={}
    for column,result in fit_evidence.items():
        period=result.get('period_hours');q=result.get('q_value')
        if period is None or not np.isfinite(period):
            fit_notes[column]='Period unavailable: '+str(result.get('message') or result.get('status','unknown'))
            continue
        test_text=(f', {significance_method} q = {float(q):.3g}' if q is not None and np.isfinite(q) else ', no corrected significance value')
        fit_notes[column]=(f"{describe(column).label}: {float(period):.2f} h, "
                           +str(result.get('method_label',fit_method))+test_text+', '
                           +str(result.get('rhythm_status') or 'unknown'))
        if result.get('period_underdetermined'):fit_notes[column]+='; insufficient observed cycles'
    from skimage.segmentation import find_boundaries
    boundaries=[find_boundaries(mask,mode='inner') for mask in masks]
    tile_table=auxiliary.get('tile_index.csv',pd.DataFrame())
    return PreparedViews({'tiles':dict(table=tile_table,crops=crops,boundaries=boundaries,lut=cell_lut or 'gray',outline=outline),'traces':dict(table=figure_data,metrics=metrics,layout=trace_layout,trace_luts=trace_luts,fit_evidence=fit_evidence,fit_notes=fit_notes)},auxiliary=auxiliary,wording=dict(title=f'Cell report card, identity {identity}',subtitle=f'{len(cell)} observed frames; image preparation affects display only.',footnote='Each measurement retains its own estimated period and significance test. Dashed curves are explicitly requested descriptive models; no common period is imposed.')),None
