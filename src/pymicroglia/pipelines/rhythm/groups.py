"""Saved rhythm-defined groups, explicit comparison summaries and sample pairing."""
from pymicroglia._results import read_document

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pymicroglia.pipelines._contracts import ArtifactRef, Settings, StepResult, content_id
from pymicroglia.pipelines._screening import _write_json, file_hash, read_screen, read_verified_tables, write_table

GROUPS = ("significant", "not-significant")
KEYS = ["source_run", "movie", "identity"]
CONTEXT = {
    "screen_duration_hours": ("Observed recording span", "h"),
    "screen_invalid_fraction": ("Unusable supplied observations", "fraction"),
}
INTERPRETATION = ("Exploratory comparison of groups selected using these recordings. "
                  "Differences do not independently validate rhythmicity; not-significant does not mean biologically arrhythmic.")


def _contrast(declaration):
    from pymicroglia.measure.spec import ContrastSpec
    from pymicroglia.measure.contrast_statistics import TESTS, PAIRED_TESTS, CORRECTIONS, SINGLE_GROUP_TESTS
    return ContrastSpec.from_dict({
        **declaration, "table": "saved_rhythm_groups", "group_by": "group", "groups": list(GROUPS),
        "description": INTERPRETATION}, tuple(TESTS), tuple(PAIRED_TESTS), tuple(CORRECTIONS), tuple(SINGLE_GROUP_TESTS), {})


def validate_options(declared, tested, compared):
    if not isinstance(declared, dict) or set(declared) - {"aggregate", "tests", "resamples", "random_state"}:
        raise ValueError("group_comparisons accepts aggregate, tests, resamples and random_state")
    result = {"aggregate": None, "tests": [], "resamples": 2000, "random_state": 20260902, **declared}
    if result["aggregate"] not in {None, "mean", "median"}:
        raise ValueError("group_comparisons.aggregate must be mean or median, or omitted for cell distributions only")
    for key in ("resamples", "random_state"):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < (100 if key == "resamples" else 0):
            raise ValueError(f"group_comparisons.{key} must be an integer >= {100 if key == 'resamples' else 0}")
    if not isinstance(result["tests"], list):
        raise ValueError("group_comparisons.tests must be a list of declared tests")
    names, families = set(), {}
    for item in result["tests"]:
        allowed = {"name", "grouping_measurement", "metrics", "unit", "test", "aggregate", "family", "correction", "alpha"}
        if not isinstance(item, dict) or set(item) - allowed:
            raise ValueError("A group comparison test needs a named grouping measurement, comparison metrics and explicit contrast settings")
        if item.get("grouping_measurement") not in tested:
            raise ValueError("Each formal grouping_measurement must be a selected test measurement; any-significant is descriptive only")
        contrast = _contrast(item)
        if contrast.name in names:
            raise ValueError("Group comparison test names must be unique")
        names.add(contrast.name)
        if set(contrast.metrics) - set(compared):
            raise ValueError("Formal metrics must be in the independently selected comparison_measurements list")
        if len(set(contrast.metrics)) != len(contrast.metrics):
            raise ValueError("Formal comparison metrics must be unique")
        settings = (contrast.correction, contrast.alpha)
        if contrast.family in families and families[contrast.family] != settings:
            raise ValueError("Each group comparison family needs one correction and alpha")
        families[contrast.family] = settings
    return result


def version():
    import pymicroglia.measure.contrast_statistics as contrasts
    import pymicroglia.measure.summarise as summarise
    return content_id({"producer": file_hash(__file__), "contrasts": file_hash(contrasts.__file__),
                       "summaries": file_hash(summarise.__file__)})


def identity(context):
    resolved = context.request
    hashes = {m.table: resolved.inputs.table_hashes[m.table] for m in resolved.comparison_measurements}
    read_verified_tables(context.table_paths, hashes)
    return content_id({"producer": version(), "screen": context.saved("rhythm-screen").outcome.scientific_id,
        "measurements": resolved.comparison_measurements, "inputs": hashes,
        "settings": resolved.request.group_comparisons})


def comparison_values(resolved, tables):
    """One explicit scalar per cell and requested comparison; never implicit averaging."""
    from pymicroglia.measure.summarise import _STATISTICS
    rows = []
    for measurement in resolved.comparison_measurements:
        frame = tables[measurement.table].copy()
        if measurement.summary is not None and measurement.summary not in _STATISTICS:
            raise ValueError(f"Comparison summary {measurement.summary!r} is unavailable; existing summaries are {', '.join(_STATISTICS)}")
        values = pd.to_numeric(frame[measurement.column], errors="coerce")
        valid = np.isfinite(values)
        if measurement.summary is not None:
            valid &= np.isfinite(pd.to_numeric(frame.hours, errors="coerce"))
        frame[measurement.column] = values.where(valid)
        blocks = {key: block for key, block in frame.groupby(["stem", "identity"], sort=False)}
        for cell in resolved.inputs.cells:
            block = blocks.get((cell.movie, cell.identity), frame.iloc[:0])
            usable = int(block[measurement.column].notna().sum())
            value = np.nan
            if usable:
                if measurement.summary is None:
                    if len(block) != 1: raise ValueError("Comparison scalar has ambiguous cell membership")
                    value = float(block[measurement.column].iloc[0])
                else:
                    value = float(_STATISTICS[measurement.summary](block.groupby(["stem", "identity"]), measurement.column).iloc[0])
            rows.append({**cell.as_dict(), "comparison": measurement.column, "value": value,
                "comparison_label": measurement.label, "comparison_unit": measurement.unit,
                "comparison_table": measurement.table, "comparison_summary": measurement.summary or "saved per-cell value",
                "comparison_rows": len(block), "comparison_usable": usable,
                "comparison_reason": "" if np.isfinite(value) else ("No supplied comparison rows" if block.empty else "No finite or defined comparison summary")})
    columns = [*KEYS, "comparison", "value", "comparison_label", "comparison_unit", "comparison_table",
               "comparison_summary", "comparison_rows", "comparison_usable", "comparison_reason"]
    return pd.DataFrame(rows, columns=columns)


def membership(screen):
    """One group per valid test; an incomplete negative union remains excluded."""
    evidence = ["method", "significance_method", "p_value", "q_value", "period_hours",
                "period_available", "period_underdetermined", "estimate_status", "reason", "family_id"]
    base = screen.results[[*KEYS, "measurement", "status", "sample", "sample_confirmed",
                           "span_hours", "input_rows", "invalid_observations", *evidence]].copy()
    base = base.rename(columns={"measurement": "grouping_measurement", "status": "group"})
    base["grouping_kind"] = "measurement"
    base["group_reason"] = np.where(base.group.eq("untestable"), "No valid significance test", "")
    base["valid_tests"] = base.group.ne("untestable").astype(int)
    base["requested_tests"] = 1
    unions = []
    for _, block in base.groupby(KEYS, sort=False):
        row = block.iloc[0].to_dict()
        significant = block.group.eq("significant").any()
        complete_negative = block.group.eq("not-significant").all()
        row.update(grouping_kind="union", grouping_measurement="any-significant", group="significant" if significant else "not-significant" if complete_negative else "untestable",
            group_reason="Display union, not a cell-level significance test" if significant or complete_negative else
                         "Incomplete negative union: at least one measurement has no valid test",
            valid_tests=int(block.valid_tests.sum()), requested_tests=len(block),
            span_hours=np.nan, input_rows=np.nan, invalid_observations=np.nan)
        # No fabricated union p-value, period, method or correction family.
        row.update({name: None for name in evidence})
        unions.append(row)
    return pd.DataFrame([*base.to_dict("records"), *unions], columns=base.columns)


def descriptive_summary(cells):
    rows = []
    dimensions = ["grouping_kind", "grouping_measurement", "comparison_kind", "comparison"]
    for key, block in cells.groupby(dimensions, sort=False):
        for group in (*GROUPS, "untestable"):
            selected = block[block.group.eq(group)]
            finite = selected[np.isfinite(selected.value)]
            rows.append({**dict(zip(dimensions, key)), "group": group,
                "population_cells": len(block), "group_cells": len(selected), "finite_cells": len(finite),
                "missing_values": len(selected) - len(finite), "movies": selected.movie.nunique(),
                "confirmed_samples": selected.loc[selected.sample_confirmed, "sample"].nunique(),
                "unknown_sample_cells": int((~selected.sample_confirmed).sum()),
                "mean": finite.value.mean(), "median": finite.value.median(),
                "minimum": finite.value.min(), "maximum": finite.value.max(),
                "comparison_label": block.comparison_label.iloc[0], "comparison_unit": block.comparison_unit.iloc[0],
                "comparison_summary": block.comparison_summary.iloc[0]})
    result = pd.DataFrame(rows)
    if not result.empty:
        for statistic in ("mean", "median"):
            wide = result.pivot(index=dimensions, columns="group", values=statistic)
            result[statistic + "_difference"] = [wide.loc[tuple(r[d] for d in dimensions), GROUPS[0]] - wide.loc[tuple(r[d] for d in dimensions), GROUPS[1]] for r in result.to_dict("records")]
    return result


def unit_ledger(cells, *, name, unit, aggregate):
    """Retain every unit, including one-sided units and wholly missing values."""
    import json
    from pymicroglia.measure.contrast_statistics import to_units, UNIT_KEYS
    frame = cells[cells.group.isin(GROUPS)].rename(columns={"movie": "stem"}).copy()
    frame["subject"] = frame["sample"].where(frame.sample_confirmed)
    if unit == "subject": frame = frame[frame.sample_confirmed]
    keys = list(UNIT_KEYS[unit])
    result = []
    dimensions = ["grouping_kind", "grouping_measurement", "comparison_kind", "comparison"]
    for dimension_key, block in frame.groupby(dimensions, sort=False):
        values = {group: {tuple(r[k] for k in keys): r for r in to_units(block[block.group.eq(group)], "value", unit, aggregate).to_dict("records")} for group in GROUPS}
        for key, original in block.groupby(keys, sort=True):
            key = key if isinstance(key, tuple) else (key,)
            a, b = (values[g].get(key, {}) for g in GROUPS)
            result.append({"contrast": name, **dict(zip(dimensions, dimension_key)),
                "unit": unit, "unit_key": json.dumps({k: v.item() if isinstance(v, np.generic) else v for k, v in zip(keys, key)}, sort_keys=True), "aggregate": aggregate or "",
                "value_a": a.get("value", np.nan), "value_b": b.get("value", np.nan),
                "cells_a": a.get("cells", 0), "cells_b": b.get("cells", 0),
                "supplied_cells_a": int(original.group.eq(GROUPS[0]).sum()), "supplied_cells_b": int(original.group.eq(GROUPS[1]).sum()),
                "paired": bool(a and b), "difference": a.get("value", np.nan) - b.get("value", np.nan),
                "reason": "" if a and b else "No finite value in both groups; excluded from paired differences"})
    return result


def design_reason(cells, contrast):
    """A declared scalar test cannot silently erase nested or repeated samples."""
    from pymicroglia.measure.contrast_statistics import PAIRED_TESTS
    if cells.empty: return "No testable cells in either rhythm-defined group"
    if not cells.sample_confirmed.all(): return "Biological sample mapping is unconfirmed for one or more grouped cells"
    if contrast.unit == "cell": return "Cell independence is not established by a movie-to-sample map; use a declared biological-sample unit"
    if contrast.unit == "movie" and cells[["movie", "sample"]].drop_duplicates()["sample"].value_counts().gt(1).any():
        return "Multiple movies share a biological sample; movie counts cannot substitute for sample counts"
    key = "sample" if contrast.unit == "subject" else "movie"
    overlap = set(cells.loc[cells.group.eq(GROUPS[0]), key]) & set(cells.loc[cells.group.eq(GROUPS[1]), key])
    if overlap and contrast.test not in PAIRED_TESTS:
        return "The same experimental unit supplies both groups; declare a paired test to preserve this design"
    return ""


def produce(context):
    import pymicroglia.measure.contrast_statistics as contrasts
    resolved = context.request
    settings = validate_options(resolved.request.group_comparisons.as_dict(),
        [m.column for m in resolved.test_measurements], [m.column for m in resolved.comparison_measurements])
    saved = context.saved("rhythm-screen")
    screen = read_screen(saved.root)
    hashes = {m.table: resolved.inputs.table_hashes[m.table] for m in resolved.comparison_measurements}
    values = comparison_values(resolved, read_verified_tables(context.table_paths, hashes))
    groups = membership(screen)
    cells = groups.merge(values, on=KEYS, how="inner", validate="many_to_many")
    cells["comparison_kind"] = "measurement"
    extra = []
    for row in groups[groups.grouping_kind.eq("measurement")].to_dict("records"):
        for metric, (label, unit) in CONTEXT.items():
            value = row["span_hours"] if metric == "screen_duration_hours" else (
                row["invalid_observations"] / row["input_rows"] if row["input_rows"] else np.nan)
            extra.append({**row, "comparison_kind": "context", "comparison": metric, "value": value, "comparison_label": label, "comparison_unit": unit,
                "comparison_table": "saved rhythm screen", "comparison_summary": "saved span" if metric == "screen_duration_hours" else "invalid observations / supplied observations",
                "comparison_rows": row["input_rows"], "comparison_usable": row["input_rows"] - row["invalid_observations"],
                "comparison_reason": "" if np.isfinite(value) else "No supplied screening observations"})
    cells = pd.concat([cells, pd.DataFrame(extra, columns=cells.columns)], ignore_index=True)
    summary = descriptive_summary(cells)
    ledgers = []
    if settings["aggregate"]:
        for unit in ("movie", "subject"):
            ledgers.extend(unit_ledger(cells, name="descriptive", unit=unit, aggregate=settings["aggregate"]))
    test_rows = []
    for item in settings["tests"]:
        contrast = _contrast(item)
        for metric in contrast.metrics:
            block = cells[cells.grouping_kind.eq("measurement") & cells.grouping_measurement.eq(item["grouping_measurement"]) & cells.comparison_kind.eq("measurement") & cells.comparison.eq(metric) & cells.group.isin(GROUPS)].copy()
            ledgers.extend(unit_ledger(block, name=contrast.name, unit=contrast.unit, aggregate=contrast.aggregate))
            reason = design_reason(block, contrast)
            if reason:
                row = contrasts._blank_row(contrast, metric, reason)
            else:
                block["_comparison_value"] = block.value
                block["stem"], block["subject"] = block.movie, block["sample"]
                row = contrasts.evaluate_contrast_rows(replace(contrast, metrics=("_comparison_value",)), block,
                    resamples=settings["resamples"], random_state=settings["random_state"])[0]
                row["metric"] = metric
                row["methods"] = row["methods"].replace("on _comparison_value;", f"on {metric};")
            test_rows.append(row)
    statistics = contrasts.correct_declared_results(test_rows)
    if not statistics.empty:
        statistics["grouping_measurement"] = statistics.contrast.map({x["name"]: x["grouping_measurement"] for x in settings["tests"]})
        statistics["grouping_kind"] = "measurement"
        statistics["interpretation"] = INTERPRETATION
    ledger_columns = ["contrast", "grouping_kind", "grouping_measurement", "comparison_kind", "comparison", "unit", "unit_key", "aggregate", "value_a", "value_b",
        "cells_a", "cells_b", "supplied_cells_a", "supplied_cells_b", "paired", "difference", "reason"]
    frames = {"cells": cells, "groups": groups, "summary": summary,
              "units": pd.DataFrame(ledgers, columns=ledger_columns), "statistics": statistics}
    context.output.mkdir(parents=True, exist_ok=True)
    refs = []
    for name, frame in frames.items():
        path = context.output / (name + ".json")
        path = write_table(path, frame)
        refs.append(ArtifactRef(name, path.name, file_hash(path), context.scientific_id))
    _write_json(context.output / "provenance.json", {"schema_version": 1, "screen_id": saved.outcome.scientific_id,
        "source_screen_provenance": read_document(saved.artifact("provenance")),
        "settings": settings, "measurements": [m.as_dict() for m in resolved.comparison_measurements], "input_hashes": hashes,
        "sample_mapping": [s.as_dict() for s in resolved.inputs.samples], "interpretation": INTERPRETATION,
        "union_policy": "Any significant test; the negative comparison requires valid non-significant tests for every requested measurement",
        "missingness_denominator": "Supplied screening rows, including invalid values/times; unsupplied frames are not included",
        "context_union": "Recording context remains measurement-specific; no synthetic union duration/missingness is invented",
        "correction_policy": "Each intended comparison contributes to its declared family; unavailable tests contribute p=1 only to correction, with saved probabilities missing",
        "producer": version(), "rhythm_analysis_recomputed": False})
    refs.append(ArtifactRef("provenance", "provenance.json", file_hash(context.output / "provenance.json"), context.scientific_id))
    return StepResult(context.step.name, context.scientific_id, "completed",
        "Saved rhythm-defined comparisons, exclusions, sample pairing and every declared test outcome", tuple(refs))
