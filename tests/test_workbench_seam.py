"""The package has one scientific dependency doorway and preserves trace clocks."""
import ast
from pathlib import Path
import numpy as np
import pytest


def test_only_the_public_seam_imports_workbench():
    root = Path(__file__).parents[1] / "src/pymicroglia"
    importers = set()
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            if any(n.split(".")[0] == "circadian_workbench" for n in names):
                importers.add(path.relative_to(root).as_posix())
    assert importers == {"workbench.py"}


def test_fft_component_gateway_does_not_require_package_attribute(monkeypatch):
    import circadian_workbench as core
    from pymicroglia import workbench

    monkeypatch.delattr(core, "component_significance", raising=False)
    expected = [{"period_hours": 24.0, "status": "ok"}]
    monkeypatch.setattr(workbench, "_test_components", lambda *args, **kwargs: expected)

    assert workbench.test_fft_components([0.0, 1.0], [1.0, 2.0], [24.0]) == expected


def test_late_start_normalization_keeps_recording_time():
    from pymicroglia import workbench
    hours = np.arange(12., 30., .5)
    result = workbench.normalize_trace(hours, np.arange(len(hours), dtype=float), method="minmax")
    np.testing.assert_array_equal(result["hours"], hours)
    np.testing.assert_array_equal(result["processed_trace"]["hours"], hours)


def test_estimator_and_significance_remain_separate(monkeypatch):
    from pymicroglia import workbench
    import pandas as pd
    called = []
    def estimate(hours, values, params, method, **kwargs):
        called.append(method)
        return {"status": "ok", "method": method, "period_hours": 8. if method == "fft_nlls" else 12.,
                "p_value": None if method == "fft_nlls" else .001, "significant": method != "fft_nlls"}
    monkeypatch.setattr(workbench, "estimate_one", estimate)
    data = pd.DataFrame({"cell": 1, "hours": np.arange(48.), "value": np.sin(np.arange(48.))})
    result = workbench.estimate_grouped_rhythms(data, group_columns=["cell"], value_column="value",
        params=workbench.PERIOD_ANALYSIS_DEFAULTS, method="fft_nlls", significance_method="lomb")
    assert called == ["fft_nlls", "lomb"]
    assert result.iloc[0].period_hours == 8.
    assert result.iloc[0].p_value == .001


def test_contrasts_use_declared_replication_unit(tmp_path, monkeypatch):
    from pymicroglia.measure.run import write_manifest
    from pymicroglia.measure.contrasts import contrasts
    import pandas as pd
    monkeypatch.setenv("PYMICROGLIA_STORE", str(tmp_path / "cache"))
    monkeypatch.setenv("PYMICROGLIA_INDEX", str(tmp_path / "index"))
    root = tmp_path / "run"
    (root / "pooled").mkdir(parents=True)
    pd.DataFrame([dict(stem=f"m{i}", subject=f"s{i}", identity=1, condition=c, value=v)
                  for c, shift in [("a", 0), ("b", 5)] for i, v in enumerate(np.arange(5.) + shift)]
                 ).to_csv(root / "pooled/cell_summary.csv", index=False)
    spec = dict(name="comparison", table="cell_summary", metrics=["value"], group_by="condition",
                groups=["a", "b"], unit="movie", aggregate="median", test="mannwhitney")
    write_manifest(root, {"settings": {"contrasts": [spec]}, "pooled": {"tables": {}}})
    record = contrasts(root, claim="Compare independent recording summaries")
    result = pd.read_csv(root / "pooled/statistics.csv")
    assert record["rows"] == 1
    assert result.iloc[0]["unit"] == "movie"
    assert result.iloc[0].n_a == result.iloc[0].n_b == 5
    with pytest.raises(ValueError, match="unit"):
        contrasts(root, contrasts=[{k: v for k, v in spec.items() if k != "unit"}])
