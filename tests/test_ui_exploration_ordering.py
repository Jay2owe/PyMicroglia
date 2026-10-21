"""Barcode ordering reuses matrix engines without rewriting scientific data."""
import csv
import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from pymicroglia.figure_tables.ordering import trace_order_catalogue, trace_pattern_order
from pymicroglia.ui.exploration_data import load_exploration
from pymicroglia.ui.exploration_ordering import load_barcode_order
from pymicroglia.ui.server import create_app


@pytest.fixture
def pattern_run(tmp_path):
    folder = tmp_path / "measure/patterns"
    folder.mkdir(parents=True)
    time = np.arange(15, dtype=float)
    # Identities intentionally do not follow the temporal pattern.
    signals = {25: np.exp(-((time - 10) / 2) ** 2),
               3: np.exp(-((time - 6) / 2) ** 2),
               42: np.exp(-((time - 2) / 2) ** 2),
               19: np.full(15, np.nan)}
    with (folder / "cell_frame.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["identity", "frame_index", "hours", "integrated_density", "area_px"])
        for identity, values in signals.items():
            for frame, value in enumerate(values):
                if identity == 3 and frame == 4:
                    continue
                writer.writerow([identity, frame, frame / 2, value, frame + identity])
    (tmp_path / "manifest.json").write_text(json.dumps({"movies": [{"stem": "patterns"}]}))
    return tmp_path, signals


@pytest.mark.parametrize("engine", ["principal_component_gradient", "spectral_continuum"])
def test_existing_engine_parity_with_gaps_and_unavailable_rows(pattern_run, engine):
    root, signals = pattern_run
    before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    data = load_exploration(root)
    assert data["barcode_ordering"] == trace_order_catalogue()
    matrix = pd.DataFrame.from_dict(signals, orient="index").sort_index()
    matrix.loc[3, 4] = np.nan
    normalized = matrix.sub(matrix.min(axis=1), axis=0).div(matrix.max(axis=1) - matrix.min(axis=1), axis=0)
    expected = trace_pattern_order(normalized, method=engine)
    result = load_barcode_order(root, "patterns", "integrated_density", engine)
    assert result["identities"] == expected
    assert set(result["identities"]) == set(signals)
    assert result["identities"][-1] == 19  # missing signal remains an identity
    assert before == {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    # Ordering changes neither the measured values nor the observation gap.
    assert load_exploration(root)["cells"] == data["cells"]


def test_mean_scale_and_source_changes_invalidate_order(pattern_run):
    root, _ = pattern_run
    result = load_barcode_order(root, "patterns", "area_px", time_mean=3, scale="recording")
    assert result["time_mean"] == 3 and result["scale"] == "recording"
    path = root / "measure/patterns/cell_frame.csv"
    path.write_text(path.read_text().replace("42,", "7,"))
    changed = load_barcode_order(root, "patterns", "area_px", time_mean=3, scale="recording")
    assert set(changed["identities"]) == {3, 7, 19, 25}


def test_routes_default_off_switch_and_validation(pattern_run):
    root, _ = pattern_run
    client = TestClient(create_app())
    params = {"run": str(root), "stem": "patterns", "metric": "integrated_density"}
    route = "/api/exploration/barcode-order"
    assert client.get(route, params=params).json()["engine"] == trace_order_catalogue()["default"]
    assert client.get(route, params={**params, "engine": "identity"}).json()["identities"] == [3, 19, 25, 42]
    assert client.get(route, params={**params, "engine": "spectral"}).json()["engine"] == "spectral_continuum"
    for options in ({"engine": "unknown"}, {"time_mean": 2}, {"scale": "unknown"}, {"metric": "missing"}):
        assert client.get(route, params={**params, **options}).status_code == 422
    assert client.get("/api/jobs").json() == []
