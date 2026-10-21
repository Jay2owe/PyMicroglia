"""A small saved Motion history flowing into real PyMicroglia measurements."""

from pathlib import Path
import sys

import numpy as np
import pytest
import tifffile
from fastapi.testclient import TestClient

from pymicroglia.ui.server import create_app


def test_corrected_motion_labels_are_measured_without_changing_source(tmp_path: Path):
    motion_root = Path(__file__).resolve().parents[2] / "Motion"
    if not (motion_root / "motion_edit" / "ui_service.py").is_file():
        pytest.skip("Optional Motion checkout is not available")
    sys.path.insert(0, str(motion_root))
    from motion_edit import ensure_code_path
    from motion_edit.ui_service import create_app as motion_app
    ensure_code_path()
    from manual_editing import prepare_editing_bundle

    labels = np.zeros((6, 20, 20), dtype=np.uint16)
    labels[:, 2:7, 2:7] = 7
    labels[:, 12:17, 12:17] = 12
    raw = np.full((6, 20, 20), 100, dtype=np.uint16)
    raw[:, 2:7, 2:7] = np.arange(6, dtype=np.uint16)[:, None, None] * 20 + 100
    composite = np.repeat(raw[:, None], 3, axis=1)
    for name, array, axes in (("labels", labels, "TYX"), ("raw", raw, "TYX"),
                              ("composite", composite, "TCYX")):
        tifffile.imwrite(tmp_path / f"{name}.tif", array, photometric="minisblack",
                         metadata={"axes": axes})
    bundle = tmp_path / "bundle"
    prepare_editing_bundle(bundle, "synthetic", tmp_path / "labels.tif",
                           tmp_path / "raw.tif", tmp_path / "composite.tif",
                           list(range(6)), 60, spatial_unit="um",
                           pixel_size_y=1, pixel_size_x=1)
    original = (bundle / "motion-editing-bundle.json").read_bytes()
    motion = TestClient(motion_app(root=tmp_path))
    edited = motion.post("/api/action", json={"action": "swap_identities", "params": {
        "source": str(bundle), "identity_a": 7, "identity_b": 12,
        "start_frame": 3, "end_frame": 4}})
    assert edited.status_code == 200, edited.text
    session = edited.json()["session"]
    handed = motion.post("/api/handoff", json={"source": session})
    assert handed.status_code == 200, handed.text
    handoff = handed.json()
    proposal = tifffile.imread(handoff["movie"]["labels"])
    proposal[1, 2:7, 7] = 7
    proposal_file = Path(handoff["folder"]) / "saved_proposal.tif"
    tifffile.imwrite(proposal_file, proposal, photometric="minisblack",
                     metadata={"axes": "TYX"})
    workbench = create_app()
    refined = TestClient(workbench).post("/api/refinements", json={
        "manifest": handoff["manifest"], "method": "saved",
        "options": {"path": str(proposal_file), "kind": "labels"}})
    assert refined.status_code == 202, refined.text
    workbench.state.jobs.pool.shutdown(wait=True)
    job = TestClient(workbench).get(f"/api/jobs/{refined.json()['id']}").json()
    assert job["state"] == "completed", job.get("error")
    assert job["identity_presence"] == "same_as_motion"
    params = job["measure_params"]
    assert params["movies"][0]["labels"] != handoff["movie"]["labels"]
    assert params["enabled_modules"] == ["intensity"]
    result = TestClient(create_app()).post("/api/validate", json={"action": "measure", "params": params})
    assert result.status_code == 200 and result.json()["ok"]
    from pymicroglia.measure import measure
    manifest = measure(**params)
    assert manifest
    run_folders = [item for item in Path(params["output_dir"]).iterdir() if item.is_dir()]
    assert len(run_folders) == 1
    availability = TestClient(create_app()).get(
        "/api/availability/all_cell_trace_grid", params={"run": str(run_folders[0])})
    assert availability.status_code == 200, availability.text
    assert "traces" in availability.json()["views"]
    assert (bundle / "motion-editing-bundle.json").read_bytes() == original
    exported = tifffile.imread(handoff["movie"]["labels"])
    assert exported[0, 3, 3] == 7 and exported[2, 3, 3] == 12 and exported[4, 3, 3] == 7
