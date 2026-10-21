import csv
import hashlib
import json

import numpy as np
import pytest
import tifffile

from pymicroglia.ui.exploration_previews import load_contact_preview, load_phase_preview


@pytest.fixture
def preview_run(tmp_path):
    folder = tmp_path / "measure/test"
    folder.mkdir(parents=True)
    with (folder / "cell_frame.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["identity", "frame_index", "hours", "integrated_density"])
        for identity, peak in [(19, 6), (26, 18), (30, None)]:
            for frame in range(100):
                if identity == 19 and 35 <= frame <= 40:
                    continue
                hours = 7 + frame / 2
                value = 10 if peak is None else 10 + np.cos(2 * np.pi * (hours - peak) / 24)
                writer.writerow([identity, frame, hours, value])
    labels = np.zeros((100, 16, 16), dtype=np.uint16)
    labels[:, 2:5, 2:5] = 19
    labels[0, 2:5, 6:9] = 26
    labels[1:, 11:14, 11:14] = 26
    tifffile.imwrite(tmp_path / "labels.tif", labels, photometric="minisblack")
    (tmp_path / "manifest.json").write_text(json.dumps({"movies": [{
        "stem": "test", "spec": {"labels": "labels.tif", "source_frame_offset": 14}}]}))
    return tmp_path


def files(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


def test_automatic_phases_keep_recording_time_gaps_and_saved_files(preview_run):
    before = files(preview_run)
    result = load_phase_preview(preview_run, "test")
    assert result["preview"]
    rows = {r["identity"]: r for r in result["rows"]}
    assert rows[19]["phase_hours"] == pytest.approx(6)
    assert rows[26]["phase_hours"] == pytest.approx(18)
    assert rows[19]["phase_fraction"] == pytest.approx(.25)
    assert rows[19]["status"] == "unknown"
    assert 30 not in rows  # constant signals have no meaningful phase
    assert "not a period estimate" in result["meaning"]
    assert files(preview_run) == before
    with pytest.raises(ValueError, match="not available"):
        load_phase_preview(preview_run, "test", "nonexistent")


def test_contacts_use_selected_mask_frame_and_keep_identity_and_time(preview_run):
    before = files(preview_run)
    source = load_contact_preview(preview_run, "test", 0)["sources"][0]
    assert source["preview"] and source["identities"] == [19, 26]
    assert len(source["rows"]) == 1
    link = source["rows"][0]
    assert (link["identity_a"], link["identity_b"]) == (19, 26)
    assert link["frame_index"] == 0 and link["hours"] == 7
    assert link["dilation_px"] == 2
    assert load_contact_preview(preview_run, "test", 1)["sources"][0]["rows"] == []
    assert files(preview_run) == before
    with pytest.raises(ValueError, match="outside"):
        load_contact_preview(preview_run, "test", 100)


def test_saved_analyses_take_precedence(preview_run, monkeypatch):
    folder = preview_run / "measure/test"
    (folder / "rhythms.csv").write_text("identity,metric,best_period_hours,best_phase_hours\n19,integrated_density,30,7.5\n")
    (folder / "contacts_frame.csv").write_text("identity_a,identity_b,frame_index,dilation_px,shared_boundary_px\n19,26,1,4,3\n")
    import pymicroglia.ui.exploration_previews as previews
    def unexpected(*args, **kwargs):
        pytest.fail("Saved analyses must not be recalculated")
    monkeypatch.setattr(previews, "_phases", unexpected)
    monkeypatch.setattr(previews, "_contacts", unexpected)
    assert load_phase_preview(preview_run, "test")["rows"][0]["period_hours"] == 30
    source = load_contact_preview(preview_run, "test", 1)["sources"][0]
    assert source["rows"][0]["dilation_px"] == 4
    assert not source.get("preview")


def test_preview_routes_populate_without_submitting_jobs(preview_run):
    from fastapi.testclient import TestClient
    from pymicroglia.ui.server import create_app
    with TestClient(create_app()) as client:
        query = {"run": str(preview_run), "stem": "test"}
        phases = client.get("/api/exploration/phases", params=query)
        contacts = client.get("/api/exploration/contacts", params={**query, "frame": 0})
        assert phases.status_code == contacts.status_code == 200
        assert len(phases.json()["rows"]) == 2
        assert len(contacts.json()["sources"][0]["rows"]) == 1
        assert client.get("/api/exploration/contacts", params={**query, "frame": -1}).status_code == 422
        assert client.get("/api/jobs").json() == []
