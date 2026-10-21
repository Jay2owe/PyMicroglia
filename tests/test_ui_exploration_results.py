import csv
from pymicroglia.ui.exploration_results import saved_results
from test_ui_exploration_data import saved_run


def write(root, name, rows):
    path = root / "measure/one" / (name + ".csv")
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def test_saved_phase_and_pair_meaning(saved_run):
    write(saved_run, "rhythms", [{"identity": 19, "metric": "integrated_density", "best_period_hours": 30, "best_phase_hours": 7.5, "rhythm_status": "unknown"}, {"identity": 26, "metric": "integrated_density", "best_period_hours": "nan", "best_phase_hours": 0, "rhythm_status": "unknown"}])
    write(saved_run, "contacts_frame", [{"identity_a": 19, "identity_b": 26, "frame_index": 7, "dilation_px": 2, "shared_boundary_px": 4}])
    write(saved_run, "lifecycle_events", [{"identity": 19, "frame_index": 8, "event": "departure", "event_status": "censored"}])
    result = saved_results(saved_run, {"stem": "one"})
    assert result["rhythms"]["rows"][0]["phase_fraction"] == .25
    assert result["rhythms"]["rows"][0]["period_hours"] == 30
    assert len(result["rhythms"]["rows"]) == 1
    link = result["relations"]["sources"][0]
    assert link["rows"][0]["frame_index"] == 7
    assert "dilation 2 px" in link["title"]
    assert "censored" in result["events"]["rows"][0]["label"]


def test_missing_results_are_explicit(saved_run):
    result = saved_results(saved_run, {"stem": "one"})
    assert result["rhythms"]["rows"] == []
    assert result["relations"]["sources"] == []
    assert "No saved" in result["rhythms"]["message"]
