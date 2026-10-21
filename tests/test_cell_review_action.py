"""Reusable unattended export action and shipped asset contracts."""
import inspect
import json
from pathlib import Path

from test_cell_review_data import recording
from pymicroglia import catalogue, run_action
from pymicroglia.registry import CLAIM_TEMPLATES, pending, resolve
from pymicroglia.visualisation.cell_review import ASSETS, SCRIPTS, cell_review


def test_action_resolves_with_its_actual_signature():
    entry = next(item for item in catalogue.actions() if item["name"] == "cell_review")
    assert entry["display_only"] and not entry["destructive"]
    assert set(entry["params"]) == set(inspect.signature(cell_review).parameters)
    assert "cell_review" not in pending()
    assert "{source}" in CLAIM_TEMPLATES["cell_review"]
    assert resolve(entry["method"]) is cell_review
    assert all((ASSETS / name).is_file() for name in (*SCRIPTS, "review.html", "review.css"))


def test_registered_action_exports_without_a_human_gate(recording, tmp_path):
    output = tmp_path / "action_review.html"
    result = run_action("cell_review", raw=str(recording[0]), labels=str(recording[1]),
                        output=str(output), source_frame_offset=1,
                        output_roots=[tmp_path], claim="review the two saved identities")
    assert output.is_file()
    receipt = json.loads((tmp_path / "action_review_review.json").read_text())
    assert receipt["ids"] == [19, 26] and receipt["display_only"]
    assert result is not None
