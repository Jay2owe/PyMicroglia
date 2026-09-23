"""Post-tracking eligibility changes consumers, never Motion identities."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from pymicroglia import knowledge, registry
from pymicroglia.eligibility import audit, evaluate


def labels() -> np.ndarray:
    values = np.zeros((10, 4, 4), np.uint16)
    values[:, 0, 0] = 1
    values[[0, 1, 7, 8, 9], 0, 2] = 2
    values[:5, 2, 0] = 3
    values[[0, 1, 6, 7, 8, 9], 2, 2] = 4
    return values


def test_default_rules_distinguish_long_gaps_missing_fraction_and_boundaries():
    rows = {row["identity"]: row for row in
            audit(labels(), frame_interval_h=1.0)}
    assert rows[1]["eligible"] is True
    assert rows[2]["longest_internal_gap_hours"] == 5.0
    assert "internal_gap_over_limit" in rows[2]["exclusion_reasons"]
    assert rows[3]["longest_internal_gap_hours"] == 0.0
    assert rows[3]["missing_fraction"] == 0.5
    assert rows[3]["exclusion_reasons"] == \
        "missing_fraction_at_or_over_limit"
    assert rows[4]["longest_internal_gap_hours"] == 4.0
    assert rows[4]["missing_fraction"] == pytest.approx(0.4)
    assert rows[4]["eligible"] is True


def test_destinations_receive_independent_id_preserving_views(tmp_path: Path):
    source = tmp_path / "motion_labels.tif"
    tifffile.imwrite(source, labels(), photometric="minisblack")
    result = evaluate(source, output_dir=tmp_path / "eligibility",
                      frame_interval_h=1.0,
                      exclude_from=("analysis", "images"))

    assert result["eligible_identities"] == [1, 4]
    assert result["excluded_identities"] == [2, 3]
    assert result["views"]["videos"]["labels"] == str(source)
    assert result["views"]["videos"]["filtered"] is False
    for destination in ("analysis", "images"):
        filtered = tifffile.imread(result["views"][destination]["labels"])
        assert set(np.unique(filtered)) == {0, 1, 4}
        assert result["views"][destination]["filtered"] is True
    assert set(np.unique(tifffile.imread(source))) == {0, 1, 2, 3, 4}


def test_thresholds_and_destination_list_are_tunable(tmp_path: Path):
    source = tmp_path / "motion_labels.tif"
    tifffile.imwrite(source, labels())
    result = evaluate(source, output_dir=tmp_path / "relaxed",
                      frame_interval_h=1.0, max_gap_hours=5.0,
                      max_missing_fraction=0.6, exclude_from="videos")
    assert result["excluded_identities"] == []
    assert result["exclude_from"] == ["videos"]
    with pytest.raises(ValueError, match="unknown destinations"):
        evaluate(source, output_dir=tmp_path / "bad", frame_interval_h=1.0,
                 exclude_from=("statistics",))


def test_outputs_refuse_implicit_overwrite(tmp_path: Path):
    source = tmp_path / "motion_labels.tif"
    tifffile.imwrite(source, labels())
    options = {"output_dir": tmp_path / "eligibility",
               "frame_interval_h": 1.0}
    evaluate(source, **options)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        evaluate(source, **options)
    evaluate(source, **options, overwrite=True)


def test_action_is_live_and_documents_the_two_default_limits():
    assert registry.REGISTRY.resolve("cell_eligibility") is evaluate
    details = knowledge.describe("cell_eligibility")
    params = {row["name"]: row for row in details["params"]}
    assert params["max_gap_hours"]["default"] == 4.0
    assert params["max_missing_fraction"]["default"] == 0.5
    assert tuple(params["exclude_from"]["default"]) == ("analysis",)
    assert details["pending"] is False
