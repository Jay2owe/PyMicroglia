from pymicroglia.ui.catalog import load_map, validate_map


def test_live_navigation_coverage_and_order():
    data = load_map()
    assert [page["key"] for page in data["pages"][:2]] == ["rhythms", "tissue"]
    assert validate_map(data) == []


def test_missing_action_is_reported():
    data = load_map()
    data["inventory"] = [row for row in data["inventory"] if row["name"] != "track"]
    assert any("track" in error for error in validate_map(data))
