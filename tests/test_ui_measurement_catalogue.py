from pymicroglia.ui.measurement_catalogue import GROUPS, RECOMMENDED, measurement_catalogue
from pymicroglia.ui.exploration_data import load_exploration
from test_ui_exploration_data import saved_run


def test_groups_cover_native_recommendations_exactly_once():
    from pymicroglia.measure import modules
    from pymicroglia.measure.declare import declared_columns
    modules.load()
    columns = declared_columns()
    keys = [row[1] for row in RECOMMENDED]
    assert len(keys) == len(set(keys))
    assert set(keys) <= set(columns)
    assert {row[0] for row in RECOMMENDED} == {key for key, _ in GROUPS}
    assert all(row[4] for row in RECOMMENDED)


def test_missing_values_are_discoverable_but_not_available():
    source = [{"key": "integrated_density", "label": "Total light", "unit": "AU"},
              {"key": "custom_cell_measure", "label": "Custom measure", "unit": "widgets"}]
    catalogue = measurement_catalogue(source)
    by_key = {row["key"]: row for row in catalogue["metrics"]}
    assert len(by_key) == len(catalogue["metrics"])
    assert {row["key"] for row in catalogue["metrics"] if row["available"]} == {row["key"] for row in source}
    assert by_key["area_px"]["group"] == "shape"
    assert by_key["area_px"]["status"] == "Not measured yet"
    assert "morphology" in by_key["area_px"]["help"]
    assert by_key["custom_cell_measure"]["group"] == "other"
    assert catalogue["groups"][-1]["key"] == "other"
    assert "available" not in source[0]


def test_calibrated_variants_do_not_show_duplicate_missing_measurements():
    catalogue = measurement_catalogue([{"key": "area_um2", "label": "Area", "unit": "µm²"}])
    assert "area_px" not in {row["key"] for row in catalogue["metrics"]}
    area = next(row for row in catalogue["metrics"] if row["key"] == "area_um2")
    assert area["available"] and area["group"] == "shape"


def test_saved_measurements_keep_values_and_tracking_with_catalogue(saved_run):
    before = {str(path): path.read_bytes() for path in saved_run.rglob("*") if path.is_file()}
    data = load_exploration(saved_run)
    by_key = {row["key"]: row for row in data["measurement_catalogue"]["metrics"]}
    assert by_key["area_px"]["available"] and by_key["area_px"]["group"] == "shape"
    assert not by_key["speed"]["available"]
    assert data["identities"] == [19, 26]
    assert data["cells"][0]["values"]["area_px"] == [20, 20, 20, None, 20, 20, 20, 20, 20, 20, 20, 20]
    assert before == {str(path): path.read_bytes() for path in saved_run.rglob("*") if path.is_file()}
