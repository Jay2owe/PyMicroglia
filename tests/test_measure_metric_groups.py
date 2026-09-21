"""Named sets of measured columns: what resolves, and what is refused.

Ported from Motion's ``analysis/test_metric_groups.py``. The point of a
group is that a set of columns is written once and checked once; every test
here is about the checking half, since a group that quietly resolved to
nothing would draw a blank figure and say nothing about why.

Motion resolved groups against the fifteen real modules and inside the
``figures`` block; here they resolve against the stand-ins in
``measure_stubs``, and the figure-block tests return with stage 07.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pymicroglia.measure.metric_groups import (COLUMN_SET_SETTINGS, PREFIX,
                                               SELECTORS, build, group_name,
                                               is_reference, resolve_metrics,
                                               resolve_one, resolve_setting)
from pymicroglia.measure.spec import MeasureConfig

from measure_stubs import stubs  # noqa: F401  - fixture

pytestmark = pytest.mark.usefixtures("stubs")

#: Three real columns, all written by the stand-ins.
SHAPE = ["area_px", "centroid_x", "track_len_px"]


def _config(**blocks) -> MeasureConfig:
    body = {"dataset": "test", "frame_interval_min": 30,
            "movies": [{"stem": "m1", "labels": "labels.tif", "raw": "raw.tif"}]}
    body.update(blocks)
    return MeasureConfig.from_mapping(body)


# ------------------------------------------------------------ the two forms

def test_a_list_group_is_taken_literally_and_in_the_order_written():
    groups = build({"shape": SHAPE})
    assert list(groups["shape"].columns) == SHAPE
    assert groups["shape"].declared == SHAPE


def test_a_selector_group_reads_what_the_modules_declare():
    groups = build({"tracks": {"module": "stub_tracks"}})
    assert "track_len_px" in groups["tracks"].columns
    assert "area_px" not in groups["tracks"].columns


def test_two_selector_keys_are_anded_not_ored():
    both = build({"g": {"module": "stub_area", "role": "morphology"}})["g"].columns
    by_module = build({"g": {"module": "stub_area"}})["g"].columns
    assert set(both) < set(by_module)
    assert set(both) == {"area_px", "centroid_x"}


def test_a_selector_group_comes_out_in_a_stable_alphabetical_order():
    columns = build({"g": {"module": "stub_area"}})["g"].columns
    assert list(columns) == sorted(columns) == ["area_px", "centroid_x", "labelled_px"]
    assert build({"g": {"module": "stub_area"}})["g"].columns == columns


def test_the_declaration_is_kept_beside_what_it_resolved_to():
    group = build({"g": {"module": "stub_area"}})["g"]
    assert group.declared == {"module": "stub_area"}
    assert len(group.columns) > 1


# ------------------------------------------------------------- what is refused

def test_a_list_naming_a_column_no_module_writes_is_refused_by_name():
    with pytest.raises(ValueError, match="area_pxx"):
        build({"shape": ["area_px", "area_pxx"]})


def test_a_near_miss_is_offered_the_column_it_probably_meant():
    with pytest.raises(ValueError, match="did you mean"):
        build({"shape": ["area_pix"]})


def test_a_selector_key_this_package_does_not_offer_is_refused():
    with pytest.raises(ValueError, match=f"selects by {' or '.join(SELECTORS)}"):
        build({"shape": {"grammar": "histogram"}})


def test_a_group_by_table_is_refused_with_the_reason_it_cannot_work():
    with pytest.raises(ValueError, match="may write several tables"):
        build({"shape": {"table": "cell_frame"}})


def test_a_selector_matching_nothing_is_refused_rather_than_returned_empty():
    with pytest.raises(ValueError, match="matches no declared column"):
        build({"shape": {"module": "nonesuch"}})


def test_a_selector_that_selects_nothing_at_all_is_refused():
    with pytest.raises(ValueError, match="every column ever measured"):
        build({"shape": {}})


def test_a_group_may_not_be_built_out_of_another_group():
    with pytest.raises(ValueError, match="One level only"):
        build({"a": ["area_px"], "b": [f"{PREFIX}a"]})


def test_an_empty_group_is_refused():
    with pytest.raises(ValueError, match="is empty"):
        build({"shape": []})


def test_a_group_name_has_to_be_a_plain_lower_case_identifier():
    with pytest.raises(ValueError, match="plain lower-case identifier"):
        build({"Shape": ["area_px"]})


def test_a_block_that_is_not_a_mapping_is_refused():
    with pytest.raises(TypeError, match="block of name"):
        build(["shape"])


def test_a_member_that_is_not_a_column_name_is_refused():
    with pytest.raises(ValueError, match="expected column names"):
        build({"shape": [7]})


# ------------------------------------------------------------- the references

def test_a_reference_is_a_string_that_starts_with_the_prefix():
    assert is_reference("@shape")
    assert not is_reference("shape")
    assert not is_reference("@")
    assert not is_reference(7)
    assert group_name("@shape") == "shape"


def test_a_plain_column_name_resolves_to_itself_in_a_list_of_one():
    assert resolve_one("area_px", {}) == ["area_px"]


def test_a_reference_to_a_group_nobody_declared_lists_the_ones_that_were():
    groups = build({"shape": SHAPE})
    with pytest.raises(ValueError, match="declared groups: shape"):
        resolve_one("@shpae", groups)


def test_a_group_is_expanded_in_place_keeping_the_order_written():
    groups = build({"shape": SHAPE})
    resolved = resolve_metrics(["labelled_px", "@shape", "state_number"], groups, where="test")
    assert list(resolved) == ["labelled_px", *SHAPE, "state_number"]


def test_a_column_in_two_groups_keeps_its_first_position_and_appears_once():
    groups = build({"a": ["area_px", "centroid_x"], "b": ["centroid_x", "track_len_px"]})
    assert list(resolve_metrics(["@a", "@b"], groups, where="test")) == [
        "area_px", "centroid_x", "track_len_px"]


def test_the_setting_a_bad_reference_was_written_in_is_named():
    with pytest.raises(ValueError, match="figures.rhythm-strength.options.metrics"):
        resolve_metrics(["@nope"], {}, where="figures.rhythm-strength.options.metrics")


def test_a_metrics_list_that_is_not_a_list_is_refused():
    with pytest.raises(ValueError, match="expected a list of column names"):
        resolve_metrics("a,b", {}, where="test")


def test_a_value_holding_no_reference_comes_back_exactly_as_written():
    groups = build({"shape": SHAPE})
    assert resolve_setting("a,b", groups, where="test") == "a,b"
    assert resolve_setting(["a", "b"], groups, where="test") == ["a", "b"]
    assert resolve_setting(30, groups, where="test") == 30


def test_a_bare_reference_becomes_the_whole_group():
    groups = build({"shape": SHAPE})
    assert resolve_setting("@shape", groups, where="test") == SHAPE


# ----------------------------------------------------------- the configuration

def test_a_configuration_with_no_metric_groups_block_declares_none():
    assert _config().metric_groups == {}


def test_a_contrast_may_name_a_group():
    config = _config(
        metric_groups={"shape": SHAPE},
        contrasts=[{"name": "by_group", "table": "cell_summary", "metrics": ["@shape"],
                    "group_by": "condition", "groups": ["a", "b"], "unit": "cell",
                    "test": "mannwhitney"}])
    (contrast,) = config.contrasts
    assert list(contrast.metrics) == SHAPE


def test_groups_are_read_before_the_blocks_that_reference_them():
    """JSON key order must not decide whether a reference resolves."""
    written = {
        "metric_groups": {"shape": SHAPE},
        "contrasts": [{"name": "by_group", "table": "cell_summary", "metrics": ["@shape"],
                       "group_by": "condition", "groups": ["a", "b"], "unit": "cell",
                       "test": "mannwhitney"}],
    }
    for order in (("metric_groups", "contrasts"), ("contrasts", "metric_groups")):
        body = {"dataset": "test", "frame_interval_min": 30,
                "movies": [{"stem": "m1", "labels": "l.tif", "raw": "r.tif"}]}
        for key in order:
            body[key] = written[key]
        config = MeasureConfig.from_mapping(body)
        assert list(config.contrasts[0].metrics) == SHAPE


def test_a_group_naming_an_undeclared_column_stops_the_load():
    with pytest.raises(ValueError, match="no module says it writes"):
        _config(metric_groups={"shape": ["area_pxx"]})


def test_the_recorded_settings_keep_the_group_as_declared():
    config = _config(metric_groups={"shape": SHAPE})
    assert config.settings()["metric_groups"] == {"shape": SHAPE}


# ----------------------------------------------------------------- the layering

def test_this_module_never_imports_the_drawing_half():
    """A run that only measures must stay possible without matplotlib."""
    import pymicroglia.measure.metric_groups as metric_groups

    source = Path(metric_groups.__file__).read_text(encoding="utf-8")
    lines = [line for line in source.splitlines() if line.startswith(("import ", "from "))]
    assert not [line for line in lines if "figures" in line]
    assert not [line for line in lines if "matplotlib" in line]


def test_a_column_holding_one_name_is_not_a_place_to_drop_a_whole_group():
    assert "size" not in COLUMN_SET_SETTINGS
