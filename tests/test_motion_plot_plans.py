"""Legacy plan expansion retains order, grouping, wording and unique names."""
import pytest
from pymicroglia.figure_tables.plans import expand, parse, problems
from pymicroglia.visualisation.figures import load
from pymicroglia.measure.metric_groups import build


def items(*rows):
    groups = build({"measurements": ["area_px", "corrected_mean"]})
    return expand(parse(list(rows), groups), {s.slug: s for s in load().values()})


def test_group_expansion_and_one_page_group_are_distinct():
    many = items({"figure": "rhythm-strength", "for_each": {"metrics": "@measurements"}})
    one = items({"figure": "rhythm-strength", "options": {"metrics": "@measurements"}})
    assert [x.options["metrics"] for x in many] == [["area_px"], ["corrected_mean"]]
    assert one[0].options["metrics"] == ["area_px", "corrected_mean"]


def test_cross_product_order_and_scalar_identity():
    rows = items({"figure": "cell-report-card", "for_each": {"stem": ["a", "b"], "identity": [1, 2]},
                  "as": "{stem}/{identity}", "title": "Synthetic {identity}"})
    assert [x.name for x in rows] == ["a/1", "a/2", "b/1", "b/2"]
    assert [x.options["identity"] for x in rows] == [1, 2, 1, 2]
    assert all(x.text == {"title": "Synthetic {identity}"} for x in rows)


def test_duplicate_output_names_and_unknown_options_refused():
    with pytest.raises(ValueError, match="both draw"):
        items({"figure": "rhythm-strength"}, {"figure": "rhythm-strength"})
    with pytest.raises(ValueError, match="not a setting"):
        items({"figure": "rhythm-strength", "options": {"invented": True}})


def test_each_named_view_is_one_plan_item():
    rows = items({"figure": "clock-face", "for_each": {"view": ["dial", "rose"]}})
    assert [x.options["view"] for x in rows] == ["dial", "rose"]
    assert len({x.name for x in rows}) == 2
