import itertools

import pytest

from pymicroglia.cell_filters import evaluate, validate


def rule(field="has_merge", value=True, cmp="eq", **extra):
    return {"field": field, "cmp": cmp, "value": value, **extra}


@pytest.mark.parametrize("a,b", itertools.product([True, False, None], repeat=2))
def test_three_valued_groups_and_not(a, b):
    cells = {19: {"has_merge": a, "has_split": b}}
    children = [rule(), rule("has_split")]
    expected = {"AND": False if False in (a, b) else None if None in (a, b) else True,
                "OR": True if True in (a, b) else None if None in (a, b) else False,
                "XOR": None if None in (a, b) else a != b}
    for op in expected:
        expression = {"op": op, "children": children}
        assert evaluate(cells, expression)["rows"][0]["match"] is expected[op]
        negated = {"op": "NOT", "children": [expression]}
        answer = None if expected[op] is None else not expected[op]
        assert evaluate(cells, negated)["rows"][0]["match"] is answer


def test_unknown_policy_applies_after_logic_without_renumbering():
    cells = {19: {"has_merge": True}, 26: {"has_merge": False}, 105: {"has_merge": None}}
    expression = {"op": "NOT", "children": [rule()]}
    assert evaluate(cells, expression, mode="keep")["identities"] == [26, 105]
    assert evaluate(cells, expression, mode="exclude")["identities"] == [19, 105]
    assert evaluate(cells, expression, mode="keep", unknown="exclude")["identities"] == [26]
    with pytest.raises(ValueError, match="Cell 105"):
        evaluate(cells, expression, unknown="error")


def test_threshold_edges_and_no_neighbour():
    cells = {19: {"longest_gap_frames": 3, "min_distance_px": float("inf")}}
    assert evaluate(cells, rule("longest_gap_frames", 3, "gt"), mode="keep")["identities"] == []
    assert evaluate(cells, rule("longest_gap_frames", 3, "gte"), mode="keep")["identities"] == [19]
    assert evaluate(cells, rule("longest_gap_frames", [3, 5], "between"), mode="keep")["identities"] == [19]
    row = evaluate(cells, rule("min_distance_px", 10, "lt"))["rows"][0]
    assert not row["match"] and row["evidence"][0]["no_neighbour"]


@pytest.mark.parametrize("bad", [
    {"op": "NOT", "children": []}, {"op": "NAND", "children": []},
    rule("bogus"), rule("has_merge", 1), rule("missing_pct", float("nan")),
    rule("missing_pct", [20, 10], "between"), rule("rhythmic"),
    {"op": "AND", "children": [{"op": "OR", "children": []}]},
    {**rule(), "unknown_parameter": 1},
])
def test_malformed_filters_are_refused(bad):
    with pytest.raises(ValueError): validate(bad)


def test_validation_bounds_and_empty_disabled_expression():
    expression = rule()
    for _ in range(14): expression = {"op": "NOT", "children": [expression]}
    with pytest.raises(ValueError, match="nested"): validate(expression)
    assert evaluate({26: {}}, {"op": "OR", "children": []})["identities"] == [26]
    with pytest.raises(ValueError, match="measurement is not available"):
        validate(rule("rhythmic", metric="absent"), metrics={"signal_mean"})
