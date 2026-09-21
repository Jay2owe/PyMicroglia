"""Tests for the condition system.

Ported from Motion's ``analysis/test_conditions.py``. Most of these pin a
*refusal*: getting a condition wrong is the one mistake in this package that
produces a complete, plausible, entirely incorrect result, so the interesting
behaviour is not what it assigns but what it declines to.

What changed in the port: a colour is kept as the palette *name* the
configuration gave (or a ``#rrggbb`` literal), and the theme -- stage 07 --
resolves names to values. The tests that went through ``load_theme`` return
with it.
"""

from __future__ import annotations

import json

import pytest

from pymicroglia.measure.conditions import (_AUTO_CYCLE, ConditionSet,
                                            resolve_colour)


def _treatment() -> ConditionSet:
    return ConditionSet.from_config({"HCQ": r"_A\d", "vehicle": r"_B\d"})


def _crossed() -> ConditionSet:
    return ConditionSet.from_config([
        {"name": "HCQ", "label": "+HCQ", "factor": "treatment", "match": r"_A\d"},
        {"name": "vehicle", "label": "Vehicle", "factor": "treatment",
         "match": r"_B\d", "control": True},
        {"name": "young", "factor": "age", "match": r"^3m"},
        {"name": "old", "factor": "age", "match": r"^18m"},
    ])


# ------------------------------------------------------------------ deriving

def test_the_shorthand_form_is_a_name_and_a_regex():
    conditions = _treatment()
    assert conditions.assign("VID95_A3").condition == "HCQ"
    assert conditions.assign("VID95_B1").condition == "vehicle"


def test_a_declared_condition_beats_a_derived_one():
    assignment = _treatment().assign("VID95_A3", declared="vehicle")
    assert assignment.condition == "vehicle"
    assert assignment.source == "declared"


def test_a_declared_condition_must_be_one_that_exists():
    assignment = _treatment().assign("VID95_A3", declared="saline")
    assert not assignment.ok
    assert "not declared" in assignment.problem


def test_matching_is_case_insensitive():
    assert _treatment().assign("vid95_a3").condition == "HCQ"


def test_the_assignment_records_which_pattern_matched():
    assignment = _treatment().assign("VID95_A3")
    assert assignment.evidence == {"condition": r"_A\d"}


# ------------------------------------------------------------------ refusing

def test_two_patterns_matching_one_name_is_an_error_not_a_race():
    assignment = _treatment().assign("VID95_A3_B1")
    assert not assignment.ok
    assert assignment.source == "ambiguous"
    assert "must not overlap" in assignment.problem


def test_a_name_no_pattern_matches_is_unassigned_and_says_so():
    assignment = _treatment().assign("VID95_C2")
    assert not assignment.ok
    assert assignment.condition == "unassigned"
    assert assignment.source == "unassigned"
    assert r"HCQ=/_A\d/" in assignment.problem


def test_no_conditions_declared_is_not_an_error():
    assignment = ConditionSet().assign("VID95_A3")
    assert assignment.ok
    assert assignment.condition == "unassigned"


def test_a_bare_condition_string_survives_with_no_conditions_block():
    assignment = ConditionSet().assign("VID95_A3", declared="control")
    assert assignment.condition == "control"
    assert assignment.source == "declared"


def test_an_invalid_regex_is_a_configuration_error():
    with pytest.raises(ValueError, match="invalid regular expression"):
        ConditionSet.from_config({"broken": "_A(\\d"})


def test_a_name_cannot_be_declared_twice():
    with pytest.raises(ValueError, match="declared twice"):
        ConditionSet.from_config([
            {"name": "HCQ", "match": r"_A\d"},
            {"name": "HCQ", "match": r"_C\d"},
        ])


def test_a_factor_has_at_most_one_control():
    with pytest.raises(ValueError, match="one reference group"):
        ConditionSet.from_config([
            {"name": "a", "match": "a", "control": True},
            {"name": "b", "match": "b", "control": True},
        ])


# ------------------------------------------------------------------- crossed

def test_two_factors_are_resolved_independently_and_combined():
    conditions = _crossed()
    assert conditions.factors() == ("treatment", "age")
    assignment = conditions.assign("3m_VID95_A3")
    assert assignment.condition == "HCQ_young"
    assert assignment.label == "+HCQ / young"
    assert assignment.factors == {"treatment": "HCQ", "age": "young"}


def test_one_unresolved_factor_leaves_the_whole_movie_unassigned():
    assignment = _crossed().assign("VID95_A3")
    assert not assignment.ok
    assert assignment.condition == "unassigned"
    assert "age" in assignment.problem


def test_the_control_is_found_within_its_own_factor():
    assert _crossed().control("treatment").name == "vehicle"


# ------------------------------------------------------------------- colours

def test_an_unnamed_condition_gets_a_colourblind_safe_colour():
    colours = _treatment().colours()
    assert set(colours) == {"HCQ", "vehicle"}
    assert len(set(colours.values())) == 2
    for value in colours.values():
        assert value in _AUTO_CYCLE


def test_auto_colours_follow_declaration_order_not_alphabet():
    first = ConditionSet.from_config({"HCQ": "a", "vehicle": "b"}).colours()
    later = ConditionSet.from_config({"HCQ": "a", "vehicle": "b", "wash": "c"}).colours()
    assert later["HCQ"] == first["HCQ"] and later["vehicle"] == first["vehicle"]


def test_a_named_colour_is_kept_as_the_name_for_the_theme_to_resolve():
    conditions = ConditionSet.from_config({"HCQ": {"match": "a", "colour": "red"}})
    assert conditions.colours()["HCQ"] == "red"


def test_a_literal_colour_is_normalised_when_the_configuration_is_read():
    assert resolve_colour("#FF00FF", "HCQ") == "#ff00ff"
    assert resolve_colour("#abc", "HCQ") == "#abc"


def test_a_nonsense_colour_is_rejected_at_configuration_time():
    with pytest.raises(KeyError, match="neither a"):
        resolve_colour("not a colour!", "HCQ")
    with pytest.raises(KeyError, match="neither a"):
        resolve_colour("#12345", "HCQ")
    with pytest.raises(TypeError):
        resolve_colour(7, "HCQ")


def test_more_conditions_than_safe_colours_refuses_rather_than_repeats():
    many = {f"c{i}": f"_{i}_" for i in range(len(_AUTO_CYCLE) + 3)}
    with pytest.raises(ValueError, match="would look"):
        ConditionSet.from_config(many).colours()


# --------------------------------------------------------------------- audit

def test_the_audit_warns_about_a_design_nothing_can_be_tested_on():
    conditions = _treatment()
    rows = [conditions.assign("VID95_A3")]
    notes = " ".join(conditions.audit(rows))
    assert "one movie" in notes
    assert "at least two groups" in notes


def test_the_audit_warns_when_no_control_is_named():
    conditions = _treatment()
    notes = " ".join(conditions.audit([conditions.assign("VID95_A3")]))
    assert "no reference group" in notes
    assert not any("reference group" in n for n in _crossed().audit([]))


def test_the_design_is_recorded_in_full():
    recorded = _crossed().as_dict()
    assert recorded["factors"] == ["treatment", "age"]
    entry = next(c for c in recorded["conditions"] if c["name"] == "HCQ")
    assert entry["match"] == [r"_A\d"] and entry["colour"] in _AUTO_CYCLE
    assert entry["colour_source"] == "auto"
    assert json.dumps(recorded)


def test_the_recorded_design_round_trips():
    original = _crossed()
    rebuilt = ConditionSet.from_config(original.as_dict()["conditions"])
    assert rebuilt.colours() == original.colours()
    assert rebuilt.assign("18m_VID95_B1").condition == "vehicle_old"


# ----------------------------------------------------------------- synthetic

def _synthetic() -> ConditionSet:
    return ConditionSet.from_config({
        "synthetic": True,
        "conditions": {"treated": r"_A\d", "control": r"_B\d"},
    })


def test_a_synthetic_design_says_so_in_every_place_it_is_recorded():
    conditions = _synthetic()
    assert conditions.synthetic
    assert conditions.as_dict()["synthetic"] is True
    assert any("SYNTHETIC" in note for note in conditions.audit([]))


def test_a_real_design_is_never_marked_synthetic_by_accident():
    assert _treatment().as_dict()["synthetic"] is False
    assert not any("SYNTHETIC" in note for note in _treatment().audit([]))


def test_the_synthetic_flag_survives_the_round_trip():
    rebuilt = ConditionSet.from_config(_synthetic().as_dict())
    assert rebuilt.synthetic is True
    assert rebuilt.assign("demo_A1").condition == "treated"


def test_the_conditions_module_never_imports_the_theme():
    """Colours are names here; resolving them is the drawing half's job."""
    from pathlib import Path

    import pymicroglia.measure.conditions as conditions

    source = Path(conditions.__file__).read_text(encoding="utf-8")
    lines = [line for line in source.splitlines() if line.startswith(("import ", "from "))]
    assert not [line for line in lines if "theme" in line or "matplotlib" in line]
