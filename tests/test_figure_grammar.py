"""Declared views and option meanings are usable before drawing a page."""
import inspect
import pytest
from pymicroglia.visualisation.figures import load, available_views
from pymicroglia.visualisation.figures._declare import Option


def test_every_view_receives_an_axes_or_figure_and_every_option_has_a_meaning():
    for spec in load().values():
        assert spec.prepare.startswith('pymicroglia.figure_tables.') or spec.prepare.startswith('pymicroglia.pipelines.')
        for view in spec.views:
            assert next(iter(inspect.signature(view.draw).parameters)) in {'ax','figure'}
        assert all(o.description for o in spec.options)


def test_unknown_option_cannot_be_declared():
    with pytest.raises(KeyError,match='Undeclared figure option'):
        Option('invented_option')


def test_clock_views_are_named_and_bad_view_is_refused():
    spec = load()['clock_face']
    assert [v.key for v in spec.views] == ['dial','rose','histogram']
    with pytest.raises(ValueError,match='choose dial, rose, histogram'):
        spec.view('invented')


def test_public_description_lists_each_figures_own_views_and_types():
    from pymicroglia.knowledge import describe
    for name, spec in load().items():
        rows = {row["name"]: row for row in describe(name)["params"]}
        assert rows["view"]["choices"] == [v.key for v in spec.views]
    rows = {r["name"]:r for r in describe("clock_face")["params"]}
    assert rows["metrics"]["type"] == "list"


def test_fresh_fit_description_includes_every_live_estimator():
    from pymicroglia import workbench
    from pymicroglia.knowledge import describe
    rows = {r["name"]:r for r in describe("all_cell_trace_grid")["params"]}
    methods = workbench.call("period_methods").data["methods"]
    assert rows["fit_method"]["choices"] == [r["key"] for r in methods]
    assert rows["significance_method"]["choices"] == [r["key"] for r in methods if r["gives_significance"]]
