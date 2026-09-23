"""Audit displays use the saved evidence and preserve complete recipe distinctions."""
from dataclasses import replace
import json
from pathlib import Path
import pandas as pd
import pytest
from pymicroglia import workbench as circadian
from pymicroglia.visualisation.figures import get_figure
from pymicroglia.pipelines.audit.workflow import run_request
from pymicroglia.pipelines._screening import read_table
from tests.test_audit_benchmarks import fixture

def display_module():
    from pymicroglia.figure_tables import audit_saved
    return audit_saved

def test_saved_views_are_registered_without_fresh_scientific_options():
    for name in ('audit-performance', 'audit-periods', 'audit-decisions', 'audit-disagreement'):
        spec = get_figure(name)
        assert spec.saved and all((table.scope == 'pipeline' for table in spec.reads))
        assert not set(circadian.CIRCADIAN_ANALYSIS_OPTIONS) & {option.name for option in spec.options}
        assert {'metrics', 'audit_candidates', 'audit_page', 'audit_page_size'} <= {option.name for option in spec.options}

def test_recipe_facets_keep_explicit_test_and_parameter_variants_separate(fixture):
    resolved, _ = fixture
    module = display_module()
    candidate = resolved.candidates[0].as_dict()
    alternative = json.loads(json.dumps(candidate))
    alternative['analysis_options']['significance_method'] = 'f'
    assert module.facet_identity(candidate) != module.facet_identity(alternative)
    alternative = json.loads(json.dumps(candidate))
    alternative['analysis_options']['detrend_lowess_fraction'] = 0.3
    assert module.facet_identity(candidate) != module.facet_identity(alternative)
    assert 'lowess fraction=0.3' in module.facet_label(alternative, [candidate, alternative])
    alternative = json.loads(json.dumps(candidate))
    alternative['analysis_options']['period_config']['example_resolution'] = 123
    assert 'example resolution=123' in module.facet_label(alternative, [candidate, alternative])

def test_pagination_preserves_all_matrix_cells_and_movie_identity():
    module = display_module()
    original = pd.DataFrame([{'panel': 'metric', 'row': f'movie-{movie} / cell {cell}', 'column': f'recipe-{recipe}', 'case': f'{movie}/{cell}/{recipe}'} for movie in (1, 2) for cell in range(9) for recipe in range(5)])
    pages = module.paginate(original, 'disagreement', 7)
    combined = pd.concat(pages, ignore_index=True)
    assert len(pages) == 6 and sorted(combined.case) == sorted(original.case)
    assert combined.case.is_unique and all((page.row.nunique() <= 7 for page in pages))

def test_production_builder_records_saved_counts_before_any_export(fixture, tmp_path, monkeypatch):
    resolved, paths = fixture

    def estimate(hours, values, params, method, **kwargs):
        return {'method': method, 'status': 'ok', 'period_hours': 12.0, 'p_value': 0.001, 'components': [], 'diagnostics': {}}
    monkeypatch.setattr(circadian, 'estimate_one', estimate)
    science = run_request(resolved, paths, tmp_path / 'science', only=('independent-confirmation',))
    assert science.successful
    module = display_module()
    from pymicroglia.pipelines._runner import figure_binding, register_figure_plan
    from pymicroglia.figure_tables.pipeline_inputs import SavedInputs
    alias = {name + '.json': pair for name, pair in module.ALIASES.items()}
    dependencies = {step: science.results[step] for step in {step for step, _ in module.ALIASES.values()}}
    binding = figure_binding(dependencies, inputs=alias)
    item = {'name': 'saved-performance', 'figure': 'audit-performance', 'options': {}, 'pipeline': binding}
    register_figure_plan(tmp_path, [item])
    spec = get_figure('audit-performance')
    ctx = SavedInputs(run=tmp_path, spec=spec, item=item['name'],
                      options={o.name:o.default for o in spec.options}, binding=binding)
    for name in ('estimate_one', 'adjust_pvalues', 'generate_benchmark_cases', 'benchmark_score_interval'):
        monkeypatch.setattr(circadian, name, lambda *a, **k: pytest.fail('renderer recomputed science'))
    data = module.load(ctx)
    shown = module.performance_records(data)
    source = read_table(science.results['development-scores'].artifact('score_summary'))
    expected = source[source.scope.eq('measurement') & source.facet.eq('overall')].iloc[0]
    assert shown.n.iloc[0] == expected.recovery_denominator
    assert shown.numeric_primary.iloc[0] == expected.recovery_rate
    assert shown.numeric_lower.iloc[0] == expected.recovery_lower
    assert module.statistics(data).query("partition == 'real'").q_value.notna().all()
    cards = module.decision_records(data)
    assert cards.display_value.str.contains('Positive support:').all()
    assert cards.display_value.str.contains('Development false alarms:').all()
    missing = source.copy()
    missing.loc[:, 'recovery_valid_fraction'] = 0
    unavailable = module.performance_records({**data, 'development': missing})
    assert unavailable.state.eq('unavailable-tests').all()
    zero = source.copy()
    zero.loc[:, 'recovery_rate'] = 0
    measured_zero = module.performance_records({**data, 'development': zero})
    assert measured_zero.numeric_primary.eq(0).all()
    assert not measured_zero.state.eq('unavailable-tests').any()
    result = module.build(ctx, 'performance')
    assert set(result.figure_data.candidate_id) == {resolved.candidates[0].candidate_id}
    assert result.auxiliary['recipes.csv'].workbench_version.eq(resolved.source.workbench_version).all()
    assert result.drawing is not None
    assert not (tmp_path / 'unsaved').exists()
