"""A displayed spatial connection must resolve to its actual saved pair evidence."""
from tests.panel_helpers import panel_canvas
from copy import deepcopy
import pandas as pd
import pytest
import pymicroglia.pipelines.coordination.map_figures as maps
from pymicroglia.pipelines._contracts import Settings

def fixture():
    layouts = []
    geometry = []
    effects = []
    pairs = []
    membership = []
    selected = []
    for movie, origin in [('a', 0.0), ('b', 100.0)]:
        for identity, x in [(7, origin + 2), (8, origin + 6), (9, origin + 9)]:
            layouts.append({'source_run': 'source', 'movie': movie, 'identity': identity, 'x': x, 'y': 3.0, 'valid_positions': 2, 'unit': 'um', 'definition': 'centroid', 'position_meaning': 'Saved coordinate-wise median'})
            for index, time in enumerate([50.0, 51.0]):
                geometry.append({'source_run': 'source', 'movie': movie, 'identity': identity, 'hours': time, 'x': x - 1 + 2 * index, 'y': 3.0, 'valid': True, 'observation_id': movie + str(identity) + str(index)})
        for target in [8, 9]:
            pair = movie + str(target)
            effect = 'effect-' + pair
            effects.append({'source_run': 'source', 'movie': movie, 'effect_id': effect, 'pair_id': pair, 'reference_identity': 7, 'target_identity': target, 'reference': 'movement', 'target': 'shape', 'question': 'simultaneous', 'evidence_level': 'pair', 'representation': 'raw', 'adjustment': 'unadjusted', 'statistic': 'pearson', 'effect': -0.5 if target == 8 else 0.7, 'p_value': 0.01, 'q_value': 0.04, 'supported': True, 'status': 'tested', 'decision': 'supported', 'source_scientific_id': 'result', 'family_id': 'family', 'correction': 'bonferroni', 'alpha': 0.05})
            pairs.append({'pair_id': pair, 'supported_effect_ids': [effect]})
            membership.append({'effect_id': effect, 'pair_id': pair})
            selected.append({'pair_id': pair, 'supported_effect_ids': [effect]})
    prepared = {'layouts': pd.DataFrame(layouts), 'geometry': pd.DataFrame(geometry), 'provenance': {'resolved_request': {'request': {'geometry': {'unit': 'um', 'definition': 'centroid'}}}}}
    evidence = {'effects': pd.DataFrame(effects), 'pair_decisions': pd.DataFrame(pairs), 'effect_pair_membership': pd.DataFrame(membership)}
    return (prepared, evidence, selected)

def test_frames_endpoints_and_deterministic_display_cap_preserve_full_support():
    prepared, evidence, selected = fixture()
    unchanged = deepcopy(selected)
    settings, _ = maps.options(Settings({'maps': {'edge_limit': 1}}))
    values, statistics, pages = maps.data(prepared, evidence, selected, settings)
    assert len(pages) == 2 and values.kind.eq('edge').sum() == 2 and (len(statistics) == 2)
    assert selected == unchanged and all((page['supported_effects'] == 2 and len(page['omitted_effect_ids']) == 1 for page in pages))
    for page in pages:
        rows = values.loc[values.entry_id.isin(page['entry_ids'])]
        assert rows.movie.nunique() == 1
        edge = rows.loc[rows.kind.eq('edge')].iloc[0]
        assert edge.reference == 'movement' and edge.target == 'shape' and (edge.value == -0.5)
        assert edge.x2 - edge.x == 4.0 and edge.requested_pair_ids == [edge.pair_id]
        assert (edge.x > 100) == (page['definition']['movie'] == 'b')

def test_snapshot_requires_actual_measurements_without_nearest_time_substitution():
    prepared, evidence, selected = fixture()
    settings, _ = maps.options(Settings({'maps': {'position': 'snapshot', 'snapshot_hours': 50.0}}))
    values, _, pages = maps.data(prepared, evidence, selected, settings)
    edge = values.loc[values.kind.eq('edge') & values.movie.eq('a')].iloc[0]
    assert edge.x == 1.0 and edge.reference_position_observation_id == 'a70'
    settings['snapshot_hours'] = 50.5
    values, _, pages = maps.data(prepared, evidence, selected, settings)
    assert values.loc[values.kind.eq('edge'), 'drawable'].eq(False).all()
    assert values.loc[values.kind.eq('node'), 'x'].isna().all() and all((page['unplaced_nodes'] == 3 for page in pages))

def test_recording_probability_cannot_supply_supported_pair_lines():
    prepared, evidence, selected = fixture()
    evidence['effects']['evidence_level'] = 'recording'
    settings, _ = maps.options(Settings())
    with pytest.raises(ValueError, match='pair-level'):
        maps.data(prepared, evidence, selected, settings)
    evidence['pair_decisions']['supported_effect_ids'] = [[] for _ in selected]
    values, _, pages = maps.data(prepared, evidence, [], settings)
    assert values.kind.eq('diagnostic').all() and pages[0]['supported_effects'] == 0
    with pytest.raises(ValueError, match='selection'):
        maps.data(prepared, evidence, selected, settings)

def test_saved_direction_is_drawn_only_when_native_resolved_region_excludes_zero():
    from pymicroglia.visualisation.figures import load as _figure_schema
    _figure_schema()
    from pymicroglia.visualisation.panels import coordination_maps
    import matplotlib.pyplot as plt
    prepared, evidence, selected = fixture()
    evidence['effects']['question'] = 'delay'
    evidence['effects']['delay_direction_supported'] = [True, False, True, False]
    evidence['effects']['delay_direction'] = ['reference_leads_target', 'unresolved', 'target_leads_reference', 'unresolved']
    settings, _ = maps.options(Settings())
    values, statistics, pages = maps.data(prepared, evidence, selected, settings)
    for page in pages:
        rows = values.loc[values.entry_id.isin(page['entry_ids'])]
        from pymicroglia.figure_tables.connection_display import prepare
        fig, axes = coordination_maps.draw(prepare(rows,page), page, canvas=panel_canvas())
        from matplotlib.patches import ArrowStyle
        arrows = [mark for mark in axes['map'].patches if isinstance(mark.get_arrowstyle(), ArrowStyle.CurveB)]
        assert len(arrows) == 1
        row = rows.loc[rows.kind.eq('edge') & rows.delay_direction_supported.eq(True)].iloc[0]
        assert arrows[0]._posA_posB[1] == ((row.x2, row.y2) if row.delay_direction == 'reference_leads_target' else (row.x, row.y))
        plt.close(fig)
