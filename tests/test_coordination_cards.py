"""Controlled saved display fixtures verify source identity, not statistical calibration."""
from tests.panel_helpers import panel_canvas
from pymicroglia.figure_tables.pair_card_display import prepare as prepare_display, segments
from copy import deepcopy
import pandas as pd
import pytest
import pymicroglia.pipelines.coordination.card_data as cards
from pymicroglia.pipelines._contracts import Settings
from tests.test_coordination_characteristics import fixture as input_fixture

def fixture(resolved=None, prepared=None):
    if prepared is None:
        resolved, _, prepared = input_fixture(n=3)
    prepared['provenance']['resolved_request'] = resolved.as_dict()
    effects = []
    members = []
    decisions = []
    selected = []
    for index, pair in enumerate(prepared['inventory'].to_dict('records')):
        eid = 'controlled-saved-effect-' + str(index)
        supported = index < 2
        effects.append({**pair, 'effect_id': eid, 'question': 'simultaneous', 'evidence_level': 'pair', 'effect': 0.4, 'p_value': 0.01 if supported else 0.9, 'q_value': 0.03 if supported else 0.9, 'supported': supported, 'decision': 'supported' if supported else 'not_detected', 'reason': 'Controlled frozen renderer input', 'source_scientific_id': 'saved-source', 'source_result_id': 'original-' + str(index)})
        members.append({'pair_id': pair['pair_id'], 'effect_id': eid})
        decisions.append({'pair_id': pair['pair_id'], 'supported_effect_ids': [eid] if supported else []})
        if supported:
            selected.append({'pair_id': pair['pair_id'], 'supported_effect_ids': [eid]})
    return ({'prepared': prepared, 'evidence': {'effects': pd.DataFrame(effects), 'effect_pair_membership': pd.DataFrame(members), 'pair_decisions': pd.DataFrame(decisions)}, 'tables': {}, 'provenances': {}}, selected)

def test_full_original_endpoints_times_and_actual_distance_survive_card_pagination():
    saved, selected = fixture()
    before = deepcopy(selected)
    settings, _ = cards.options(Settings({'pair_cards': {'pair_limit': 1}}))
    values, statistics, pages, selection = cards.pages(saved, selected, settings)
    assert len(pages) == 1 and selected == before and (len(selection['omitted_pair_ids']) == 1)
    page = pages[0]
    assert page['view'] == 'core'
    for role in ['reference', 'target']:
        frame = values.loc[values.kind.eq('raw_trace') & values.endpoint_role.eq(role)]
        assert frame.endpoint_id.eq(page['pair'][role + '_endpoint_id']).all() and frame.identity.eq(page['pair'][role + '_identity']).all()
        assert frame.hours.tolist() == [50.0 + 0.5 * i for i in range(8)]
    distances = values.loc[values.kind.eq('distance')]
    assert len(distances) == 8 and distances.reference_frame_index.tolist() == list(range(8))
    assert set(statistics.effect_id) == set((selected_item['supported_effect_ids'][0] for selected_item in selected if selected_item['pair_id'] == page['pair']['pair_id']))
    from pymicroglia.visualisation.figures import load as _figure_schema
    _figure_schema()
    from pymicroglia.visualisation.panels import coordination_cards
    import matplotlib.pyplot as plt
    fig, _ = coordination_cards.draw(prepare_display(values, page), page, canvas=panel_canvas())
    plt.close(fig)

def test_an_observation_from_another_source_cannot_enter_a_plausible_card():
    saved, selected = fixture()
    endpoint = saved['prepared']['inventory'].iloc[0].reference_endpoint_id
    saved['prepared']['traces'].loc[saved['prepared']['traces'].endpoint_id.eq(endpoint), 'source_run'] = 'wrong-source'
    settings, _ = cards.options(Settings())
    with pytest.raises(ValueError, match='another original endpoint'):
        cards.pages(saved, selected, settings)

def test_plotting_preserves_frame_gaps_invalid_values_and_original_time():
    from pymicroglia.visualisation.figures import load as _figure_schema
    _figure_schema()
    from pymicroglia.visualisation.panels.coordination_cards import observed_line
    import matplotlib.pyplot as plt
    frame = pd.DataFrame({'hours': [50.0, 50.5, 51.0, 51.5, 52.0, 52.5], 'frame_index': [0, 1, 3, 4, 5, 6], 'raw_value': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 'raw_valid': [True, True, True, True, False, True]})
    fig, axis = plt.subplots()
    count = observed_line(axis, segments(frame, 'hours', 'raw_value', max_gap=1.0, frame_columns=['frame_index'], valid_columns=['raw_valid']), colour='black')
    assert count == 5 and [line.get_xdata().tolist() for line in axis.lines] == [[50.0, 50.5], [51.0, 51.5], [52.5]]
    plt.close(fig)

def test_cross_measurement_roles_repeated_cell_ids_and_missing_positions_remain_explicit():
    from pymicroglia.pipelines.coordination.inputs import prepare
    from tests.test_coordination_options import request, resolve
    _, tables, _ = input_fixture(n=2)
    frame = tables['cell_frame'].copy()
    frame['custom_shape'] = 100 - frame.custom_signal
    frame['centroid_x'] = float('nan')
    frame['centroid_y'] = float('nan')
    frame = pd.concat([frame, frame.assign(stem='another', custom_signal=frame.custom_signal + 30)], ignore_index=True)
    tables = {'cell_frame': frame, 'cell_summary': frame[['stem', 'identity']].drop_duplicates()}
    resolved = resolve(tables, request(target_measurements=['custom_shape']))
    prepared, definitions = prepare(resolved, tables)
    prepared.update(pair_definitions=definitions, provenance={})
    saved, selected = fixture(resolved, prepared)
    for row in saved['evidence']['pair_decisions'].to_dict('records'):
        if row['supported_effect_ids']:
            continue
        member = saved['evidence']['effect_pair_membership'].loc[saved['evidence']['effect_pair_membership'].pair_id.eq(row['pair_id'])].iloc[0]
        saved['evidence']['effects'].loc[saved['evidence']['effects'].effect_id.eq(member.effect_id), 'supported'] = True
        saved['evidence']['pair_decisions'].loc[saved['evidence']['pair_decisions'].pair_id.eq(row['pair_id']), 'supported_effect_ids'] = pd.Series([[member.effect_id]], index=saved['evidence']['pair_decisions'].index[saved['evidence']['pair_decisions'].pair_id.eq(row['pair_id'])])
        selected.append({'pair_id': row['pair_id'], 'supported_effect_ids': [member.effect_id]})
    settings, _ = cards.options(Settings())
    values, statistics, pages, selection = cards.pages(saved, selected, settings)
    assert len(pages) == 4 and len(selection['displayed_pair_ids']) == 4 and (not values.kind.eq('distance').any())
    for page in pages:
        pair = page['pair']
        assert pair['reference'] == 'custom_signal' and pair['target'] == 'custom_shape' and pair['oriented']
        original = values.loc[values.entry_id.isin(page['entry_ids']) & values.kind.eq('raw_trace')]
        assert original.movie.eq(pair['movie']).all()
        for role in ['reference', 'target']:
            rows = original.loc[original.endpoint_role.eq(role)]
            assert rows.endpoint_id.eq(pair[role + '_endpoint_id']).all()
            assert rows.hours.tolist() == [50.0 + 0.5 * i for i in range(8)]
    from pymicroglia.visualisation.figures import load as _figure_schema
    _figure_schema()
    from pymicroglia.visualisation.panels import coordination_cards
    import matplotlib.pyplot as plt
    page = pages[0]
    fig, axes = coordination_cards.draw(prepare_display(values.loc[values.entry_id.isin(page['entry_ids'])], page), page, canvas=panel_canvas())
    assert any(('No matched observed positions' in text.get_text() for text in axes['2'].texts))
    plt.close(fig)
