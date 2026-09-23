"""Measured endpoint geometry and explanatory text for saved spatial edges."""
import pandas as pd
from pymicroglia.visualisation.panels._format import numeric

def prepare(values,settings):
    if not settings['supported_effects']:
        return {'located':pd.DataFrame(),'finite':pd.DataFrame(),'lines':[]}
    nodes = values.loc[values.kind.eq('node')]
    edges = values.loc[values.kind.eq('edge')]
    located = nodes.loc[nodes[['x', 'y']].apply(numeric, errors='coerce').notna().all(axis=1)]
    finite = edges.loc[edges[['x', 'y', 'x2', 'y2']].apply(numeric, errors='coerce').notna().all(axis=1)].copy()
    finite['value'] = numeric(finite.value, errors='coerce')
    finite['arrow'] = finite.get('delay_direction_supported', finite.value.map(lambda _: False)).eq(True)
    reverse = finite.arrow & finite.get('delay_direction', finite.value.map(lambda _: '')).eq('target_leads_reference')
    finite.loc[reverse, ['x', 'y', 'x2', 'y2']] = finite.loc[reverse, ['x2', 'y2', 'x', 'y']].to_numpy()
    definition = settings['definition']
    lines = ['Question: ' + str(definition['question']).replace('_', ' '), 'Statistic: ' + str(definition.get('statistic') or 'saved effect'), 'Representation: ' + str(definition.get('representation') or 'accepted cell states'), 'Adjustment: ' + str(definition.get('adjustment') or 'none'), str(settings['shown_effects']) + ' / ' + str(settings['supported_effects']) + ' supported effects included', str(len(finite)) + ' edges have both plotted positions', str(settings['nodes']) + ' recorded cells; ' + str(settings['unplaced_nodes']) + ' without this position']
    if definition.get('reference'):
        lines.insert(1, 'Measurement roles: ' + definition['reference'] + ' → ' + definition['target'])
    if definition.get('model_id'):
        lines += ['Accepted model: ' + definition['model_id'][:12], 'State-indicator IDs: ' + str(definition.get('reference_state_id') or 'switching')[:12] + ' → ' + str(definition.get('target_state_id') or 'switching')[:12], 'Full model and state identities accompany the figure.']
    roles = ['Reference cell ' + str(int(row.reference_identity)) + ' / target cell ' + str(int(row.target_identity)) for row in edges.itertuples()]
    if len(roles) <= 8:
        lines += ['Displayed endpoint roles:\n' + '\n'.join(roles)]
    else:
        lines += ['Exact reference/target cell roles are listed in the accompanying edge table.']
    return {'located':located,'finite':finite,'lines':lines}
