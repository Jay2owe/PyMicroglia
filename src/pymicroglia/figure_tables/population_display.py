"""Complete original cell points and saved biological sample summaries."""
import json
import numpy as np
from pymicroglia.visualisation.panels._format import numeric
from pymicroglia.visualisation.panels.relationship_populations import number

def prepare(values,statistics,settings):
    between=settings['view']=='between_cells';delay=settings['view']=='delay'
    groups=settings['groups'];output=[]
    if between:
        cells=values.loc[values.kind.eq('cell_scalar')]
        units=values.loc[values.kind.eq('sample_scalar')]
        for group in groups:
            selected=cells.loc[cells.display_group.eq(group)&cells.eligible.eq(True)]
            output.append(dict(group=group,x=selected.reference_value.to_numpy(),y=selected.target_value.to_numpy()))
        samples=dict(x=units.reference_value.to_numpy(),y=units.target_value.to_numpy())
    else:
        cells=values.loc[values.kind.eq('cell_effect')]
        units=values.loc[values.kind.eq('sample_effect')]
        field='delay_hours' if delay else 'effect'
        unit_field='delay_summary_hours' if delay else 'effect'
        for index,group in enumerate(groups):
            group_cells=cells.loc[cells.display_group.eq(group)]
            eligible=group_cells.loc[group_cells.delay_supported.eq(True)] if delay else group_cells.loc[group_cells.eligible.eq(True)]
            eligible=eligible.loc[numeric(eligible[field],errors='coerce').notna()]
            offsets=eligible[field].copy()
            offsets.iloc[:]=np.linspace(-.22,.22,len(eligible)) if len(eligible)>1 else np.zeros(len(eligible))
            marks=[]
            for sign,marker in (('positive','o'),('negative','s'),(None,'o')) if delay else ((None,'o'),):
                selected=eligible.loc[eligible.association_sign.eq(sign)] if sign else eligible.iloc[:0] if delay else eligible
                marks.append(dict(x=selected[field].to_numpy(),y=(index+offsets.loc[selected.index]).to_numpy(),marker=marker))
            selected=units.loc[units.display_group.eq(group)]
            row=selected.iloc[0] if len(selected) else None
            value=row[unit_field] if row is not None else None
            message=str(row.delay_summary_status if delay else row.status) if row is not None else 'Sample summary unavailable'
            output.append(dict(group=group,marks=marks,sample=value,sample_available=number(value)!='unavailable',message=message))
        samples={}
    note=[]
    if between:
        if len(statistics):
            row = statistics.iloc[0]
            note.append(f'Complete population: {int(row.cells_eligible)}/{int(row.cells_requested)} eligible/requested cells; {int(row.recordings_requested)} recordings; {int(row.confirmed_samples)} confirmed biological samples among eligible cells.')
            note.append(f'Saved inference unit: {row.experimental_unit}. Coefficient: {number(row.effect)}; p={number(row.p_value)}, q={number(row.q_value)}. {row.status}: {row.reason}')
            interval = row.effect_interval
            if isinstance(interval, str):
                try:
                    interval = json.loads(interval)
                except ValueError:
                    interval = None
            if isinstance(interval, (list, tuple)) and len(interval) == 2:
                note.append(f'Saved coefficient interval: {interval[0]:.3g} to {interval[1]:.3g}; {row.interval_status}.')
            else:
                note.append('Coefficient interval: ' + str(row.interval_status) + '.')
        note.append('Cell-summary scatter and independent-sample inference answer a different question from cofluctuation within a cell. No fitted line or new band is added.')
    else:
        all_cells = statistics.loc[statistics.level.eq('all_cells')] if 'level' in statistics else statistics.iloc[:0]
        if len(all_cells):
            row = all_cells.iloc[0]
            note.append(f'Complete population: {int(row.eligible_cells)}/{int(row.requested_cells)} eligible/requested cell effects; {int(row.requested_recordings)} recordings; {int(row.confirmed_samples)} confirmed biological samples.')
            if delay:
                note.append(f'Resolved supported delays: {int(row.delay_supported_cells)}; supported associations with unresolved delay: {int(row.delay_unresolved_cells)}; no detected association: {int(row.not_detected_cells)}; untestable: {int(row.untestable_cells)}; disabled: {int(row.disabled_cells)}.')
                note.append(f'Saved population delay summary: {row.delay_summary_status}. {row.delay_summary_reason}')
            else:
                note.append('Saved effect population: ' + str(row.effect_population) + '. Aggregation: ' + str(row.aggregation) + '.')
        else:
            note.append('Population summaries were not saved; no counts or sample statistics are reconstructed by this figure.')
        across = statistics.loc[statistics.level.eq('across_samples')] if 'level' in statistics else statistics.iloc[:0]
        if not delay and len(across):
            row = across.iloc[0]
            note.append(f'Saved sample-sign evidence: {row.status}; p={number(row.p_value)}, q={number(row.q_value)}. It concerns observed sample-sign frequency, not uncertainty of the average coefficient.')
        if delay:
            note.append('Circles: positive association; squares: inverse association. Incompatible groups are not pooled into a new delay. Unresolved delays are never shown at zero. Negative lag means reference leads target.')
    return dict(groups=output,samples=samples,notes=note)
