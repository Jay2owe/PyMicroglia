"""Prepare saved state matrices, bout marks, sample positions and contrast bounds."""
import math
import numpy as np
from pymicroglia.visualisation.panels.behaviour_summaries import finite,cell,label,quantity,bout_marker
from pymicroglia.visualisation.panels.behaviour_profiles import number,wrap,unpack

def matrix(values,probability=True,texts=None):
    maximum=1. if probability or not np.isfinite(values).any() else max(1.,float(np.nanmax(values)))
    labels=texts if texts is not None else [[number(value) if finite(value) else 'NA' for value in row] for row in values]
    return dict(values=np.ma.masked_invalid(values),maximum=maximum,labels=labels,bright=np.isfinite(values)&(values>.65*maximum))

def occupancy(values,settings):
    states=settings['states']
    arrays = [np.full((len(settings['cells']), len(states)), np.nan) for _ in range(2)]
    unknown = np.full((len(settings['cells']), 1), np.nan)
    labels = []
    for index, key in enumerate(settings['cells']):
        source = cell(values, key)
        inventory = source.loc[source.kind.eq('cell')].iloc[0]
        labels.append(label(key) + '\n' + number(inventory.observed_hours) + ' / ' + number(inventory.assigned_hours) + ' h')
        unknown[index, 0] = float(inventory.unknown_fraction_observed) if finite(inventory.unknown_fraction_observed) else np.nan
        for column, state in enumerate(states):
            rows = source.loc[source.kind.eq('occupancy') & source.state_id.eq(state)]
            if len(rows) != 1:
                raise ValueError('Saved occupancy must contain one row per cell and accepted state')
            for array, quantity in zip(arrays, ['fraction_observed', 'fraction_assigned']):
                value = rows.iloc[0][quantity]
                array[index, column] = float(value) if finite(value) else np.nan
    return dict(arrays=[matrix(array) for array in arrays],unknown=matrix(unknown),labels=labels)

def transitions(values,settings):
    shape = (len(settings['cells']), len(settings['pairs']))
    counts, probabilities = (np.full(shape, np.nan), np.full(shape, np.nan))
    texts = [['NA'] * shape[1] for _ in range(shape[0])]
    for index, key in enumerate(settings['cells']):
        source = cell(values, key)
        source = source.loc[source.kind.eq('transition')]
        for column, pair in enumerate(settings['pairs']):
            selected = source.loc[source.source_state_id.eq(pair['source']) & source.target_state_id.eq(pair['target'])]
            if len(selected) > 1:
                raise ValueError('Transition display attempted to mix observation spacings or repeated cell/state pairs')
            if selected.empty:
                continue
            row = selected.iloc[0]
            counts[index, column] = row['count']
            probabilities[index, column] = float(row.probability) if finite(row.probability) else np.nan
            texts[index][column] = number(row['count']) + ' / ' + number(row.source_opportunities)
    return dict(counts=matrix(counts,False,texts),probabilities=matrix(probabilities),shape=shape)

def prepare(values,settings):
    view=settings['view']
    if view=='occupancy':return occupancy(values,settings)
    if view=='transitions':return transitions(values,settings)
    if view in {'switch_rates','bouts'}:
        cells=[]
        for index,key in enumerate(settings['cells']):
            source=cell(values,key)
            if view=='switch_rates':
                rows=source.loc[source.kind.eq('switch_rate')]
                if len(rows)!=1:raise ValueError('Saved switch rates must retain one row per requested cell and spacing')
                cells.append(dict(row=rows.iloc[0],label=label(key)))
            else:
                inventory=source.loc[source.kind.eq('cell')].iloc[0]
                rows=source.loc[source.kind.eq('bout')].sort_values('bout_id')
                if rows.bout_id.duplicated().any():raise ValueError('A saved observed bout appeared more than once')
                marks=[];unavailable=0
                for ordinal,row in enumerate(rows.to_dict('records')):
                    if not finite(row['observed_duration_hours']):unavailable+=1;continue
                    marks.append(dict(x=row['observed_duration_hours'],y=index+(ordinal*.61803398875%1.-.5)*.48,state_id=row['state_id'],marker=bout_marker(row)))
                cells.append(dict(label=label(key)+'\n'+number(inventory.complete_bouts)+' complete; '+number(inventory.incomplete_bouts)+' incomplete',marks=marks,unavailable=unavailable,empty=rows.empty))
        return cells
    if view=='samples':
        questions=[]
        for question in settings['questions']:
            rows=values.loc[values.kind.eq('sample') & values.question_id.eq(question)].sort_values(['condition','sample','unit_id'],na_position='last')
            valid=rows.value.map(finite);conditions=rows.condition.fillna('No condition').to_numpy()
            groups=list(dict.fromkeys(conditions));positions=np.zeros(len(rows));labels=[]
            for group_index,condition in enumerate(groups):
                members=np.flatnonzero(conditions==condition)
                positions[members]=group_index+(np.linspace(-.32,.32,len(members)) if len(members)>1 else 0.)
                labels.append(wrap(condition,25)+'\n(n='+str(len(members))+')')
            row=rows.iloc[0]
            title=quantity(row,settings)+'\n'+str(int(valid.sum()))+'/'+str(len(rows))+' units with values; '+str(int(rows.cells_eligible.min()))+'\u2013'+str(int(rows.cells_eligible.max()))+' eligible cells/unit; '+str(int(rows.recordings_eligible.min()))+'\u2013'+str(int(rows.recordings_eligible.max()))+' recordings/unit'
            questions.append(dict(rows=rows.to_dict('records'),positions=positions,labels=labels,title=title,metric=row.metric))
        return questions
    rows=values.loc[values.kind.eq('contrast')]
    if not len(rows):return []
    lookup=rows.set_index('comparison_id');contrasts=[]
    for comparison in settings['comparisons']:
        row=lookup.loc[comparison];interval=unpack(row.effect_interval);points=[0.];interval_label='';shown=None
        if finite(row.effect):
            points.append(float(row.effect))
            if isinstance(interval,list):
                if len(interval)!=2 or not all(finite(v) for v in interval) or interval[1]<interval[0]:raise ValueError('Invalid saved effect interval')
                declared=settings['sample_provenance'].get('settings',{}).get('interval',{});confidence=declared.get('confidence')
                if declared.get('method')!='independent_bootstrap' or not finite(confidence) or not 0<confidence<1:raise ValueError('Saved effect interval lacks its declared method and confidence level')
                interval_label='; '+number(100*confidence)+'% unadjusted';shown=interval;points.extend(interval)
        if finite(row.descriptive_effect):points.append(float(row.descriptive_effect))
        span=max(points)-min(points);padding=.12*span if span else .1
        contrasts.append(dict(row=row,interval=shown,interval_label=interval_label,bounds=(min(points)-padding,max(points)+padding)))
    return contrasts
