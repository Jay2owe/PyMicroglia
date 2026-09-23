"""Exact state members, observed intervals, trace links and image layers."""
import numpy as np
from pymicroglia.visualisation.panels.behaviour_cards import finite

def feature_rows(frame,feature):
    return frame.loc[frame.measurement.eq(feature['measurement']) & frame.source_table.eq(feature['source_table']) & frame.representation.eq(feature['representation'])]

def trace(frame,feature,example):
    source=frame.loc[frame.kind.eq('trace') & frame.measurement.eq(feature['measurement'])].sort_values('source_position')
    if source.frame_index.duplicated().any():raise ValueError('State example trace repeats an original feature frame')
    field='raw_value' if feature['representation']=='raw' else 'processed_value'
    valid='raw_valid' if feature['representation']=='raw' else 'processed_valid'
    shown=source.loc[source.hours.map(finite) & source[field].map(finite) & source[valid].eq(True) & source.within_range.eq(True)]
    observations=frame.loc[frame.kind.eq('observation')].set_index('observation_id')
    lookup=shown.set_index('frame_index');segments=[]
    for row in frame.loc[frame.kind.eq('step') & frame.valid_observed_interval.eq(True)].to_dict('records'):
        endpoints=[]
        for name in ('source_observation_id','target_observation_id'):
            if row[name] not in observations.index:raise ValueError('Saved time step lost an original observation')
            observation=observations.loc[row[name]]
            if observation.frame_index not in lookup.index:break
            point=lookup.loc[observation.frame_index]
            if point.hours!=observation.hours:break
            endpoints.append((point.hours,point[field]))
        if len(endpoints)==2:segments.append(([point[0] for point in endpoints],[point[1] for point in endpoints]))
    selected=shown.loc[shown.frame_index.eq(example['frame_index']) & shown.hours.eq(example['hours'])]
    return dict(x=shown.hours.to_numpy(),y=shown[field].to_numpy(),segments=segments,selected=(selected.hours.to_numpy(),selected[field].to_numpy()))

def prepare(values,settings,archive):
    result=dict(state=values.loc[values.kind.eq('state')].iloc[0],profiles=[],population=[],examples={})
    for feature in settings['features']:
        selected=feature_rows(values.loc[values.kind.eq('profile')],feature)
        if len(selected)!=1:raise ValueError('State profile does not resolve to exactly one saved measurement/representation')
        result['profiles'].append(selected.iloc[0])
    occupancy=values.loc[values.kind.eq('population_occupancy')].sort_values(['source_run','movie','identity'])
    cells=values.loc[values.kind.eq('population_cell')].sort_values(['source_run','movie','identity'])
    for index,(frame,column,label) in enumerate([(occupancy,'fraction_observed','State / observed time'),(occupancy,'fraction_assigned','State / assigned time'),(cells,'unknown_fraction_observed','Unknown / observed time')]):
        rows=frame.loc[frame[column].map(finite)]
        offsets=(np.arange(len(rows))*.61803398875%1.-.5)*.22
        result['population'].append(dict(x=rows[column].to_numpy(),y=index+offsets,label=label+'\n'+str(len(rows))+'/'+str(len(frame))+' cells with values'))
    examples=values.loc[values.kind.eq('example')].set_index('example_id')
    for example_id in settings['example_ids']:
        example=examples.loc[example_id].to_dict();example['example_id']=example_id
        frame=values.loc[values.example_id.eq(example_id)]
        cells=frame.loc[frame.kind.eq('cell')]
        if len(cells)!=1:raise ValueError('State example lacks a unique full cell time inventory')
        inventory=cells.iloc[0]
        intervals=frame.loc[frame.kind.eq('interval')].to_dict('records')
        for row in intervals:row['width']=row['end_hours']-row['start_hours']
        observations=frame.loc[frame.kind.eq('observation') & frame.hours.map(finite) & frame.within_range.eq(True) & frame.time_status.eq('recorded')].to_dict('records')
        bounds=(inventory.reference_start_hours,inventory.reference_end_hours) if finite(inventory.reference_start_hours) and finite(inventory.reference_end_hours) and inventory.reference_end_hours>inventory.reference_start_hours else (float(example['hours'])-.5,float(example['hours'])+.5)
        image=[row for row in settings['images'] if row['example_id']==example_id]
        if len(image)!=1:raise ValueError('State example has no unique image availability record')
        image=dict(image[0])
        if image['image_status']=='available':
            tile=image['tiles'][0]
            if tile['frame_index']!=example['frame_index'] or tile['hours']!=example['hours'] or not tile['cell_present']:raise ValueError('Saved source image does not match the exact selected state observation')
            image.update(pixels=archive[image['archive_key']+'_display'][0],mask=archive[image['archive_key']+'_mask'][0].astype(float))
        result['examples'][example_id]=dict(example=example,inventory=inventory,intervals=intervals,observations=observations,bounds=bounds,traces=[trace(frame,feature,example) for feature in settings['features']],image=image)
    return result
