"""Exact selected endpoints, original observations and saved pair diagnostics."""
from pymicroglia._results import read_document
import json
import pandas as pd

from pymicroglia.pipelines.coordination.inputs import read_inputs
from pymicroglia.pipelines.coordination.evidence import read_evidence
from pymicroglia.pipelines.coordination.display import entry
from pymicroglia.pipelines._screening import _json_value, read_table

CLAIM='Original paired observations, measured distance and saved evidence explain each selected connection without shifting traces or repeating analysis.'
TABLES={'simultaneous-coordination':['series','series_inventory','pair_views','reference_summary','reference_membership','matching_intervals'],
    'lagged-coordination':['profiles'], 'changing-proximity':['windows'],
    'rhythm-coordination':['endpoint_evidence','segments','timecourse'],
    'state-coordination':['pair_summary','joint_exposure','joint_occupancy','switch_events','switch_matches']}


def options(presentation):
    declared=presentation.as_dict().get('pair_cards',{})
    if not isinstance(declared,dict) or set(declared)-{'pair_limit','evidence_rows_per_page','text'}:raise ValueError('Pair-card presentation accepts pair_limit, evidence_rows_per_page and text')
    settings={'pair_limit':None,'evidence_rows_per_page':8,**{key:value for key,value in declared.items() if key!='text'}}
    limit=settings['pair_limit'];size=settings['evidence_rows_per_page']
    if limit is not None and (isinstance(limit,bool) or not isinstance(limit,int) or limit<1):raise ValueError('pair_limit must be null or a positive integer')
    if isinstance(size,bool) or not isinstance(size,int) or not 1<=size<=12:raise ValueError('evidence_rows_per_page must be an integer from one to twelve')
    return settings,declared.get('text',{})


def collect(context):
    prepared=read_inputs(context.saved('pair-inputs'));evidence=read_evidence(context.saved('coordination-evidence'),context.saved('pair-inputs').outcome.scientific_id)
    saved={'prepared':prepared,'evidence':evidence,'tables':{},'provenances':{}}
    for step,names in TABLES.items():
        source=context.dependencies.get(step)
        if source is None or source.outcome.status not in {'completed','reused'}:continue
        saved['provenances'][step]=read_document(source.artifact('provenance'))
        saved['tables'][step]={name:read_table(source.artifact(name)) for name in names}
    return saved


def pages(data,selected,settings):
    prepared,evidence=data['prepared'],data['evidence'];inventory=prepared['inventory'].set_index('pair_id')
    chosen={member['pair_id']:member for member in selected}
    expected=set(evidence['pair_decisions'].loc[evidence['pair_decisions'].supported_effect_ids.map(bool),'pair_id'])
    if set(chosen)!=expected:raise ValueError('Cards must use the full saved pair support selection before applying a display cap')
    order=sorted(chosen);kept=order if settings['pair_limit'] is None else order[:settings['pair_limit']]
    values={};statistics={};result=[]
    def rows(frame,kind,**extra):
        return [entry(**{**{key:value for key,value in record.items() if key not in {'native_details','native_timing_evidence','source_evidence'}},**extra,'kind':kind})
            for record in _json_value(frame.to_dict('records'))]
    def add(view,pair,items,effects,**extra):
        if not items:return
        for row in items:values[row['entry_id']]=row
        for row in effects:statistics[row['entry_id']]=row
        result.append({'view':view,'pair':pair,'title':'Cell-pair report | '+pair['movie']+' | '+str(pair['reference_identity'])+' / '+str(pair['target_identity']),
            'claim':CLAIM,'entry_ids':[row['entry_id'] for row in items], 'effect_entries':[row['entry_id'] for row in effects],
            'footnote':'Original times and values are retained. Gaps are not interpolated, traces are not shifted, and different measurements are shown in their own units. Saved support is a relationship test, not a causal claim.',
            'max_gap_hours':prepared['provenance']['resolved_request']['request']['support']['max_gap_hours'],**extra})
    definitions={item['pair_id']:item['key'] for item in prepared['pair_definitions']}
    for pair_id in kept:
        pair=_json_value(inventory.loc[pair_id].to_dict());pair['pair_id']=pair_id;pair['endpoint_definitions']=definitions[pair_id]
        membership=evidence['effect_pair_membership'];effect_ids=membership.loc[membership.pair_id.eq(pair_id),'effect_id']
        all_effects=evidence['effects'].loc[evidence['effects'].effect_id.isin(effect_ids)]
        if set(chosen[pair_id]['supported_effect_ids'])!=set(all_effects.loc[all_effects.supported,'effect_id']):raise ValueError('A selected card no longer resolves its original supported effects')
        effect_rows=rows(all_effects,'effect',card_pair_id=pair_id);original=[]
        for role in ['reference','target']:
            endpoint=pair[role+'_endpoint_id'];trace=prepared['traces'].loc[prepared['traces'].endpoint_id.eq(endpoint)]
            if not trace.source_run.eq(pair['source_run']).all() or not trace.movie.eq(pair['movie']).all() or not trace.identity.eq(pair[role+'_identity']).all():raise ValueError('Card trace belongs to another original endpoint')
            original+=rows(trace,'raw_trace',card_pair_id=pair_id,endpoint_role=role)
            original+=rows(prepared['scalars'].loc[prepared['scalars'].endpoint_id.eq(endpoint)],'scalar',card_pair_id=pair_id,endpoint_role=role)
        geometry=prepared['geometry'].set_index('observation_id')
        distances=prepared['pair_positions'].loc[prepared['pair_positions'].pair_id.eq(pair_id)].copy()
        for role in ['reference','target']:
            distances[role+'_frame_index']=[geometry.loc[observation,'frame_index'] for observation in distances[role+'_observation']]
        original+=rows(distances,'distance',card_pair_id=pair_id)
        original+=rows(prepared['intervals'].loc[prepared['intervals'].pair_id.eq(pair_id)],'joint_support',card_pair_id=pair_id)
        # The core page always retains both original measurement roles and all
        # saved effect rows, including unsupported optional questions.
        size=settings['evidence_rows_per_page']
        for start in range(0,len(effect_rows),size):
            group=effect_rows[start:start+size];add('core',pair,original+group,group,evidence_part=start//size+1)
        temporal=data['tables'].get('simultaneous-coordination',{})
        if temporal:
            views=temporal['pair_views'].loc[temporal['pair_views'].pair_id.eq(pair_id)]
            for view in _json_value(views.to_dict('records')):
                if view['representation']=='raw' and view['adjustment']=='none':continue
                items=[]
                for role in ['reference','target']:
                    series=temporal['series'].loc[temporal['series'].series_id.eq(view[role+'_series_id'])]
                    items+=rows(series,'processed_trace',card_pair_id=pair_id,endpoint_role=role)
                ref=temporal['reference_summary']
                if len(ref):items+=rows(ref.loc[ref.pair_id.eq(pair_id)&ref.adjusted_series_id.isin([view['reference_series_id'],view['target_series_id']])],'reference',card_pair_id=pair_id)
                support=temporal['matching_intervals']
                items+=rows(support.loc[support.view_id.eq(view['view_id'])],'matched_support',card_pair_id=pair_id)
                applicable=[row for row in effect_rows if row.get('view_id')==view['view_id']]
                add('representation',pair,items+applicable,applicable,temporal_view=view,
                    adjustment_meaning='The saved shared reference is subtracted with its recorded coverage. This can change effect signs; it is not a conditional-independence test.')
        lag=data['tables'].get('lagged-coordination',{}).get('profiles',pd.DataFrame())
        if len(lag):
            for view,group in lag.loc[lag.pair_id.eq(pair_id)].groupby('view_id',sort=True):
                applicable=[row for row in effect_rows if row.get('view_id')==view and row['question']=='delay']
                add('lag',pair,rows(group,'lag_profile',card_pair_id=pair_id)+applicable,applicable,
                    lag_convention='Negative lag: reference leads target. Original per-lag overlap and fixed common-window tested coefficients are different saved populations.')
        proximity=data['tables'].get('changing-proximity',{}).get('windows',pd.DataFrame())
        if len(proximity):
            for view,group in proximity.loc[proximity.pair_id.eq(pair_id)].groupby('view_id',sort=True):
                applicable=[row for row in effect_rows if row.get('view_id')==view and row['question']=='proximity']
                add('proximity',pair,rows(group,'proximity_window',card_pair_id=pair_id)+applicable,applicable)
        rhythm=data['tables'].get('rhythm-coordination',{})
        if rhythm:
            items=[]
            for table,kind in [('endpoint_evidence','rhythm_endpoint'),('segments','rhythm_segment'),('timecourse','rhythm_timepoint')]:
                items+=rows(rhythm[table].loc[rhythm[table].pair_id.eq(pair_id)],kind,card_pair_id=pair_id)
            applicable=[row for row in effect_rows if row['question']=='rhythm']
            add('rhythm',pair,items+applicable,applicable)
        states=data['tables'].get('state-coordination',{})
        if states:
            ids=all_effects.loc[all_effects.question.eq('states'),'state_pair_id'].dropna().unique() if 'state_pair_id' in all_effects else []
            items=[]
            for table,kind in [('pair_summary','state_summary'),('joint_exposure','joint_state_time'),('joint_occupancy','state_occupancy'),('switch_events','switch_event'),('switch_matches','switch_match')]:
                items+=rows(states[table].loc[states[table].state_pair_id.isin(ids)],kind,card_pair_id=pair_id)
            applicable=[row for row in effect_rows if row['question']=='states']
            provenance=data['provenances'].get('state-coordination',{})
            # Original provenance remains a hashed source artifact. Display
            # settings carry model identities, not external workspace paths.
            add('states',pair,items+applicable,applicable,state_provenance={key:provenance.get(key) for key in ['model_id','model_status','scientific_id']})
    return pd.DataFrame(values.values()),pd.DataFrame(statistics.values()),result,{
        'selected_pair_ids':order,'displayed_pair_ids':kept,'omitted_pair_ids':order[len(kept):],
        'selection_changed':False,'display_order':'Full original pair identity; independent of effects and display geometry'}
