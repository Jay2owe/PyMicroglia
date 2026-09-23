"""Correct complete spatial hypotheses before saving report selections."""
from __future__ import annotations
from pymicroglia._results import read_document

import json
from pathlib import Path

import numpy as np

from pymicroglia.pipelines._contracts import ArtifactRef, SelectionRecord, Settings, StepResult, content_id
from pymicroglia.pipelines.coordination.inputs import PAIR_KEYS, _table, read_inputs
from pymicroglia.pipelines.coordination.options import BRANCHES
from pymicroglia.pipelines._runner import Unavailable
from pymicroglia.pipelines._screening import _json_value, _write_json, file_hash, read_table, write_table


EFFECT_KEYS=['effect_id','source_step','source_scientific_id','source_result_id','question','evidence_level',
    'source_run','movie','pair_id','state_pair_id','effect','p_value','q_value','status','reason','decision','decision_reason',
    'is_hypothesis','formal_hypothesis','supported','family_id','family_requested','family_tested','correction','alpha']
FAMILY_KEYS=['family_id','evidence_level','question','scope','correction','alpha','requested','tested','unavailable','members','missing_probability_policy']
TABLES=['effects','families','pair_decisions','recording_decisions','effect_pair_membership','branches']
MISSING_POLICY='Untestable declared hypotheses retain their family slots using one only inside correction; saved p/q remain unavailable'


def implementation():
    import pymicroglia.workbench as circadian
    return {'code':{path.name:file_hash(path) for path in [Path(__file__),Path(circadian.__file__)]},
        'correction_authority':'analysis.circadian.adjust_pvalues','workbench_version':circadian.WORKBENCH_VERSION}


def identity(context):
    return content_id({'inputs':{name:{'scientific_id':saved.outcome.scientific_id,
        'artifacts':[ref.as_dict() for ref in saved.outcome.artifacts]} for name,saved in context.dependencies.items()},
        'inference':context.request.request.inference,'questions':{name:q['enabled'] for name,q in context.request.request.questions.items()},
        'implementation':implementation()})


def collect(context,prepared):
    """Read verified branch results; missing success artifacts are errors."""
    inventory={row['pair_id']:row for row in prepared['inventory'].to_dict('records')}
    effects=[];memberships=[];branches=[];provenances={}
    for step,question in BRANCHES.items():
        q=context.request.request.questions[question];saved=context.dependencies.get(step)
        branch={'step':step,'question':question,'enabled':q['enabled'],'source_scientific_id':saved.outcome.scientific_id if saved else None,
            'execution_status':saved.outcome.status if saved else 'not_required',
            'status':'disabled' if not q['enabled'] else saved.outcome.status if saved else 'unavailable',
            'reason':saved.outcome.reason if saved else 'Disabled scientific branch requires no result artifact'}
        if not q['enabled']:
            branches.append({**branch,'effect_rows':0,'formal_hypotheses':0});continue
        if saved is None or saved.outcome.status not in {'completed','reused'}:raise Unavailable('Required enabled coordination branch did not complete: '+step)
        provenance=read_document(saved.artifact('provenance'))
        if provenance.get('scientific_id')!=saved.outcome.scientific_id or provenance.get('pair_inputs_id')!=context.saved('pair-inputs').outcome.scientific_id:
            raise ValueError('Coordination branch belongs to another scientific input: '+step)
        provenances[step]=provenance
        tables=[('pair_characteristics' if question=='characteristics' else 'pair_effects','pair')]
        if question=='characteristics':tables.append(('recording_effects','recording'))
        state_members={};timing={}
        if question=='rhythm':
            records=read_document(saved.artifact('engine_details'))
            timing={item['result_id']:item for item in records}
            if len(timing)!=len(records) or {item['pair_id'] for item in records}!=set(inventory):raise ValueError('Native timing evidence lost its complete pair population')
        if question=='states':
            mapping=read_table(saved.artifact('pair_membership'));summary=read_table(saved.artifact('pair_summary'))
            if set(mapping.pair_id)!=set(inventory) or mapping.pair_id.duplicated().any():raise ValueError('State comparison lost or repeated a requested pair membership')
            for item in mapping.to_dict('records'):
                original=inventory[item['pair_id']]
                if any(item[key]!=original[key] for key in PAIR_KEYS):raise ValueError('State membership changed its requested endpoints')
                state_members.setdefault(item['state_pair_id'],[]).append(item['pair_id'])
            if set(summary.state_pair_id)!=set(state_members) or summary.state_pair_id.duplicated().any():raise ValueError('State summary lost its complete whole-cell pair population')
            branch['model_status']=provenance['model_status']
            if provenance['model_status']!='accepted':branch.update(status='unaccepted_model',reason='No accepted comparable state vocabulary; core coordination results are retained')
        start=len(effects)
        for name,level in tables:
            frame=read_table(saved.artifact(name));seen=set()
            for raw in frame.to_dict('records'):
                row=_json_value(raw)
                if row.get('question')!=question or row.get('evidence_level',level)!=level:raise ValueError('Branch evidence changed its question or evidence level')
                source_result=row.get('result_id') if level=='pair' else row.get('recording_result_id')
                if not source_result or source_result in seen:raise ValueError('Branch result lacks a unique original scientific identity')
                seen.add(source_result)
                if question=='rhythm':
                    native=timing.get(source_result)
                    if native is None or any(native[key]!=row[key] for key in PAIR_KEYS):raise ValueError('Native rhythm evidence belongs to another exact pair result')
                    row['native_timing_evidence']={key:native['result'].get(key) for key in ['status','reason','period_hours','offset_hours',
                        'offset_interval_hours','phase_definition','sign_convention','period_evidence','temporal_status']}
                if level=='pair':
                    pair_ids=state_members.get(row.get('state_pair_id'),[]) if question=='states' else [row['pair_id']]
                    if not pair_ids:raise ValueError('State effect has no requested whole-cell pair membership')
                    for pair_id in pair_ids:
                        original=inventory.get(pair_id)
                        keys=['source_run','movie','reference_identity','target_identity'] if question=='states' else PAIR_KEYS
                        if original is None or any(row[key]!=original[key] for key in keys):raise ValueError('Branch effect changed its exact endpoint identity')
                else:pair_ids=[]
                method=row.get('method',q['evidence']['method']);hypothesis=bool(row.get('is_hypothesis',True))
                formal=hypothesis and method!='none' and q['evidence']['method']!='none' and not (question in {'characteristics','rhythm'} and level=='pair')
                probability=row.get('p_value')
                if probability is not None:
                    if isinstance(probability,bool) or not isinstance(probability,(int,float)) or not np.isfinite(probability) or not 0<=probability<=1:
                        raise ValueError('Invalid saved coordination probability')
                    if not formal or row['status']!='tested':raise ValueError('Probability is not attached to a tested hypothesis at its own evidence level')
                effect_id=content_id({'source_step':step,'scientific_id':saved.outcome.scientific_id,'source_result_id':source_result,'evidence_level':level})
                effects.append({**row,'effect_id':effect_id,'source_step':step,'source_scientific_id':saved.outcome.scientific_id,
                    'source_result_id':source_result,'evidence_level':level,'is_hypothesis':hypothesis,'formal_hypothesis':formal,
                    'method':method,'effect':row.get('effect',row.get('estimate')),'p_value':probability})
                memberships.extend({'effect_id':effect_id,'pair_id':pair_id,'question':question,'state_pair_id':row.get('state_pair_id'),
                    'meaning':'Whole-cell state evidence; membership is not another measurement-specific test' if question=='states' else 'Exact original measured endpoint pair'} for pair_id in pair_ids)
            if level=='pair' and question!='states' and set(frame.pair_id)!=set(inventory):raise ValueError('Branch omitted requested pairs, including untestable cases')
            if question in {'simultaneous','delay','proximity'}:
                views=read_table(context.saved('simultaneous-coordination').artifact('pair_views'))
                if set(frame.view_id)!=set(views.view_id) or frame.view_id.duplicated().any():raise ValueError('Branch omitted or repeated a declared temporal representation/reference view')
            if level=='recording':
                expected={(sample.movie,pair['reference'],pair['target']) for sample in context.request.inputs.samples for pair in context.request.request.measurement_pairs}
                actual={(item['movie'],item['reference'],item['target']) for item in frame.to_dict('records')}
                if actual!=expected or len(frame)!=len(expected):raise ValueError('Spatial branch omitted or repeated a requested recording hypothesis')
            if question=='states':
                settings=provenance['settings']
                expected={(pair,'state_cooccupancy',a,b) for pair in state_members for a,b in settings.get('resolved_state_pairs',[]) if settings['occupancy']}
                if provenance['model_status']=='accepted' and settings['switching']:
                    expected.update((pair,'switch_process',None,None) for pair in state_members)
                actual={(item['state_pair_id'],item['evidence_kind'],item.get('reference_state_id'),item.get('target_state_id')) for item in _json_value(frame.to_dict('records'))}
                if actual!=expected or len(frame)!=len(expected):raise ValueError('State branch omitted or repeated a declared indicator hypothesis')
        local=effects[start:]
        branches.append({**branch,'effect_rows':len(local),'formal_hypotheses':sum(row['formal_hypothesis'] for row in local)})
    return effects,memberships,branches,provenances


def resolve_evidence(rows,inference):
    """Retain original effects/statuses and put decisions in separate fields."""
    import pymicroglia.workbench as circadian
    rows=[dict(row) for row in rows];groups={};families=[]
    if len({row['effect_id'] for row in rows})!=len(rows):raise ValueError('Repeated scientific effect identity')
    scope=inference.get('correction_scope','all');correction=inference.get('multiple_testing','none');alpha=inference.get('alpha')
    for row in rows:
        row.update(q_value=None,supported=False,family_id=None,family_requested=0,family_tested=0,correction=None,alpha=None)
        if row['formal_hypothesis']:
            if alpha is None:raise ValueError('Formal hypotheses require declared inference settings')
            key=(row['evidence_level'],row['question'] if scope=='question' else 'all')
            groups.setdefault(key,[]).append(row)
    for (level,question),members in sorted(groups.items()):
        members.sort(key=lambda row:row['effect_id']);ids=[row['effect_id'] for row in members]
        family_id=content_id({'level':level,'question':question,'scope':scope,'members':ids,'alpha':alpha,'correction':correction})
        usable=[row['p_value'] is not None for row in members]
        adjusted=circadian.adjust_pvalues([row['p_value'] if valid else 1. for row,valid in zip(members,usable)],correction) if any(usable) else [None]*len(members)
        for row,valid,q in zip(members,usable,adjusted):
            row.update(q_value=float(q) if valid else None,supported=bool(valid and q<=alpha),family_id=family_id,
                family_requested=len(members),family_tested=sum(usable),correction=correction,alpha=alpha)
        families.append({'family_id':family_id,'evidence_level':level,'question':question,'scope':scope,'correction':correction,'alpha':alpha,
            'requested':len(members),'tested':sum(usable),'unavailable':len(members)-sum(usable),'members':ids,'missing_probability_policy':MISSING_POLICY})
    for row in rows:
        if row['supported']:decision,reason='supported','The declared corrected test supports this exact question at its own evidence level'
        elif row['formal_hypothesis'] and row['p_value'] is not None:decision,reason='not_detected','The corrected test did not detect this relationship; this does not establish its absence'
        elif row['question']=='rhythm' and row['status']=='eligible':decision,reason='comparable_timing','Native timing comparability with conditional uncertainty; no pair significance probability is supplied'
        else:decision,reason=row['status'],row['reason']
        row.update(decision=decision,decision_reason=reason)
        if row['question']=='delay':
            estimate=row.get('estimated_delay_hours')
            resolved=bool(row['supported'] and row.get('resolution_status')=='resolved' and isinstance(estimate,(int,float)) and np.isfinite(estimate))
            row.update(delay_supported=resolved,delay_hours=estimate if resolved else None,
                delay_decision='resolved_supported_delay' if resolved else 'association_supported_delay_unresolved' if row['supported'] else 'no_supported_delay',
                delay_uncertainty_meaning='Conditional simultaneous within-curve delay set; not coverage adjusted for selecting pairs')
            candidates=row.get('candidate_lags_hours') or []
            finite=bool(candidates) and all(isinstance(value,(int,float)) and np.isfinite(value) for value in candidates)
            negative=resolved and finite and max(candidates)<0
            positive=resolved and finite and min(candidates)>0
            row.update(delay_direction_supported=bool(negative or positive),
                delay_direction='reference_leads_target' if negative else 'target_leads_reference' if positive else 'unresolved',
                delay_direction_meaning='Temporal ordering only when the complete resolved native candidate region excludes zero; no causal direction')
    return rows,families


def selections(inventory,rows,memberships,scientific_id,inference):
    effects={row['effect_id']:row for row in rows};by_pair={row['pair_id']:[] for row in inventory.to_dict('records')}
    for member in memberships:
        if member['pair_id'] not in by_pair or member['effect_id'] not in effects:raise ValueError('Evidence membership has an unknown pair or effect')
        by_pair[member['pair_id']].append(effects[member['effect_id']])
    pair_rows=[]
    for pair in inventory.to_dict('records'):
        local=by_pair[pair['pair_id']];supported=sorted(row['effect_id'] for row in local if row['supported'])
        timing=sorted(row['effect_id'] for row in local if row['decision']=='comparable_timing')
        pair_rows.append({**pair,'status':'supported' if supported else 'no_supported_pair_evidence','supported_effect_ids':supported,
            'timing_comparable_effect_ids':timing,'all_effect_ids':sorted(row['effect_id'] for row in local),
            'decisions':sorted({row['decision'] for row in local}),'support_questions':sorted({row['question'] for row in local if row['supported']}),
            'meaning':'Only corrected pair-level hypotheses certify a connection; recording tests and trace rhythmicity cannot do so'})
    recordings=[row for row in rows if row['evidence_level']=='recording']
    def pair_member(row):return Settings({**{key:row[key] for key in PAIR_KEYS},'supported_effect_ids':row['supported_effect_ids'],
        'timing_comparable_effect_ids':row['timing_comparable_effect_ids'],'support_questions':row['support_questions']})
    rule={'inference':inference,'full_population':True,'display_limits_applied':False,'missing_probability_policy':MISSING_POLICY,
        'timing_probability':'Native comparable timing and endpoint rhythmicity are separate from tested pair support'}
    selected=(SelectionRecord('supported-pairs',scientific_id,Settings({**rule,'rule':'At least one corrected supported pair-level effect'}),tuple(pair_member(row) for row in pair_rows if row['supported_effect_ids'])),
        SelectionRecord('supported-recordings',scientific_id,Settings({**rule,'rule':'Corrected recording-level test only; never copied to pair edges'}),
            tuple(Settings({key:row[key] for key in ['source_run','movie','effect_id','source_result_id','question','evidence_level']}) for row in recordings if row['supported'])),
        SelectionRecord('supported-effects',scientific_id,Settings({**rule,'rule':'Exact corrected hypothesis identities, with evidence level'}),
            tuple(Settings({key:row[key] for key in ['effect_id','source_result_id','question','evidence_level']}) for row in rows if row['supported'])),
        SelectionRecord('timing-comparable-pairs',scientific_id,Settings({**rule,'rule':'Native comparable rhythm timing; this is not pair significance'}),
            tuple(pair_member(row) for row in pair_rows if row['timing_comparable_effect_ids'])),
        SelectionRecord('resolved-delays',scientific_id,Settings({**rule,'rule':'Corrected complete-search support and resolved native interior delay region'}),
            tuple(Settings({key:row[key] for key in ['effect_id','pair_id','delay_hours','delay_interval_hours','lag_convention']}) for row in rows if row.get('delay_supported',False))))
    return pair_rows,recordings,selected


def produce(context):
    prepared=read_inputs(context.saved('pair-inputs'));rows,memberships,branches,provenances=collect(context,prepared)
    rows,families=resolve_evidence(rows,context.request.request.inference)
    pairs,recordings,chosen=selections(prepared['inventory'],rows,memberships,context.scientific_id,context.request.request.inference)
    outputs={'effects':_table(rows,EFFECT_KEYS),'families':_table(families,FAMILY_KEYS),
        'pair_decisions':_table(pairs,PAIR_KEYS+['status','supported_effect_ids','timing_comparable_effect_ids','all_effect_ids','decisions','support_questions']),
        'recording_decisions':_table(recordings,EFFECT_KEYS),
        'effect_pair_membership':_table(memberships,['effect_id','pair_id','question','state_pair_id','meaning']),
        'branches':_table(branches,['step','question','enabled','source_scientific_id','execution_status','status','reason','effect_rows','formal_hypotheses'])}
    context.output.mkdir(parents=True);refs=[]
    for name,frame in outputs.items():
        path=context.output/(name+'.json');path = write_table(path,frame)
        refs.append(ArtifactRef(name,path.name,file_hash(path),context.scientific_id,columns=tuple(frame.columns)))
    path=context.output/'provenance.json'
    _write_json(path,{'schema_version':1,'scientific_id':context.scientific_id,'pair_inputs_id':context.saved('pair-inputs').outcome.scientific_id,
        'inference':context.request.request.inference,'branch_provenance':provenances,'implementation':implementation(),
        'requested_pairs':len(prepared['inventory']),'effects':len(rows),'families':len(families),'missing_probability_policy':MISSING_POLICY,
        'sample_population':'Complete effects, including non-significant and untestable outcomes; no report-selection filter',
        'scientific_computations':'Declared correction and persisted selection only; no estimation, fitting, time shifts, phase, assignment or lag search',
        'state_membership':'One whole-cell state hypothesis can belong to several requested measurement reports; it is corrected once and never relabelled as measurement-specific evidence',
        'selections':{item.name:item.as_dict() for item in chosen}})
    refs.append(ArtifactRef('provenance',path.name,file_hash(path),context.scientific_id))
    return StepResult(context.step.name,context.scientific_id,'completed','Saved complete effects, distinct evidence-level families and immutable report decisions',tuple(refs),chosen)


def read_evidence(saved,expected_inputs=None):
    provenance=read_document(saved.artifact('provenance'))
    if provenance.get('schema_version')!=1 or provenance.get('scientific_id')!=saved.outcome.scientific_id:raise ValueError('Coordination evidence schema/identity mismatch')
    if expected_inputs is not None and provenance['pair_inputs_id']!=expected_inputs:raise ValueError('Coordination evidence belongs to another pair population')
    tables={name:read_table(saved.artifact(name)) for name in TABLES};effects=tables['effects'];pairs=tables['pair_decisions']
    if effects.effect_id.duplicated().any() or pairs.pair_id.duplicated().any():raise ValueError('Evidence repeats scientific effects or pair identities')
    families=tables['families'];lookup=effects.set_index('effect_id').to_dict('index')
    for family in families.to_dict('records'):
        if len(family['members'])!=family['requested'] or len(set(family['members']))!=len(family['members']):raise ValueError('Saved correction-family membership is inconsistent')
        if any(member not in lookup or lookup[member]['family_id']!=family['family_id'] or lookup[member]['evidence_level']!=family['evidence_level'] for member in family['members']):
            raise ValueError('Saved family changed its evidence level or effect membership')
    for selection in saved.outcome.selections:
        if provenance['selections'].get(selection.name)!=selection.as_dict():raise ValueError('Saved report selection differs from its scientific provenance')
    supported={member['pair_id'] for selection in saved.outcome.selections if selection.name=='supported-pairs' for member in selection.members}
    if supported!=set(pairs.loc[pairs.supported_effect_ids.map(bool),'pair_id']):raise ValueError('Supported pair selection does not match its saved decisions')
    tables['provenance']=provenance
    return tables
