"""Paired original-cell changes and recurrence at the biological-sample level."""
from pymicroglia._results import read_document
from pymicroglia._sources import source_file
from pathlib import Path
from importlib.metadata import version
from collections import Counter
import json
import numpy as np
import pymicroglia.workbench as circadian
from pymicroglia.pipelines._contracts import ArtifactRef, StepResult, content_id
from pymicroglia.pipelines.rhythm.discovery import _pairs
from pymicroglia.pipelines.coordination.inputs import _table
from pymicroglia.pipelines.intervention.windows import finite, read_windows
from pymicroglia.pipelines.intervention.evidence import MISSING_POLICY, read_evidence
from pymicroglia.pipelines.intervention.control_inputs import unit_id, clock_key, window_key
import pymicroglia.pipelines.intervention.pattern_statistics as native
from pymicroglia.pipelines._screening import _json_value, _write_json, file_hash, read_table, write_table

TABLES=['cell_pairs','recording_pairs','sample_pairs','definitions','associations','families','native_details']


def policy(block,measurements,inference):
    if not block.get('enabled',False):return {'enabled':False}
    pairs=[pair.as_dict() for pair in _pairs(block.get('pairs'),tuple(measurements))]
    if not pairs:raise ValueError('Coordinated responses require at least one distinct measurement pair')
    aggregation=block.get('aggregation')
    if aggregation not in {'mean','median'}:raise ValueError('Declare mean or median aggregation for paired cell changes')
    quantity=block.get('quantity','absolute_change')
    if quantity not in {'absolute_change','estimate','relative_change'}:raise ValueError('Choose absolute_change, estimate or relative_change for paired effects')
    evidence=native.settings(block.get('evidence',{'method':'none'}))
    if evidence['method']!='none' and not inference:raise ValueError('Coordinated sample association requires declared inference settings')
    return {'enabled':True,'pairs':pairs,'aggregation':aggregation,'quantity':quantity,'evidence':evidence}


def identity(context):
    request=context.request.request;chosen=policy(request.coordinated.as_dict(),[m.column for m in context.request.measurements],request.inference.as_dict())
    return content_id({'windows':context.saved('aligned-windows').outcome.scientific_id,
        'evidence':context.saved('response-evidence').outcome.scientific_id,'policy':chosen,'inference':request.inference.as_dict(),
        'libraries':{name:version(name) for name in ['numpy','pandas','scipy']},'workbench':circadian.WORKBENCH_VERSION,
        'code':{name:file_hash(source_file(name)) for name in ['intervention_patterns.py','intervention_pattern_statistics.py','intervention_control_inputs.py']},
        'correction_gateway':file_hash(Path(circadian.__file__))})


def category(a,b):
    if not finite(a['p_value']) or not finite(b['p_value']):
        return 'both_inconclusive' if not finite(a['p_value']) and not finite(b['p_value']) else 'reference_inconclusive' if not finite(a['p_value']) else 'target_inconclusive'
    first,second=a['response_supported'],b['response_supported']
    if first and second:return 'both_supported_same_direction' if a['response_direction']==b['response_direction'] else 'both_supported_opposite_directions'
    if first:return 'reference_supported_only'
    if second:return 'target_supported_only'
    return 'neither_has_a_detected_response'


def direction(a,b):
    if not finite(a) or not finite(b):return 'unavailable'
    if a==0 or b==0:return 'at_least_one_zero_estimate'
    return 'same_signed_changes' if np.sign(a)==np.sign(b) else 'opposite_signed_changes'


def counts(rows):
    requested=len(rows);paired=sum(row['paired_available'] for row in rows);tested=sum(row['jointly_tested'] for row in rows)
    categories=dict(Counter(row['joint_response'] for row in rows));both=sum(categories.get(key,0) for key in ['both_supported_same_direction','both_supported_opposite_directions'])
    return {'requested_cells':requested,'paired_available_cells':paired,'jointly_tested_cells':tested,'both_supported_cells':both,
        'both_supported_fraction_of_jointly_tested':both/tested if tested else None,'joint_response_counts':categories,
        'direction_counts':dict(Counter(row['joint_direction'] for row in rows)),
        'fraction_meaning':'Descriptive original-cell count / explicitly jointly tested count; two absent detections do not establish coordination'}


def analyse(prepared,evidence,resolved,chosen):
    original={row['window_id']:row for row in _json_value(prepared['windows'].to_dict('records'))}
    effects=_json_value(evidence['effects'].to_dict('records'));lookup={tuple(row[key] for key in ['movie','identity','measurement','baseline','target_window']):row for row in effects}
    if len(lookup)!=len(effects):raise ValueError('Repeated original intervention effect')
    designs=resolved.recordings.as_dict();clocks={movie:clock_key(prepared,movie,item['anchor']['hours']) for movie,item in designs.items()}
    cells=[];definitions={};quantity=chosen['quantity'];reduce=np.mean if chosen['aggregation']=='mean' else np.median
    for a in effects:
        for pair in chosen['pairs']:
            if a['measurement']!=pair['reference']:continue
            key=(a['movie'],a['identity'],pair['target'],a['baseline'],a['target_window']);b=lookup.get(key)
            if b is None:raise ValueError('A declared original measurement pair is missing its full comparison inventory')
            if a['source_run']!=b['source_run']:raise ValueError('Paired measurements belong to different original source runs')
            bounds=[];comparable=True
            for row in [a,b]:
                before,after=[original[row[key]] for key in ['baseline_window_id','target_window_id']]
                comparable=comparable and all(w['coordinate']!='frames' or clocks[row['movie']] is not None for w in [before,after])
                bounds.append({'baseline':window_key(before,clocks[row['movie']]),'target':window_key(after,clocks[row['movie']])})
            if bounds[0]!=bounds[1]:raise ValueError('Within-cell coordinated effects use different original comparison windows')
            design=designs[a['movie']];sample=design['sample'];uid=unit_id(a['source_run'],a['movie'],sample)
            definition={**pair,'baseline':a['baseline'],'target_window':a['target_window'],'windows':bounds[0],
                'quantity':quantity,'reference_unit':'fraction' if quantity=='relative_change' else a['unit'],
                'target_unit':'fraction' if quantity=='relative_change' else b['unit'],
                'reference_summary':a['summary_operation'],'target_summary':b['summary_operation'],
                'condition':design['condition'],'anchor_kind':design['anchor']['kind'],'clock_comparable':comparable,
                'model':resolved.request.evidence.as_dict() if quantity=='estimate' else None,
                'relative_settings':{name:resolved.request.relative_effects.get(name) for name in [pair['reference'],pair['target']]} if quantity=='relative_change' else None}
            if not comparable:definition['unestablished_recording']=a['movie']
            did=content_id(definition);definitions[did]={'definition_id':did,**definition}
            av,bv=a.get(quantity),b.get(quantity);available=bool(comparable and a['eligible'] and b['eligible'] and finite(av) and finite(bv))
            cells.append({'source_run':a['source_run'],'movie':a['movie'],'identity':a['identity'],'unit_id':uid,'sample':sample['sample'],'sample_confirmed':sample['confirmed'],
                'pair_id':content_id({'reference_effect':a['effect_id'],'target_effect':b['effect_id'],'definition_id':did}),
                'definition_id':did,**pair,'baseline':a['baseline'],'target_window':a['target_window'],'quantity':quantity,
                'reference_effect_id':a['effect_id'],'target_effect_id':b['effect_id'],'reference_comparison_id':a['comparison_id'],'target_comparison_id':b['comparison_id'],
                'reference_value':float(av) if finite(av) else None,'target_value':float(bv) if finite(bv) else None,
                'reference_outcome':a['outcome'],'target_outcome':b['outcome'],'reference_q_value':a['q_value'],'target_q_value':b['q_value'],
                'reference_response_supported':a['response_supported'],'target_response_supported':b['response_supported'],
                'paired_available':available,'jointly_tested':finite(a['p_value']) and finite(b['p_value']),
                'joint_response':category(a,b),'joint_direction':direction(av,bv),
                'eligible_reason':'Both original paired changes have declared support' if available else 'One/both original changes or their window support is unavailable',
                'population':'All original requested within-cell pairs; no responder filtering'})
    recording_groups={}
    for row in cells:recording_groups.setdefault((row['definition_id'],row['movie']),[]).append(row)
    recordings=[];sample_groups={}
    for (did,movie),rows in sorted(recording_groups.items()):
        valid=[row for row in rows if row['paired_available']];first=rows[0]
        item={key:first[key] for key in ['source_run','movie','unit_id','sample','sample_confirmed','definition_id','reference','target','baseline','target_window']}
        item.update(recording_pair_id=content_id({'definition_id':did,'movie':movie,'source_run':first['source_run']}),
            reference_value=float(reduce([r['reference_value'] for r in valid])) if valid else None,
            target_value=float(reduce([r['target_value'] for r in valid])) if valid else None,
            paired_available=bool(valid),cell_pair_ids=[r['pair_id'] for r in rows],included_pair_ids=[r['pair_id'] for r in valid],
            aggregation=chosen['aggregation'],**counts(rows))
        recordings.append(item);sample_groups.setdefault((did,first['unit_id']),[]).append(item)
    samples=[]
    for (did,uid),rows in sorted(sample_groups.items()):
        first=rows[0];valid=[r for r in rows if r['paired_available']];pair_ids={pid for row in rows for pid in row['cell_pair_ids']}
        cell_rows=[row for row in cells if row['pair_id'] in pair_ids]
        item={key:first[key] for key in ['source_run','unit_id','sample','sample_confirmed','definition_id','reference','target','baseline','target_window']}
        item.update(sample_pair_id=content_id({'definition_id':did,'unit_id':uid}),recordings=[row['movie'] for row in rows],
            requested_recordings=len(rows),included_recordings=len(valid),recording_pair_ids=[row['recording_pair_id'] for row in rows],
            reference_value=float(reduce([r['reference_value'] for r in valid])) if valid else None,
            target_value=float(reduce([r['target_value'] for r in valid])) if valid else None,
            paired_available=bool(valid),formal_eligible=bool(valid and first['sample_confirmed'] and definitions[did]['clock_comparable']),
            cell_pair_ids=sorted(pair_ids),aggregation=chosen['aggregation'],
            experimental_unit='original_biological_sample' if first['sample_confirmed'] else 'unconfirmed_recording',
            aggregation_meaning='Same available original paired cells within a recording; equal available recording summaries within the original sample',**counts(cell_rows))
        samples.append(item)
    associations=[];details=[]
    for did,definition in sorted(definitions.items()):
        rows=[row for row in samples if row['definition_id']==did];valid=[row for row in rows if row['formal_eligible']]
        uid=[row['unit_id'] for row in valid];native_result=native.compare([row['reference_value'] for row in valid],[row['target_value'] for row in valid],uid,chosen['evidence'])
        aid=content_id({'definition_id':did,'units':[row['unit_id'] for row in rows],'method':chosen['evidence']})
        associations.append({'association_id':aid,**definition,'method':chosen['evidence']['method'],'statistic':'spearman',
            'estimate':native_result['estimate'],'p_value':native_result['p_value'],'q_value':None,'supported':False,
            'status':native_result['status'],'reason':native_result['reason'],'family_id':None,
            'requested_units':len(rows),'confirmed_samples':sum(row['sample_confirmed'] for row in rows),'eligible_samples':len(valid),
            'requested_cells':sum(row['requested_cells'] for row in rows),'paired_available_cells':sum(row['paired_available_cells'] for row in rows),
            'sample_pair_ids':[row['sample_pair_id'] for row in rows],'included_sample_pair_ids':[row['sample_pair_id'] for row in valid],
            'evidence_level':'biological_sample_association','selection':'Complete original population; no selected-responder association',
            'non_significance_meaning':'Two non-significant changes do not establish coordination; absence of an association detection does not establish independence'})
        details.append({'association_id':aid,'native_result':native_result})
    families=[];inference=resolved.request.inference.as_dict();groups={}
    if chosen['evidence']['method']!='none':
        for row in associations:
            scope='all' if inference['correction_scope']=='all' else row['reference']+' / '+row['target'];groups.setdefault(scope,[]).append(row)
    for scope,rows in sorted(groups.items()):
        rows.sort(key=lambda row:row['association_id']);ids=[r['association_id'] for r in rows];valid=[finite(r['p_value']) for r in rows]
        fid=content_id({'level':'biological_sample_association','scope':scope,'members':ids,'inference':inference})
        q=circadian.adjust_pvalues([r['p_value'] if use else 1. for r,use in zip(rows,valid)],inference['multiple_testing']) if any(valid) else [None]*len(rows)
        for row,use,adjusted in zip(rows,valid,q):row.update(q_value=float(adjusted) if use else None,family_id=fid,supported=bool(use and adjusted<=inference['alpha']))
        families.append({'family_id':fid,'evidence_level':'biological_sample_association','scope':scope,'requested_scope':inference['correction_scope'],
            'scope_meaning':'All requested measurement-pair/condition/window definitions' if scope=='all' else 'One measurement pair across its original condition/window definitions',
            'members':ids,'requested':len(rows),'tested':sum(valid),'alpha':inference['alpha'],'correction':inference['multiple_testing'],'missing_probability_policy':MISSING_POLICY})
    return {'cell_pairs':_table(cells,['pair_id','source_run','movie','identity','reference','target','joint_response','paired_available']),
        'recording_pairs':_table(recordings,['recording_pair_id','definition_id','unit_id','movie','reference_value','target_value']),
        'sample_pairs':_table(samples,['sample_pair_id','definition_id','unit_id','sample_confirmed','reference_value','target_value','formal_eligible']),
        'definitions':_table(list(definitions.values()),['definition_id','reference','target','baseline','target_window']),
        'associations':_table(associations,['association_id','definition_id','estimate','p_value','q_value','supported','status']),
        'families':_table(families,['family_id','members','requested','tested']),
        'native_details':_table(details,['association_id','native_result'])}


def produce(context):
    request=context.request.request;chosen=policy(request.coordinated.as_dict(),[m.column for m in context.request.measurements],request.inference.as_dict())
    if not chosen['enabled']:return StepResult(context.step.name,context.scientific_id,'skipped-empty','Coordinated response analysis was explicitly disabled')
    windows=context.saved('aligned-windows');source=context.saved('response-evidence')
    data=analyse(read_windows(windows),read_evidence(source,windows.outcome.scientific_id),context.request,chosen)
    context.output.mkdir(parents=True);refs=[]
    for name,frame in data.items():
        path=context.output/(name+'.json');path = write_table(path,frame);refs.append(ArtifactRef(name,path.name,file_hash(path),context.scientific_id,columns=tuple(frame.columns)))
    path=context.output/'provenance.json';_write_json(path,{'schema_version':1,'scientific_id':context.scientific_id,'windows_id':windows.outcome.scientific_id,
        'evidence_id':source.outcome.scientific_id,'resolved_request':context.request.as_dict(),'applied_policy':chosen,
        'libraries':{name:version(name) for name in ['numpy','pandas','scipy']},'references':native.REFERENCES,
        'population':'Complete original within-cell paired changes, including one/both inconclusive outcomes; no selected-responder population',
        'experimental_unit':'One confirmed biological sample, with equal available recording summaries; conditions and original window definitions remain separate',
        'control_independence':'This branch consumes original change evidence; it does not require an optional treatment/control group comparison',
        'missing_probability_policy':MISSING_POLICY})
    refs.append(ArtifactRef('provenance',path.name,file_hash(path),context.scientific_id))
    return StepResult(context.step.name,context.scientific_id,'completed','Saved full within-cell response pairs, original sample recurrence and separate native sample associations',tuple(refs))


def read_patterns(saved,expected_evidence_id=None):
    provenance=read_document(saved.artifact('provenance'))
    if provenance.get('schema_version')!=1 or provenance.get('scientific_id')!=saved.outcome.scientific_id:raise ValueError('Coordinated response identity/schema mismatch')
    if expected_evidence_id is not None and provenance['evidence_id']!=expected_evidence_id:raise ValueError('Coordinated responses use different original change evidence')
    data={name:read_table(saved.artifact(name)) for name in TABLES}
    for name,key in [('cell_pairs','pair_id'),('recording_pairs','recording_pair_id'),('sample_pairs','sample_pair_id'),('associations','association_id')]:
        if data[name][key].duplicated().any():raise ValueError('Repeated original coordinated response result: '+name)
    for row in data['sample_pairs'].to_dict('records'):
        if set(row['recording_pair_ids'])-set(data['recording_pairs'].recording_pair_id) or set(row['cell_pair_ids'])-set(data['cell_pairs'].pair_id):raise ValueError('Coordinated response membership lost its original sample/cell')
    data['provenance']=provenance;return data
