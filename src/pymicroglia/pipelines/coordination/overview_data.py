"""Display-only coverage, exact effects, matrices and sample rows from saved results."""
import math
import pandas as pd

from pymicroglia.pipelines._contracts import content_id
from pymicroglia.pipelines.coordination.options import BRANCHES
from pymicroglia.pipelines.coordination.inputs import read_inputs
from pymicroglia.pipelines.coordination.evidence import read_evidence
from pymicroglia.pipelines.coordination.samples import read_samples
from pymicroglia.pipelines.coordination.display import entry
from pymicroglia.pipelines._screening import _json_value, read_table

VIEWS=['coverage','effects','matrices','proximity','timing','samples']
GROUP=['question','evidence_level','representation','adjustment','statistic','reference','target','reference_measurement_id','target_measurement_id',
    'is_hypothesis','model_id','evidence_kind','reference_state_id','target_state_id','distance_effect','coordination_question','coordination_measure','distance_unit']
CLAIMS={'coverage':'Saved coverage distinguishes evaluated questions, unavailable data and unsupported relationships at their original evidence level.',
    'effects':'Complete saved spatial effects retain their distance definitions and original pair or recording evidence, including results without detected support.',
    'matrices':'Saved cell-pair effects retain the original measurement orientation; an unavailable or unrequested matrix entry is not a measured zero.',
    'proximity':'Original saved windows show the relationship between changing cell distance and measured coordination without treating windows as independent samples.',
    'timing':'Native timing descriptions retain each measured pair period, comparability and uncertainty; different periods do not establish shared timing.',
    'samples':'Complete sample summaries and independent-sample contrasts retain biological membership, exclusions and saved uncertainty.'}


def options(presentation):
    declared=presentation.as_dict().get('overview',{})
    if not isinstance(declared,dict) or set(declared)-{'views','matrix_block_size','rows_per_page','text'}:raise ValueError('Coordination overview accepts views, matrix_block_size, rows_per_page and text')
    result={'views':VIEWS,'matrix_block_size':8,'rows_per_page':16,**{k:v for k,v in declared.items() if k!='text'}}
    if not isinstance(result['views'],list) or not result['views'] or set(result['views'])-set(VIEWS) or len(set(result['views']))!=len(result['views']):
        raise ValueError('Overview views must uniquely name coverage, effects, matrices, proximity, timing or samples')
    for key,maximum in [('matrix_block_size',12),('rows_per_page',32)]:
        if isinstance(result[key],bool) or not isinstance(result[key],int) or not 1<=result[key]<=maximum:raise ValueError(key+' must be a positive integer no larger than '+str(maximum))
    return result,declared.get('text',{})


def collect(context):
    available=lambda name:name in context.dependencies and context.saved(name).outcome.status in {'completed','reused'}
    data={'prepared':read_inputs(context.saved('pair-inputs')) if available('pair-inputs') else None,
        'evidence':read_evidence(context.saved('coordination-evidence')) if available('coordination-evidence') else None,
        'samples':read_samples(context.saved('coordination-samples')) if available('coordination-samples') else None,
        'proximity':read_table(context.saved('changing-proximity').artifact('windows')) if available('changing-proximity') else pd.DataFrame()}
    records=[]
    for step,question in [*BRANCHES.items(),('coordination-samples','samples')]:
        saved=context.dependencies.get(step);enabled=context.request.request.sample_summary['enabled'] if question=='samples' else context.request.request.questions[question]['enabled']
        frame=data['samples']['comparisons'] if question=='samples' and data['samples'] is not None else data['evidence']['effects'].loc[data['evidence']['effects'].question.eq(question)] if question!='samples' and data['evidence'] is not None else pd.DataFrame()
        levels=(sorted(frame.evidence_level.dropna().unique()) if 'evidence_level' in frame else []) or ['biological_sample' if question=='samples' else 'not_evaluated']
        for level in levels:
            rows=frame.loc[frame.evidence_level.eq(level)] if 'evidence_level' in frame else frame
            records.append(entry(kind='coverage',question=question,evidence_level=level,source_step=step,
                status='disabled' if not enabled else 'available' if available(step) else saved.outcome.status if saved else 'unavailable',
                reason='Question explicitly disabled' if not enabled else saved.outcome.reason if saved else 'No saved result',
                effect_rows=len(rows),tested=int(rows.p_value.notna().sum()) if 'p_value' in rows else 0,
                supported=int(rows.supported.sum()) if 'supported' in rows else 0,source_scientific_id=saved.outcome.scientific_id if saved else None,
                effect=None,p_value=None,q_value=None))
    data['coverage']=records
    return data


def definition(row):return {key:row.get(key) for key in GROUP}


def effect_rows(data):
    if data['evidence'] is None:return []
    inventory=data['prepared']['inventory'] if data['prepared'] is not None else pd.DataFrame()
    result=[]
    for row in _json_value(data['evidence']['effects'].to_dict('records')):
        # Native model/null payloads are retained in the source-bound evidence
        # artifact. Only the saved values/uncertainty actually encoded by this
        # figure belong in its plotted table; no fitted model is redrawn here.
        row={key:value for key,value in row.items() if key not in {'native_timing_evidence','native_details','source_evidence'}}
        geometry=inventory.loc[inventory.pair_id.eq(row.get('pair_id'))] if len(inventory) else inventory
        if len(geometry)==0 and row['question']=='states' and len(inventory):
            geometry=inventory.loc[inventory.source_run.eq(row['source_run'])&inventory.movie.eq(row['movie'])&
                inventory.reference_identity.eq(row['reference_identity'])&inventory.target_identity.eq(row['target_identity'])]
        distances=geometry.static_distance.dropna().unique() if len(geometry) else []
        if len(distances)>1:raise ValueError('One original oriented cell pair has conflicting saved geometry')
        unit=geometry.distance_unit.iloc[0] if len(geometry) else row.get('distance_unit')
        row={**row,'distance':float(distances[0]) if len(distances) else None,'distance_unit':unit,
            'oriented':bool(geometry.oriented.iloc[0]) if len(geometry) and row['question']!='states' else True,
            'display_value':row.get('effect'),'display_value_meaning':'Original saved effect; its observation population and evidence remain in the accompanying table'}
        result.append(entry(**row,kind='effect',group_id=content_id(definition(row))))
    return result


def matrix(rows,row_ids,column_ids):
    """Reshape exact saved effects; mirror only explicitly symmetric pairs."""
    lookup={};group=rows[0]['group_id'];movie=rows[0]['movie'];source=rows[0]['source_run']
    for row in rows:
        a,b=int(row['reference_identity']),int(row['target_identity'])
        for key,mirrored in [((a,b),False)]+([((b,a),True)] if not row['oriented'] else []):
            if key in lookup:raise ValueError('A matrix definition repeats an ordered pair effect')
            lookup[key]=(row,mirrored)
    output=[]
    for y,a in enumerate(row_ids):
        for x,b in enumerate(column_ids):
            original,mirrored=lookup.get((a,b),({},False))
            base={k:v for k,v in original.items() if k not in {'entry_id','kind'}}
            output.append(entry(**{**base,'kind':'matrix','group_id':group,'source_run':source,'movie':movie,
                'row_index':y,'column_index':x,'row_identity':a,'column_identity':b,'mirrored_symmetric':mirrored,
                'display_value':original.get('display_value'),'effect_id':original.get('effect_id'),
                'status':original.get('decision','same_cell_not_tested' if a==b else 'not_requested'),
                'reason':original.get('decision_reason','No self-pair analysis' if a==b else 'This oriented endpoint pair was not requested'),
                'supported':bool(original.get('supported',False))}))
    return output


def pages(data,settings):
    values={};statistics={};result=[];size=settings['rows_per_page']
    def add(view,title,rows,**extra):
        if not rows:return
        for row in rows:
            values[row['entry_id']]=row
            if row.get('kind') not in {'matrix'} or row.get('effect_id'):statistics[row['entry_id']]=row
        claim_view={'recordings':'effects','contrasts':'samples','sample_timing':'samples'}.get(view,view)
        result.append({'view':view,'title':title,'claim':CLAIMS[claim_view],
            'footnote':extra.pop('footnote','Values and decisions are saved results. Unavailable is not zero. Figures do not refit, correct or select scientific support.'),
            'entry_ids':[row['entry_id'] for row in rows],**extra})
    if 'coverage' in settings['views']:add('coverage','Spatial questions, evidence and coverage',data['coverage'])
    effects=effect_rows(data);groups={}
    for row in effects:groups.setdefault(row['group_id'],[]).append(row)
    for group,rows in sorted(groups.items()):
        info=definition(rows[0]);question=rows[0]['question']
        if question=='rhythm':
            if 'timing' in settings['views']:
                for start in range(0,len(rows),size):add('timing','Saved timing between measured cell pairs',rows[start:start+size],definition=info,
                    footnote='Offsets retain their own native reference and period. Positive offset means target follows reference. Different pair periods are not a shared clock; original uncertainty and refusals remain explicit.')
            continue
        if 'effects' in settings['views']:
            if rows[0]['evidence_level']=='pair':add('effects','Saved pair effects and measured distance',rows,definition=info,
                footnote='Distance is between coordinate-wise recording medians, not persistent proximity. Points show original saved effects; no fitted curve, new bins or independent-pair population test.')
            else:
                for start in range(0,len(rows),size):add('recordings','Saved whole-recording spatial evidence',rows[start:start+size],definition=info)
        if 'matrices' in settings['views'] and rows[0]['evidence_level']=='pair':
            for movie in sorted({row['movie'] for row in rows}):
                members=[row for row in rows if row['movie']==movie]
                ids=sorted({int(row[key]) for row in members for key in ['reference_identity','target_identity']})
                blocks=[ids[start:start+settings['matrix_block_size']] for start in range(0,len(ids),settings['matrix_block_size'])]
                for row_ids in blocks:
                    for column_ids in blocks:
                        cells=matrix(members,row_ids,column_ids)
                        if not any(row.get('effect_id') for row in cells):continue
                        add('matrices','Saved cell-pair effects | '+movie,cells,definition=info,row_ids=row_ids,column_ids=column_ids,
                            footnote='Rows: reference cells. Columns: target cells. Symmetric copies appear only for explicitly unordered same-measurement pairs. Grey is unavailable or unrequested; a dot marks saved corrected pair support.')
    if 'proximity' in settings['views'] and not data['proximity'].empty:
        frame=data['proximity']
        for (pair,view),group in frame.groupby(['pair_id','view_id'],sort=True):
            rows=[entry(**row,kind='window') for row in _json_value(group.to_dict('records'))]
            add('proximity','Changing proximity | '+rows[0]['movie'],rows,definition={key:rows[0].get(key) for key in GROUP},
                footnote='Each point is an original saved time window. Overlapping windows are dependent and supply no independent sample count. Missing support remains in the accompanying table.')
    if 'samples' in settings['views'] and data['samples'] is not None:
        samples=data['samples'];definitions={row['question_id']:row for row in _json_value(samples['definitions'].to_dict('records'))}
        for qid,group in samples['unit_summaries'].groupby('question_id',sort=True):
            if definitions[qid]['value_kind']=='native_rhythm_timing':continue
            rows=[entry(**row,kind='sample') for row in _json_value(group.to_dict('records'))]
            finite=[row for row in rows if isinstance(row.get('value'),(int,float)) and math.isfinite(row['value'])]
            unavailable=[row for row in rows if row not in finite]
            chunks=[finite[start:start+size] for start in range(0,len(finite),size)] or [[]]
            for number,chunk in enumerate(chunks):add('samples','Recorded biological-sample summaries',chunk+(unavailable if number==0 else []),definition=definitions[qid],
                total_units=len(rows),available_units=len(finite),footnote=samples['provenance']['scalar_weighting']+'. Hollow points are excluded from formal inference. All unavailable units remain in the accompanying table; pairs are not independent samples.')
        for qid,group in samples['comparisons'].groupby('question_id',sort=True):
            comparisons=[entry(**row,kind='comparison') for row in _json_value(group.to_dict('records'))]
            for start in range(0,len(comparisons),size):add('contrasts','Independent biological-sample contrasts',comparisons[start:start+size],definition=definitions[qid],
                footnote='Effect: target condition minus reference condition. Intervals, when present, describe the saved observed-sample aggregate contrast; they are not simultaneous family intervals. Corrected sample evidence stays separate from pair evidence.')
        for row in _json_value(samples['timing_populations'].to_dict('records')):
            add('sample_timing','Native timing summary across biological samples',[entry(**row,kind='sample_timing')],
                footnote=samples['provenance']['timing_policy'].get('weighting','Native timing was not requested')+'. Per-condition descriptions are not a direct condition-phase test.')
    if not result:
        add('coverage','No requested display has available saved results',data['coverage'])
    return pd.DataFrame(values.values()),pd.DataFrame(statistics.values()),result
