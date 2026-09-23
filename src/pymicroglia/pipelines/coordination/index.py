"""Portable exact pair/effect/recording navigation over saved coordination results."""
from pymicroglia._results import report_name, read_document
from .._saved_figures import table_name
from pymicroglia._sources import source_file
from pathlib import Path
import json
import pandas as pd

from pymicroglia.pipelines.audit.index import copy_evidence, inventory_artifact
from pymicroglia.pipelines._contracts import ArtifactRef, Settings, StepResult, content_id, result_to_dict, result_from_dict
from pymicroglia.pipelines._runner import SavedResult
from pymicroglia.pipelines._screening import _json_value, _write_json, file_hash, read_table
from pymicroglia.pipelines.coordination.inputs import read_inputs
from pymicroglia.pipelines.coordination.evidence import read_evidence
from pymicroglia.pipelines.coordination.samples import read_samples

RENDERS=['coordination-overview','connection-maps','pair-report-cards']


def version():
    return content_id({name:file_hash(source_file(name)) for name in ['coordination_index.py','coordination_index_html.py','audit_index.py','behaviour_index.py','coordination_inputs.py','coordination_evidence.py','coordination_samples.py']})


def cell_id(source,movie,identity):return 'cell-'+content_id({'source_run':source,'movie':movie,'identity':int(identity)})
def recording_id(source,movie):return 'recording-'+content_id({'source_run':source,'movie':movie})


def build(output,inventory,requested,title='Spatial coordination'):
    output=Path(output)
    def source(step,name):
        reference=inventory_artifact(output,inventory,step,name)
        if reference is None:return None
        path=(output/reference['path']).resolve()
        if not path.is_relative_to(output.resolve()) or not path.is_file() or file_hash(path)!=reference['sha256']:raise ValueError('Missing or changed coordination index source')
        return path
    def table(step,name):
        path=source(step,name);return read_table(path) if path else pd.DataFrame()
    def metadata(step,name):
        path=source(step,name);return read_document(path) if path else {}
    def path_for(step,name):
        if source(step,name) is None:raise ValueError('A navigation target has no saved artifact: '+step+'/'+name)
        return inventory_artifact(output,inventory,step,name)['path']
    def saved(step):
        item=inventory.get(step,{})
        if item.get('status') not in {'completed','reused'}:return None
        record=(output/item['record']).resolve()
        if not record.is_relative_to(output.resolve()) or file_hash(record)!=item['record_sha256']:raise ValueError('Changed original execution outcome')
        return SavedResult(output/'evidence'/step,result_from_dict(read_document(record)))
    prepared=read_inputs(saved('pair-inputs')) if saved('pair-inputs') else {}
    original=prepared.get('provenance',{}).get('resolved_request',{})
    if original and original['inputs']!=requested['inputs']:raise ValueError('Index inputs differ from the original measured pair population')
    evidence=read_evidence(saved('coordination-evidence'),inventory['pair-inputs']['scientific_id']) if saved('coordination-evidence') else {}
    sample_data=read_samples(saved('coordination-samples'),inventory['coordination-evidence']['scientific_id']) if saved('coordination-samples') else {}
    def rows(data,key):return _json_value(data.get(key,pd.DataFrame()).to_dict('records'))
    source_run=requested['inputs']['source_run'];cells={};recordings={}
    for row in rows(prepared,'cells') or requested['inputs']['cells']:
        key=cell_id(row['source_run'],row['movie'],row['identity']);rid=recording_id(row['source_run'],row['movie'])
        if row['source_run']!=source_run or key in cells:raise ValueError('Repeated or foreign original cell identity')
        cells[key]={**row,'id':key,'recording':rid,'pairs':[],'pages':[]}
        recordings.setdefault(rid,{'id':rid,'source_run':row['source_run'],'movie':row['movie'],'cells':[],'pairs':[],'effects':[],'pages':[],'samples':[]})['cells'].append(key)
    pairs={};decisions={row['pair_id']:row for row in rows(evidence,'pair_decisions')}
    for row in rows(prepared,'inventory'):
        endpoint_cells=[cell_id(row['source_run'],row['movie'],row[role+'_identity']) for role in ['reference','target']]
        if any(key not in cells for key in endpoint_cells):raise ValueError('Pair lost an exact original cell')
        rid=recording_id(row['source_run'],row['movie']);decision=decisions.get(row['pair_id'],{})
        pairs[row['pair_id']]={**row,'id':'pair-'+row['pair_id'],'cells':endpoint_cells,'recording':rid,'decision':decision,'effects':[],'pages':[],'card_pages':[]}
        recordings[rid]['pairs'].append('pair-'+row['pair_id'])
        for key in endpoint_cells:cells[key]['pairs'].append('pair-'+row['pair_id'])
    effects={};members={}
    for row in rows(evidence,'effect_pair_membership'):members.setdefault(row['effect_id'],[]).append(row['pair_id'])
    for original_effect in rows(evidence,'effects'):
        row={key:value for key,value in original_effect.items() if key not in {'native_timing_evidence','native_details','source_evidence'}}
        eid=row['effect_id'];rid=recording_id(row['source_run'],row['movie'])
        if rid not in recordings:raise ValueError('Effect lost its original recording')
        related=members.get(eid,[])
        if any(key not in pairs for key in related):raise ValueError('Effect lost its exact measured pair membership')
        effects[eid]={**row,'id':'effect-'+eid,'pairs':['pair-'+key for key in related],'recording':rid,'pages':[],
            'source_table':path_for('coordination-evidence','effects'),'original_source':path_for(row['source_step'],'recording_effects' if row['evidence_level']=='recording' else 'pair_characteristics' if row['question']=='characteristics' else 'pair_effects')}
        recordings[rid]['effects'].append('effect-'+eid)
        for key in related:pairs[key]['effects'].append('effect-'+eid)
    units={}
    # Even a disabled sample-analysis branch retains the original declared
    # sample memberships. This is inventory bookkeeping, not an aggregate.
    unit_rows=rows(sample_data,'unit_inventory')
    if not unit_rows:
        for assignment in requested['inputs']['samples']:
            uid=content_id({'source_run':source_run,'level':'biological_sample' if assignment['confirmed'] else 'recording',
                'sample':assignment['sample'] if assignment['confirmed'] else assignment['movie']})
            existing=next((row for row in unit_rows if row['unit_id']==uid),None)
            if existing:existing['movies'].append(assignment['movie'])
            else:unit_rows.append({'source_run':source_run,'unit_id':uid,'sample':assignment['sample'],'sample_confirmed':assignment['confirmed'],
                'movies':[assignment['movie']],'condition':requested['request']['conditions'].get(assignment['movie']),'unit_level':'biological_sample' if assignment['confirmed'] else 'unconfirmed_recording'})
    for row in unit_rows:
        rids=[recording_id(row['source_run'],movie) for movie in row['movies']]
        if any(rid not in recordings for rid in rids):raise ValueError('Sample lost its original recording membership')
        units[row['unit_id']]={**row,'id':'sample-'+row['unit_id'],'recording_targets':rids,'pages':[],
            'summaries':[item for item in rows(sample_data,'unit_summaries') if item['unit_id']==row['unit_id']]}
        for rid in rids:recordings[rid]['samples'].append('sample-'+row['unit_id'])
    pages=[];entries=[];display_omissions={}
    for step in RENDERS:
        manifest=metadata(step,'coordination_display_manifest.json');provenance=metadata(step,'coordination_display_provenance.json')
        if not manifest:continue
        if provenance.get('evidence_id') and provenance['evidence_id']!=inventory.get('coordination-evidence',{}).get('scientific_id'):raise ValueError('Figure uses another scientific evidence population')
        if provenance.get('pair_inputs_id') and provenance['pair_inputs_id']!=inventory.get('pair-inputs',{}).get('scientific_id'):raise ValueError('Figure uses another original input population')
        display_omissions[step]={'selection':provenance.get('selection',{}),'settings':manifest.get('settings',{})};local={}
        for page in manifest['pages']:
            master=page['master'];key='figure-'+content_id({'step':step,'scientific_id':inventory[step]['scientific_id'],'master':master})
            if master in local:raise ValueError('Repeated figure target')
            local[master]={'id':key,'step':step,'view':page['view'],'title':page['title'],'path':path_for(step,master),
                'semantics':{key:value for key,value in page.items() if key not in {'entry_ids','effect_entries'}},'cells':[],'pairs':[],'effects':[],'recordings':[],'samples':[],
                'data':[{'label':label,'path':path_for(step,table_name(master,kind,lambda name:source(step,name)))} for kind,label in [('values','Exact plotted values'),('statistics','Saved statistics'),('display','Frozen display settings')]]}
            pages.append(local[master])
        for entry in _json_value(table(step,'entries.json').to_dict('records')):
            page=local.get(entry['master'])
            if page is None:raise ValueError('Semantic entry has no saved figure target')
            pair_ids=[key for key in [entry.get('pair_id'),entry.get('card_pair_id'),*(entry.get('requested_pair_ids') or [])] if key]
            pair_ids=list(dict.fromkeys(pair_ids));eid=entry.get('effect_id')
            if any(key not in pairs for key in pair_ids):raise ValueError('Figure entry uses an unknown measured pair')
            if eid:
                actual=effects.get(eid)
                if actual is None:raise ValueError('Figure entry uses an unknown scientific effect')
                for key in ['source_scientific_id','source_result_id','question','evidence_level','model_id','representation','adjustment']:
                    if entry.get(key) is not None and entry[key]!=actual.get(key):raise ValueError('Figure entry changed its scientific effect meaning')
                if any('pair-'+key not in actual['pairs'] for key in pair_ids):raise ValueError('Figure attached an effect to another measured pair')
                page['effects'].append('effect-'+eid);effects[eid]['pages'].append(page['id'])
            for pid in pair_ids:
                original_pair=pairs[pid]
                for key in ['source_run','movie','reference_identity','target_identity','reference_endpoint_id','target_endpoint_id']:
                    if entry.get(key) is not None and entry[key]!=original_pair.get(key):raise ValueError('Figure entry swapped its original pair endpoints')
                page['pairs'].append(original_pair['id']);original_pair['pages'].append(page['id'])
                page['cells'].extend(original_pair['cells'])
                if step=='pair-report-cards':original_pair['card_pages'].append(page['id'])
            if entry.get('source_run') and entry.get('movie'):
                rid=recording_id(entry['source_run'],entry['movie'])
                if rid not in recordings:raise ValueError('Figure entry changed recording identity')
                page['recordings'].append(rid)
                for field in ['identity','reference_identity','target_identity','row_identity','column_identity']:
                    if entry.get(field) is not None:
                        cid=cell_id(entry['source_run'],entry['movie'],entry[field])
                        if cid not in cells:raise ValueError('Figure entry changed cell identity')
                        page['cells'].append(cid)
            uid=entry.get('unit_id')
            if uid:
                if uid not in units:raise ValueError('Figure entry changed experimental-unit identity')
                page['samples'].append(units[uid]['id'])
            entries.append({'entry_id':entry['entry_id'],'page':page['id'],'pairs':['pair-'+key for key in pair_ids],'effect':'effect-'+eid if eid else None,
                'observation_id':entry.get('observation_id'),'hours':entry.get('hours'),'view':page['view']})
        for page in local.values():
            for key in ['cells','pairs','effects','recordings','samples']:page[key]=list(dict.fromkeys(page[key]))
            for cid in page['cells']:cells[cid]['pages'].append(page['id']);page['recordings'].append(cells[cid]['recording'])
            page['recordings']=list(dict.fromkeys(page['recordings']))
            for rid in page['recordings']:
                recordings[rid]['pages'].append(page['id']);page['samples'].extend(recordings[rid]['samples'])
            page['samples']=list(dict.fromkeys(page['samples']))
            for uid in page['samples']:units[uid.removeprefix('sample-')]['pages'].append(page['id'])
    for collection in [cells,pairs,effects,recordings,units]:
        for row in collection.values():
            for key in ['pages','card_pages']:
                if key in row:row[key]=list(dict.fromkeys(row[key]))
    navigation={'schema_version':1,'source_run':source_run,'analysis_recomputed':False,'requested_settings':requested,'original_prepared_settings':original,
        'cells':list(cells.values()),'pairs':list(pairs.values()),'effects':list(effects.values()),'recordings':list(recordings.values()),'samples':list(units.values()),
        'families':rows(evidence,'families'),'branches':rows(evidence,'branches'),'pages':pages,'entries':entries,'display_omissions':display_omissions,
        'sample_comparisons':rows(sample_data,'comparisons'),'inputs':inventory,'selections':evidence.get('provenance',{}).get('selections',{}),
        'external_source_policy':'Copied tables, native result artifacts and figure bundles move with this report. Original image stacks and external source execution paths remain external references; they were not re-analysed or copied as original image data.'}
    from pymicroglia.pipelines.coordination.index_html import document
    _write_json(output/'navigation.json',navigation);(output/'index.html').write_text(document(navigation,title),encoding='utf-8')
    return navigation


def validate_links(output):
    from pymicroglia.pipelines.behaviour.index import validate_links as validate
    return validate(output)


def produce(context):
    appearance=context.presentation.as_dict().get('report',{})
    if not isinstance(appearance,dict) or set(appearance)-{'title'} or not isinstance(appearance.get('title',''),str):raise ValueError('Report presentation accepts a title string')
    context.output.mkdir(parents=True);inventory=copy_evidence(context.dependencies,context.output)
    directory=context.output/'execution-records';directory.mkdir()
    for name,saved in context.dependencies.items():
        path=directory/(name+'.json');path = _write_json(path,result_to_dict(saved.outcome));inventory[name].update(record=path.relative_to(context.output).as_posix(),record_sha256=file_hash(path))
    navigation=build(context.output,inventory,context.request.as_dict(),**appearance)
    _write_json(context.output/'link-validation.json',validate_links(context.output))
    refs=tuple(ArtifactRef(report_name(path,context.output),path.relative_to(context.output).as_posix(),file_hash(path),context.scientific_id)
        for path in sorted(context.output.rglob('*')) if path.is_file() and path.name!="artefacts.json")
    return StepResult(context.step.name,context.scientific_id,'completed','Saved portable full pair, effect, recording, sample and figure navigation with original outcomes',refs,
        provenance=Settings({'analysis_recomputed':False,'source_run':navigation['source_run']}))
