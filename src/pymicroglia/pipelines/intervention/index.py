"""Portable, verified navigation over original intervention evidence; no analysis."""
from pymicroglia._results import report_name, read_document
from .._saved_figures import table_name
from pymicroglia._sources import source_file
from pathlib import Path
import json
import pandas as pd
from pymicroglia.pipelines.audit.index import copy_evidence, inventory_artifact
from pymicroglia.pipelines._contracts import ArtifactRef, Settings, StepResult, cell_number, content_id, result_to_dict, result_from_dict
from pymicroglia.pipelines._runner import SavedResult
from pymicroglia.pipelines._screening import _json_value, _write_json, file_hash, read_table
from pymicroglia.pipelines.intervention.control_inputs import unit_id

RENDERS=['response-overview','cell-reports-and-grids','timing-and-rhythm-figures','sample-and-pattern-figures']
KEYS=['source_run','movie','identity']
SPECS=[('windows','aligned-windows','windows','window_id'),('comparisons','aligned-windows','comparisons','comparison_id'),
    ('effects','response-evidence','effects','effect_id'),('timing','response-timing','timing','timing_id'),
    ('rhythm_windows','rhythm-changes','window_results','window_id'),('rhythm_comparisons','rhythm-changes','comparisons','rhythm_comparison_id'),
    ('direct_changes','rhythm-changes','direct_comparisons','direct_id'),('control_comparisons','control-comparisons','comparisons','comparison_id'),
    ('cell_pairs','coordinated-responses','cell_pairs','pair_id'),('sample_pairs','coordinated-responses','sample_pairs','sample_pair_id'),
    ('associations','coordinated-responses','associations','association_id')]


def version():
    names=['intervention_index.py','intervention_index_html.py','audit_index.py','behaviour_index.py',
        'intervention_windows.py','intervention_evidence.py','intervention_controls.py','intervention_timing.py','intervention_rhythms.py','intervention_patterns.py',
        'intervention_control_inputs.py','rhythm_discovery.py']
    return content_id({name:file_hash(source_file(name)) for name in names})


def cell_key(row):return content_id({**{key:row[key] for key in KEYS[:2]},'identity':cell_number(row['identity'])})
def recording_key(row):return content_id({key:row[key] for key in KEYS[:2]})


def build(output,inventory,requested,title='Intervention response'):
    output=Path(output).resolve();saved={}
    def verified(relative,expected):
        path=(output/relative).resolve()
        if not path.is_relative_to(output) or not path.is_file() or file_hash(path)!=expected:raise ValueError('Missing or changed index source')
        return path
    def artifact(step,name):
        ref=inventory_artifact(output,inventory,step,name)
        return verified(ref['path'],ref['sha256']) if ref else None
    def path_for(step,name):
        if artifact(step,name) is None:raise ValueError('Navigation target has no saved artifact: '+step+'/'+name)
        return inventory_artifact(output,inventory,step,name)['path']
    def metadata(step,name):
        path=artifact(step,name);return read_document(path) if path else {}
    for step,item in inventory.items():
        record=read_document(verified(item['record'],item['record_sha256']))
        outcome=result_from_dict(record)
        if outcome.step!=step:raise ValueError('Index changed the original step identity')
        if any(getattr(outcome,key)!=item[key] for key in ['status','reason','scientific_id']):raise ValueError('Index changed the original execution outcome')
        for ref in outcome.artifacts:
            actual=item['artifacts'].get(ref.name)
            if actual is None or actual['sha256']!=ref.sha256 or actual['scientific_id']!=ref.scientific_id:raise ValueError('Index changed the original artifact receipt')
            if actual['path']!=(Path('evidence')/step/ref.path).as_posix():raise ValueError('Index changed the original artifact location')
            artifact(step,ref.name)
        if set(item['artifacts'])!={ref.name for ref in outcome.artifacts}:raise ValueError('Index added an unrecorded artifact')
        if outcome.status in {'completed','reused'}:saved[step]=SavedResult(output/'evidence'/step,outcome)
    from pymicroglia.pipelines.intervention.windows import read_windows
    from pymicroglia.pipelines.intervention.evidence import read_evidence
    from pymicroglia.pipelines.intervention.controls import read_controls
    from pymicroglia.pipelines.intervention.timing import read_timing
    from pymicroglia.pipelines.intervention.rhythms import read_rhythms
    from pymicroglia.pipelines.intervention.patterns import read_patterns
    data={};wid=inventory.get('aligned-windows',{}).get('scientific_id');eid=inventory.get('response-evidence',{}).get('scientific_id')
    for step,reader,expected in [('aligned-windows',read_windows,None),('response-evidence',read_evidence,wid),
            ('control-comparisons',read_controls,eid),('response-timing',read_timing,eid),('rhythm-changes',read_rhythms,wid),('coordinated-responses',read_patterns,eid)]:
        if step in saved:
            data[step]=reader(saved[step]) if step=='aligned-windows' else reader(saved[step],expected)
            if step!='aligned-windows' and data[step]['provenance']['windows_id']!=wid:raise ValueError('Index mixes different original windows')
    original=data.get('aligned-windows',{}).get('provenance',{}).get('resolved_request',{})
    design=metadata('intervention-design','intervention_design')
    original=original or design.get('request',{})
    if original and original['inputs']!=requested['inputs']:raise ValueError('Index differs from the original requested population')
    def rows(step,table):return _json_value(data.get(step,{}).get(table,pd.DataFrame()).to_dict('records'))
    collections={name:{} for name in ['cells','recordings','samples','declared_windows','declared_pairs','families',*[spec[0] for spec in SPECS]]};nodes={}
    def add(group,key,row,source=None):
        if key in collections[group]:raise ValueError('Repeated original '+group+' identity')
        node={'id':group+'-'+key,'record':row,'links':[],'pages':[],'context_pages':[],'source_table':source}
        collections[group][key]=node;nodes[node['id']]=node;return node
    def get(group,key):
        node=collections[group].get(key)
        if node is None:raise ValueError('Lost original '+group+' membership')
        return node
    def connect(a,b):
        if a['id']!=b['id']:
            if b['id'] not in a['links']:a['links'].append(b['id'])
            if a['id'] not in b['links']:b['links'].append(a['id'])
    source_run=requested['inputs']['source_run']
    for row in requested['inputs']['cells']:
        if row['source_run']!=source_run:raise ValueError('Foreign original cell source')
        cell=add('cells',cell_key(row),row);rid=recording_key(row)
        recording=collections['recordings'].get(rid) or add('recordings',rid,{'source_run':source_run,'movie':row['movie'],**requested['recordings'][row['movie']]})
        connect(cell,recording)
    original_cells=rows('aligned-windows','cells')
    if original_cells and {cell_key(r) for r in original_cells}!=set(collections['cells']):raise ValueError('Prepared cells differ from complete requested inventory')
    for assignment in requested['inputs']['samples']:
        uid=unit_id(source_run,assignment['movie'],assignment);rid=recording_key({'source_run':source_run,'movie':assignment['movie']})
        unit=collections['samples'].get(uid)
        if unit is None:unit=add('samples',uid,{'source_run':source_run,'unit_id':uid,'sample':assignment['sample'],'sample_confirmed':assignment['confirmed'],'movies':[]})
        unit['record']['movies'].append(assignment['movie']);recording=get('recordings',rid);connect(unit,recording)
        for cid in list(recording['links']):
            if cid.startswith('cells-'):connect(unit,nodes[cid])
    for group,step,table,key in SPECS:
        for row in rows(step,table):
            node=add(group,row[key],row,path_for(step,table))
            if all(k in row for k in KEYS):connect(node,get('cells',cell_key(row)))
            if row.get('unit_id'):connect(node,get('samples',row['unit_id']))
    if 'aligned-windows' not in data:
        for cell in requested['inputs']['cells']:
            for measured in requested['measurements']:
                for window in requested['recordings'][cell['movie']]['windows']:
                    row={**cell,'measurement':measured['column'],**window,'status':'not_prepared','reason':inventory.get('aligned-windows',{}).get('reason','No prepared original windows')}
                    connect(add('declared_windows',content_id(row),row),get('cells',cell_key(cell)))
    if 'coordinated-responses' not in data and requested['request']['coordinated']['enabled']:
        from pymicroglia.pipelines.rhythm.discovery import _pairs
        pairs=_pairs(requested['request']['coordinated'].get('pairs'),tuple(m['column'] for m in requested['measurements']))
        for cell in requested['inputs']['cells']:
            for pair in pairs:
                for window in requested['recordings'][cell['movie']]['windows']:
                    if window['baseline'] is None:continue
                    row={**cell,**pair.as_dict(),'baseline':window['baseline'],'target_window':window['name'],
                        'status':'not_produced','reason':inventory.get('coordinated-responses',{}).get('reason','No original paired result was produced')}
                    connect(add('declared_pairs',content_id(row),row),get('cells',cell_key(cell)))
    # Link namespaces explicitly: a control comparison is never a cell comparison.
    for group,_,_,_ in SPECS:
        for node in collections[group].values():
            row=node['record']
            if group!='control_comparisons':
                for field,target in [('baseline_window_id','windows'),('target_window_id','windows'),('comparison_id','comparisons'),
                        ('effect_id','effects'),('rhythm_comparison_id','rhythm_comparisons'),('reference_effect_id','effects'),('target_effect_id','effects'),
                        ('reference_comparison_id','comparisons'),('target_comparison_id','comparisons')]:
                    if row.get(field):connect(node,get(target,row[field]))
            if group=='rhythm_windows':connect(node,get('windows',row['window_id']))
            for field,target in [('cell_pair_ids','cell_pairs'),('sample_pair_ids','sample_pairs')]:
                for key in row.get(field,[]):connect(node,get(target,key))
    for row in rows('control-comparisons','unit_inventory'):
        actual=get('samples',row['unit_id'])['record']
        if row['source_run']!=source_run or row['sample']!=actual['sample'] or row['sample_confirmed']!=actual['sample_confirmed'] or set(row['movies'])!=set(actual['movies']):raise ValueError('Control analysis changed original sample membership')
        actual.update(row)
    for row in rows('control-comparisons','unit_summaries'):
        unit=get('samples',row['unit_id']);unit.setdefault('summaries',[]).append(row)
        for key in row['all_effect_ids']:connect(unit,get('effects',key))
    for row in rows('control-comparisons','comparison_members'):
        comparison=get('control_comparisons',row['comparison_id']);unit=get('samples',row['unit_id']);connect(comparison,unit)
        comparison.setdefault('members',[]).append(row)
    for row in rows('control-comparisons','matches'):get('control_comparisons',row['comparison_id']).setdefault('matching',[]).append(row)
    family_specs=[('response-evidence','families','effects'),('control-comparisons','families','control_comparisons'),
        ('rhythm-changes','window_families','rhythm_windows'),('rhythm-changes','direct_families','direct_changes'),('coordinated-responses','families','associations')]
    for step,table,group in family_specs:
        for row in rows(step,table):
            family=add('families',group+'-'+row['family_id'],{**row,'member_namespace':group},path_for(step,table))
            for key in row['members']:connect(family,get(group,key))
    pages=[];entries=[];displays={}
    original_traces={r['observation_id']:r for r in rows('aligned-windows','traces')}
    original_sample_changes={(r['definition_id'],r['unit_id']):r for r in rows('control-comparisons','unit_summaries')}
    kind_map={'window':('windows','window_id'),'comparison':('comparisons','comparison_id'),'effect':('effects','effect_id'),
        'cell_effect':('effects','effect_id'),'timing':('timing','timing_id'),'rhythm_window':('rhythm_windows','window_id'),
        'rhythm_change':('direct_changes','direct_id'),'control_comparison':('control_comparisons','comparison_id'),
        'cell_pair':('cell_pairs','pair_id'),'sample_pair':('sample_pairs','sample_pair_id'),'association':('associations','association_id')}
    for step in RENDERS:
        manifest=metadata(step,'intervention_display_manifest.json');provenance=metadata(step,'intervention_display_provenance.json')
        if not manifest:continue
        if 'source_outcomes' in manifest and manifest['source_outcomes']!=provenance['source_outcomes']:raise ValueError('Figure receipts disagree on their scientific outcomes')
        for name,outcome in provenance['source_outcomes'].items():
            if name not in inventory or outcome['scientific_id']!=inventory[name]['scientific_id']:raise ValueError('Figure uses another original scientific outcome')
        displays[step]={'settings':manifest.get('settings',{}),'provenance':provenance};local={};seen={}
        for page in manifest['pages']:
            master=page['master']
            if master in local:raise ValueError('Repeated figure target')
            key='figure-'+content_id({'step':step,'scientific_id':inventory[step]['scientific_id'],'master':master})
            plotted=page.get('plotted_entry_ids',page['entry_ids'])
            if set(plotted)-set(page['entry_ids']):raise ValueError('Plotted entries escape the original page')
            local[master]={'id':key,'step':step,'title':page['title'],'view':page['view'],'path':path_for(step,master),
                'semantics':page,'displayed_targets':[],'context_targets':[],
                'data':[{'label':label,'path':path_for(step,table_name(master,kind,lambda name:artifact(step,name)))} for kind,label in [('values','Exact plotted values'),('statistics','Saved statistics'),('display','Display settings')]]}
            pages.append(local[master]);seen[master]=set()
        for row in _json_value(read_table(artifact(step,'entries.json')).to_dict('records')):
            page=local.get(row['master'])
            if page is None or row['entry_id'] not in page['semantics']['entry_ids'] or row['entry_id'] in seen[row['master']]:raise ValueError('Figure entry lost its exact page membership')
            seen[row['master']].add(row['entry_id']);targets=[];kind=row['kind']
            additional=original_traces.get(row.get('observation_id')) if kind=='trace' else original_sample_changes.get((row.get('definition_id'),row.get('unit_id'))) if kind=='sample_change' else None
            if kind in {'trace','sample_change'} and additional is None:raise ValueError('Figure lost its original observation or sample change')
            if additional:
                for field,value in additional.items():
                    if field in row and row[field]!=value:raise ValueError('Figure changed original '+kind+' meaning: '+field)
            if kind in kind_map:
                group,field=kind_map[kind]
                if row.get(field):
                    node=get(group,row[field]);actual=node['record']
                    # Additional display labels/definition fields are allowed;
                    # original values, probabilities and identities cannot change.
                    for field,value in actual.items():
                        if field in row and row[field]!=value:raise ValueError('Figure changed original '+group+' meaning: '+field)
                    targets.append(node['id'])
            if all(row.get(k) is not None for k in KEYS):targets.append(get('cells',cell_key(row))['id'])
            if row.get('unit_id'):targets.append(get('samples',row['unit_id'])['id'])
            # Sample dots represent their recorded member cells; association
            # context is linked to all its samples without claiming they appear.
            if kind=='sample_change':
                for key in row.get('all_effect_ids',[]):targets.append(get('effects',key)['id'])
            queue=list(targets);expanded=set()
            while queue:
                target=queue.pop()
                if target in expanded:continue
                expanded.add(target);actual=nodes[target]['record'];members=[]
                if actual.get('unit_id'):members.append(get('samples',actual['unit_id'])['id'])
                if all(actual.get(k) is not None for k in KEYS):members.append(get('cells',cell_key(actual))['id'])
                for field,group in [('sample_pair_ids','sample_pairs'),('cell_pair_ids','cell_pairs')]:
                    members.extend(get(group,key)['id'] for key in actual.get(field,[]))
                if target.startswith('control_comparisons-'):members.extend(get('samples',r['unit_id'])['id'] for r in nodes[target].get('members',[]))
                targets.extend(members);queue.extend(members)
            for target in list(dict.fromkeys(targets)):
                if target.startswith('cells-'):
                    targets.extend(key for key in nodes[target]['links'] if key.startswith(('recordings-','samples-')))
            targets=list(dict.fromkeys(targets));plotted=row['entry_id'] in page['semantics'].get('plotted_entry_ids',page['semantics']['entry_ids'])
            field='displayed_targets' if plotted else 'context_targets';page[field].extend(targets)
            for target in targets:nodes[target]['pages' if plotted else 'context_pages'].append(page['id'])
            entries.append({'entry_id':row['entry_id'],'page':page['id'],'kind':kind,'plotted':plotted,'targets':targets,
                'observation_id':row.get('observation_id'),'hours':row.get('hours')})
        for master,page in local.items():
            if seen[master]!=set(page['semantics']['entry_ids']):raise ValueError('Figure omitted original semantic entries')
            for field in ['displayed_targets','context_targets']:page[field]=list(dict.fromkeys(page[field]))
    for node in nodes.values():
        for field in ['pages','context_pages']:node[field]=list(dict.fromkeys(node[field]))
    navigation={'schema_version':1,'source_run':source_run,'analysis_recomputed':False,'requested_settings':requested,'original_prepared_settings':original,
        **{group:list(members.values()) for group,members in collections.items()},'pages':pages,'entries':entries,'display_settings':displays,'inputs':inventory,
        'original_scientific_provenance':{step:value['provenance'] for step,value in data.items()},
        'selections':{step:[selection.as_dict() for selection in value.outcome.selections] for step,value in saved.items()},
        'source_policy':'Copied numerical evidence, original execution receipts and complete figure bundles move with this report. External original image stacks are not copied or re-analysed.'}
    from pymicroglia.pipelines.intervention.index_html import document
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
    return StepResult(context.step.name,context.scientific_id,'completed','Saved portable original cell, window, sample, pair, family and exact figure navigation',refs,
        provenance=Settings({'analysis_recomputed':False,'source_run':navigation['source_run']}))
