"""Portable, model-scoped navigation over immutable saved state evidence."""
from pymicroglia._results import workings_link
from pymicroglia._results import report_name, read_document
from .._saved_figures import table_name
from pymicroglia._sources import source_file
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

import pandas as pd

from pymicroglia.pipelines.audit.index import copy_evidence, inventory_artifact
from pymicroglia.pipelines._contracts import ArtifactRef, Settings, StepResult, cell_number, content_id, result_to_dict
from pymicroglia.pipelines.relationships.inputs import KEYS
from pymicroglia.pipelines.rhythm.index import escaped, link
from pymicroglia.pipelines._screening import _json_value, _write_json, file_hash, read_table

MANIFESTS = {'state-support-figures':'state_display_manifest.json','state-profile-figures':'state_display_manifest.json',
    'state-timelines':'state_timelines_manifest.json','state-switching-figures':'state_summaries_manifest.json','state-report-cards':'state_cards_manifest.json'}
LABELS = {'behaviour-design':'Requested features and observation policy','feature-inputs':'Original observations and protected validation groups',
    'candidate-models':'Fitted candidates and learning transformations','state-support':'Independent state-support decision',
    'state-assignments':'Frozen assignments and observed state profiles','durations-and-switches':'Observed time, unique bouts and transitions',
    'state-sample-comparisons':'Complete samples and declared contrasts','state-support-figures':'Support diagnostics',
    'state-profile-figures':'Observed state profiles','state-timelines':'Original-clock cell timelines',
    'state-switching-figures':'Time and sample summaries','state-report-cards':'Real-member state cards'}


def version():
    return content_id({name:file_hash(source_file(name)) for name in ['behaviour_index.py','audit_index.py','rhythm_index.py']})


def cell_key(row):return {**{name:row[name] for name in KEYS[:-1]},'identity':cell_number(row['identity'])}
def cell_id(row,model):return 'cell-'+content_id({'model_id':model,**cell_key(row)})
def state_id(row):return 'state-'+content_id({'model_id':row['model_id'],'state_id':row['state_id']})


def build(output,inventory,requested,title='Cell behaviour states'):
    output=Path(output)
    def source(step,name):
        ref=inventory_artifact(output,inventory,step,name)
        if ref is None:return None
        path=(output/ref['path']).resolve()
        if not path.is_relative_to(output.resolve()) or not path.is_file() or file_hash(path)!=ref['sha256']:
            raise ValueError('Missing or changed state index evidence: '+step+'/'+name)
        return path
    def metadata(step,name):
        path=source(step,name);return read_document(path) if path else {}
    def table(step,name):
        path=source(step,name);return read_table(path) if path else pd.DataFrame()
    def path_for(step,name):
        if source(step,name) is None:raise ValueError('State target references an absent artifact: '+step+'/'+name)
        return inventory_artifact(output,inventory,step,name)['path']
    def science_id(step):return inventory.get(step,{}).get('scientific_id')
    original=metadata('feature-inputs','provenance').get('resolved_request',{})
    if original and original.get('inputs')!=requested['inputs']:raise ValueError('State index inputs differ from original prepared observations')
    decision=metadata('state-support','support_decision')
    model=decision.get('accepted_model_id')
    if model and decision.get('status')!='accepted':raise ValueError('An unaccepted state decision cannot supply a vocabulary')
    provenance={step:metadata(step,'provenance') for step in ['state-assignments','durations-and-switches','state-sample-comparisons']}
    for step,record in provenance.items():
        if record and (not model or record.get('model_id')!=model):raise ValueError('State evidence uses a different accepted model')
    for step,field,target in [('state-assignments','feature_input_id','feature-inputs'),('state-assignments','support_id','state-support'),
        ('durations-and-switches','assignment_id','state-assignments'),('state-sample-comparisons','duration_id','durations-and-switches')]:
        if provenance[step].get(field) and provenance[step][field]!=science_id(target):raise ValueError('State evidence belongs to a different prerequisite result')
    definitions=table('state-assignments','state_definitions')
    states={}
    for row in _json_value(definitions.to_dict('records')):
        if not model or row['model_id']!=model:raise ValueError('State definition uses another vocabulary')
        if row['state_id'] in states:raise ValueError('Duplicate state definition')
        states[row['state_id']]={**row,'id':state_id(row),'pages':[],'examples':[],'bouts':[]}
    cells={cell_id(row,model):{**cell_key(row),'model_id':model,'id':cell_id(row,model),'pages':[],'bouts':[],'samples':[],'examples':[],
        'time':{},'occupancy':[]} for row in requested['inputs']['cells']}
    if len(cells)!=len(requested['inputs']['cells']):raise ValueError('Requested cells repeat full source identities')
    def checked_cell(row):
        key=cell_id(row,model)
        if key not in cells:raise ValueError('State evidence names an unknown full cell identity')
        if row.get('model_id',model)!=model:raise ValueError('Cell evidence uses a different accepted model')
        return key
    def checked_state(value):
        if value not in states:raise ValueError('State evidence names an unknown accepted state')
        return value
    assignments=table('state-assignments','assignments');observations={}
    for row in _json_value(assignments.to_dict('records')):
        checked_cell(row)
        if row['observation_id'] in observations:raise ValueError('Repeated original assignment observation')
        if row.get('state_id') is not None:checked_state(row['state_id'])
        observations[row['observation_id']]=row
    for name,target in [('cell_statistics','time'),('occupancy','occupancy')]:
        for row in _json_value(table('durations-and-switches',name).to_dict('records')):
            cell=cells[checked_cell(row)]
            if target=='time':cell[target]=row
            else:checked_state(row['state_id']);cell[target].append(row)
    bouts={}
    for row in _json_value(table('durations-and-switches','bouts').to_dict('records')):
        cell=checked_cell(row);state=checked_state(row['state_id'])
        if row['bout_id'] in bouts:raise ValueError('Repeated unique saved bout')
        for oid in row['observation_ids']:
            member=observations.get(oid)
            if member is None or checked_cell(member)!=cell or member['state_id']!=state:raise ValueError('Bout contains an observation from another cell or state')
        key='bout-'+content_id({'model_id':model,'bout_id':row['bout_id'],**cell_key(row)})
        bouts[row['bout_id']]={**row,'id':key,'cell':cell,'state':states[state]['id'],'pages':[]}
        cells[cell]['bouts'].append(key);states[state]['bouts'].append(key)
    samples={};unit_metrics=table('state-sample-comparisons','unit_metrics');comparisons=table('state-sample-comparisons','comparisons')
    for row in _json_value(table('state-sample-comparisons','unit_inventory').to_dict('records')):
        if row['model_id']!=model or row['source_run']!=requested['inputs']['source_run']:raise ValueError('Experimental unit belongs to another model or source')
        if row['unit_id'] in samples:raise ValueError('Duplicate experimental-unit identity')
        members=[checked_cell(member) for member in row['members']]
        key='sample-'+content_id({'model_id':model,'source_run':row['source_run'],'unit_id':row['unit_id']})
        samples[row['unit_id']]={**row,'id':key,'cell_targets':members,'pages':[],
            'metrics':_json_value(unit_metrics.loc[unit_metrics.unit_id.eq(row['unit_id'])].to_dict('records'))}
        for cell in members:cells[cell]['samples'].append(key)
    pages=[];examples=[];entries=[]
    for step,manifest_name in MANIFESTS.items():
        manifest=metadata(step,manifest_name)
        if manifest.get('model_id') and manifest['model_id']!=model:raise ValueError('State figure manifest uses another model')
        if manifest.get('decision_id') and manifest['decision_id']!=decision.get('decision_id'):raise ValueError('State figure refers to a different support decision')
        local={}
        for page in manifest.get('pages',[]):
            master=page['master']
            if master in local:raise ValueError('Repeated saved state figure page')
            key='figure-'+content_id({'step':step,'master':master,'scientific_id':science_id(step)})
            value={'id':key,'step':step,'master':master,'path':path_for(step,master),'semantics':page,'cells':[],'states':[],'samples':[],
                'title':LABELS[step]+' | '+str(page.get('view','cells' if step=='state-timelines' else 'state card')).replace('_',' ')+' | page '+str(page.get('page',len(local)+1)),'data':[]}
            for kind,label in [('values','Plotted values'),('statistics','Saved statistics'),('display','Display settings')]:
                name=table_name(master,kind,lambda filename:source(step,filename))
                if source(step,name):value['data'].append({'label':label,'path':path_for(step,name)})
            member_cells=[checked_cell(row) for row in page.get('cells',[])]
            member_states=[checked_state(state) for state in page.get('states',[])]
            if page.get('state_id'):member_states.append(checked_state(page['state_id']))
            for pair in page.get('pairs',[]):member_states.extend([checked_state(pair['source']),checked_state(pair['target'])])
            if step=='state-switching-figures' and page.get('view')=='samples':
                for unit in samples.values():
                    if any(row['question_id'] in page['questions'] for row in unit['metrics']):value['samples'].append(unit['unit_id'])
            if step=='state-switching-figures' and page.get('view')=='contrasts':
                selected=comparisons.loc[comparisons.comparison_id.isin(page['comparisons'])]
                if set(selected.comparison_id)!=set(page['comparisons']):raise ValueError('Contrast page names an unavailable saved comparison')
                for contrast in _json_value(selected.to_dict('records')):
                    for unit in [*contrast['eligible_units'],*[row['unit_id'] for row in contrast['excluded_units']]]:
                        if unit not in samples:raise ValueError('Contrast names an unknown experimental unit')
                        value['samples'].append(unit)
                value['samples']=list(dict.fromkeys(value['samples']))
            value['cells']=list(dict.fromkeys(member_cells));value['states']=list(dict.fromkeys(member_states))
            local[master]=value;pages.append(value)
        for entry in _json_value(table(step,'entries.json').to_dict('records')):
            page=local.get(entry['master'])
            if page is None:raise ValueError('State entry has no saved figure page')
            if entry.get('model_id') and step!='state-support-figures' and entry['model_id']!=model:raise ValueError('State entry uses another model')
            for field,target in [('assignment_id','state-assignments'),('time_id','durations-and-switches'),('sample_id','state-sample-comparisons')]:
                if entry.get(field) and entry[field]!=science_id(target):raise ValueError('State entry refers to a different scientific result')
            if step=='state-timelines':
                if entry['source_scientific_id']!=science_id('durations-and-switches') or checked_cell(entry) not in page['cells']:raise ValueError('Timeline target points at another cell or time analysis')
            if step=='state-profile-figures':
                if entry.get('source_scientific_id')!=science_id('state-assignments'):raise ValueError('Profile target points at a different saved assignment result')
                if entry.get('state_id') and entry['state_id'] not in page['states']:raise ValueError('Profile target points at a different state page')
            if step=='state-switching-figures':
                for field in ['view','cells','states','pairs','spacing','questions','comparisons']:
                    if field in page['semantics'] and entry.get(field)!=page['semantics'][field]:raise ValueError('Summary target points at a different saved question or population')
            if step=='state-report-cards':
                if entry['state_id']!=page['semantics']['state_id']:raise ValueError('Card entry points at another accepted state')
                if set(entry['example_ids'])!=set(page['semantics']['example_ids']):raise ValueError('Card page and entry examples differ')
                if set(entry['example_ids'])!={row['example_id'] for row in entry['examples']}:raise ValueError('Card entry omits or substitutes a declared example')
                for example in entry['examples']:
                    actual=observations.get(example['observation_id'])
                    if actual is None or any(actual[name]!=example[name] for name in ['model_id','state_id',*KEYS,'frame_index','hours']):raise ValueError('Card example disagrees with its exact saved observation')
                    if example['state_id']!=entry['state_id'] or actual['status']!='assigned':raise ValueError('Card example is not assigned to the displayed state')
                    cell=checked_cell(example);page['cells'].append(cell)
                    bout=bouts.get(example.get('bout_id'))
                    if example.get('bout_id') and (bout is None or example['observation_id'] not in bout['observation_ids']):raise ValueError('Card example points at another unique bout')
                    image=next((row for row in manifest.get('images',{}).get('examples',[]) if row['example_id']==example['example_id']),None)
                    if image is None:raise ValueError('State card has no recorded image availability')
                    if any(image[name]!=example[name] for name in ['model_id','state_id','observation_id',*KEYS,'frame_index','hours']):raise ValueError('State card image belongs to a different observation')
                    if image['image_status']=='available' and (len(image['tiles'])!=1 or image['tiles'][0]['hours']!=example['hours'] or image['tiles'][0]['frame_index']!=example['frame_index']):raise ValueError('State image frame is not the exact selected observation')
                    key='example-'+example['example_id']
                    existing=next((row for row in examples if row['id']==key),None)
                    if existing is None:
                        existing={**example,'id':key,'cell':cell,'state':states[example['state_id']]['id'],'bout':bout['id'] if bout else None,'image':image,'pages':[],
                            'frozen_images':path_for(step,'state_example_tiles.npz'),'image_inventory':path_for(step,'state_example_images.json')}
                        examples.append(existing);cells[cell]['examples'].append(key);states[example['state_id']]['examples'].append(key)
                    existing['pages'].append(page['id'])
                    if bout:bout['pages'].append(page['id'])
            entries.append({**entry,'step':step,'page':page['id'],'path':page['path']})
        for page in local.values():
            page['cells']=list(dict.fromkeys(page['cells']))
            for cell in page['cells']:cells[cell]['pages'].append(page['id'])
            for state in page['states']:states[state]['pages'].append(page['id'])
            for unit in page['samples']:samples[unit]['pages'].append(page['id'])
            if step=='state-timelines' or (step=='state-switching-figures' and page['semantics'].get('view')=='bouts'):
                for bout in bouts.values():
                    if bout['cell'] in page['cells']:bout['pages'].append(page['id'])
    navigation={'schema_version':1,'analysis_recomputed':False,'source_run':requested['inputs']['source_run'],'model_id':model,'decision':decision,
        'states':list(states.values()),'cells':list(cells.values()),'bouts':list(bouts.values()),'samples':list(samples.values()),'examples':examples,'pages':pages,'entries':entries,
        'comparisons':_json_value(comparisons.to_dict('records')),'inputs':inventory,'requested_settings':requested,'original_prepared_settings':original,'scientific_provenance':provenance,
        'image_portability':'Recorded exact crops and masks are bundled. Full original image stacks remain external references and are not promised to move with this report.'}
    _write_json(output/'navigation.json',navigation)
    (output/'index.html').write_text(document(navigation,title),encoding='utf-8')
    return navigation


def document(data,title):
    by_page={row['id']:row for row in data['pages']}
    def details(label,value):return '<details><summary>'+escaped(label)+'</summary><pre>'+escaped(json.dumps(value,ensure_ascii=False,indent=2))+'</pre></details>'
    def page_links(keys):return '<ul>'+''.join('<li>'+link('#'+key,by_page[key]['title'])+'</li>' for key in dict.fromkeys(keys))+'</ul>'
    def target_links(keys,label):return ' | '.join(link('#'+key,label+' '+str(i+1)) for i,key in enumerate(dict.fromkeys(keys)))
    states=[]
    for row in data['states']:
        primary=[key for key in row['pages'] if by_page[key]['step'] in {'state-profile-figures','state-report-cards'}]
        other=[key for key in row['pages'] if key not in primary]
        states.append('<section id="'+row['id']+'"><h3>'+escaped(row['label'])+'</h3><p>'+escaped(str(row['assigned_observations'])+' assigned observations; '+str(row['cells'])+' cells; '+str(row['movies'])+' recordings; '+str(row['confirmed_samples'])+' confirmed samples.')+'</p>'
            +page_links(primary)+target_links(row['examples'],'Recorded example')+'<details><summary>Time and sample evidence containing this state</summary>'+page_links(other)+'</details>'
            +details('Complete saved state definition and population counts',row)+'</section>')
    cell_rows=[];cells=[];by_bout={row['id']:row for row in data['bouts']}
    for row in data['cells']:
        label=row['movie']+' / cell '+str(row['identity']);time=row['time']
        cell_rows.append('<tr><td>'+link('#'+row['id'],label)+'</td><td>'+escaped(time.get('status','No accepted time result'))+'</td><td>'+escaped(time.get('observed_hours','Unavailable'))+'</td><td>'+escaped(time.get('unknown_hours','Unavailable'))+'</td></tr>')
        bout_rows=[]
        for key in row['bouts']:
            bout=by_bout[key]
            bout_rows.append('<details id="'+key+'"><summary>'+escaped('Observed bout: '+str(bout.get('first_hours',bout.get('start_hours','')))+'; '+bout['duration_status'])+'</summary>'
                +link('#'+bout['state'],'Accepted state')+page_links(bout['pages'])+details('Exact unique bout, original member observations and censoring',bout)+'</details>')
        cells.append('<details id="'+row['id']+'"><summary>'+escaped(label)+'</summary>'+page_links(row['pages'])+target_links(row['examples'],'Recorded example')+'<p>'+target_links(row['samples'],'Experimental unit')+'</p>'
            +details('Full source identity, saved time denominators and occupancy',row)+''.join(bout_rows)+'</details>')
    examples=[]
    for row in data['examples']:
        examples.append('<details id="'+row['id']+'"><summary>'+escaped(row['example_role'].replace('_',' ')+' | '+row['movie']+' / cell '+str(row['identity'])+' | '+str(row['hours'])+' h')+'</summary><p>'
            +link('#'+row['state'],'Accepted state')+' | '+link('#'+row['cell'],'Original cell timeline')+(' | '+link('#'+row['bout'],'Unique observed bout') if row['bout'] else '')+'</p><p>'
            +escaped('Image: '+row['image']['image_status']+'. '+row['image'].get('image_reason',''))+'</p>'+page_links(row['pages'])+'<p>'+link(row['frozen_images'],'Frozen original crops, masks and display pixels')+' | '+link(row['image_inventory'],'Image availability and exact source-frame mapping')+'</p>'+details('Exact saved member, selection rule and source image references',row)+'</details>')
    samples=[]
    for row in sorted(data['samples'],key=lambda item:(str(item.get('condition') or ''),str(item.get('sample') or ''),item['id'])):
        label=('Biological sample ' if row['sample_confirmed'] else 'Descriptive recording ')+str(row['sample'])
        samples.append('<details id="'+row['id']+'"><summary>'+escaped(label+' | '+str(row.get('condition') or 'No condition'))+'</summary><p>'+escaped('Independent of state choice: '+str(row['independent_of_state_choice'])+'; '+str(row['cells'])+' cells from '+str(row['recordings'])+' recordings.')+'</p>'
            +page_links(row['pages'])+target_links(row['cell_targets'],'Contributing cell')+details('Saved unit values, denominators and eligible/excluded membership',row)+'</details>')
    figures=[]
    for row in data['pages']:
        figures.append('<details id="'+row['id']+'"><summary>'+escaped(row['title'])+'</summary><p>'+link(row['path'],'Open saved figure')+' | '+' | '.join(link(item['path'],item['label']) for item in row['data'])+'</p>'
            +'<img loading="lazy" src="'+quote(row['path'],safe='/')+'" alt="'+escaped(row['title'])+'">'+details('Exact saved page population and model',row)+'</details>')
    branches=[]
    for step,row in data['inputs'].items():
        branches.append('<details><summary>'+escaped(LABELS.get(step,step)+': '+row['status'])+'</summary><p>'+escaped(row['reason'])+'</p>'+link(row['record'],'Actual execution outcome and selection')+'<ul>'+''.join('<li>'+link(ref['path'],name)+'</li>' for name,ref in row['artifacts'].items())+'</ul></details>')
    decision=data['decision'];status=decision.get('status','No completed scientific decision')
    failures=[LABELS.get(name,name) for name,row in data['inputs'].items() if row['status'] in {'unavailable','failed'}]
    banner='<p class="status">Requested outputs remain unavailable or failed: '+escaped(', '.join(failures))+'. See their recorded reasons below.</p>' if failures else ''
    return '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'''+escaped(title)+'''</title>
<style>body{font:16px Arial,sans-serif;max-width:1200px;margin:30px auto;padding:0 20px;color:black}a{color:steelblue}nav{display:flex;gap:20px;flex-wrap:wrap}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;border:1px solid silver;text-align:left}details{margin:12px 0;padding:9px;border:1px solid silver}summary{cursor:pointer;font-weight:bold}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:whitesmoke;padding:10px}img{width:100%;height:auto}section{border-top:1px solid silver;padding:12px 0}.status{background:whitesmoke;padding:14px}</style></head><body><h1>'''+escaped(title)+'''</h1>'''+banner+'''<nav><a href="#support">Support decision</a><a href="#states">States</a><a href="#cells">Cells and bouts</a><a href="#examples">Recorded examples</a><a href="#samples">Samples</a><a href="#figures">Figures</a><a href="#settings">Settings</a><a href="#evidence">Execution and evidence</a></nav>
<h2 id="support">'''+escaped(status.replace('_',' '))+'''</h2><p>'''+escaped(decision.get('reason','No completed support decision is available.'))+'''</p><p>When accepted, states provide a supported shared vocabulary for these measured features. Their labels do not establish biological activation categories. Unknown assignments, observation gaps and censored bouts remain explicit.</p>'''+details('Saved validation decision, frozen model and scope',decision)+page_links([row['id'] for row in data['pages'] if row['step']=='state-support-figures'])+'''<h2 id="states">Accepted states</h2>'''+(''.join(states) or '<p>No accepted state cards or assignments are available. The scientific decision and branch reasons remain linked.</p>')+'''<h2 id="cells">Every requested cell and its observed bouts</h2><p>Times use each original recording clock. Observed, assigned and unknown hours are distinct denominators; gaps do not become time spent in a state.</p><table><tr><th>Movie / cell</th><th>Time status</th><th>Observed hours</th><th>Unknown hours</th></tr>'''+''.join(cell_rows)+'''</table>'''+''.join(cells)+'''<h2 id="examples">Exact recorded examples</h2><p>'''+escaped(data['image_portability'])+'</p>'+''.join(examples)+'''<h2 id="samples">Complete experimental units</h2><p>All cells remain represented. Formal contrasts require declared independent samples unused for choosing states. Observed bout durations retain censoring and do not estimate complete latent dwell times.</p>'''+''.join(samples)+details('Saved complete contrast family',data['comparisons'])+'''<h2 id="figures">Saved figures</h2>'''+''.join(figures)+'''<h2 id="settings">Scientific settings and provenance</h2>'''+details('Current resolved request',data['requested_settings'])+details('Original prepared request and validation population',data['original_prepared_settings'])+details('Frozen assignment, time and sample definitions',data['scientific_provenance'])+'''<h2 id="evidence">Actual execution and immutable evidence</h2>'''+''.join(branches)+'<p>'+link(workings_link('navigation.json'),'Complete portable navigation inventory')+'''</p><script>function reveal(){var e=document.getElementById(decodeURIComponent(location.hash.slice(1)));for(var p=e;p;p=p.parentElement){if(p.tagName==='DETAILS')p.open=true}if(e)e.scrollIntoView()}addEventListener('hashchange',reveal);addEventListener('DOMContentLoaded',reveal)</script></body></html>'''


def validate_links(output):
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.links=[];self.ids=[]
        def handle_starttag(self,tag,attrs):
            attrs=dict(attrs);self.links.extend(attrs[key] for key in ['href','src'] if key in attrs)
            if 'id' in attrs:self.ids.append(attrs['id'])
    output=Path(output).resolve();parser=Links();parser.feed((output/'index.html').read_text(encoding='utf-8'))
    if len(set(parser.ids))!=len(parser.ids):raise ValueError('Duplicate state navigation target')
    for value in parser.links:
        parsed=urlsplit(value)
        if parsed.scheme or parsed.netloc:raise ValueError('State index internal links must be portable relative targets')
        if parsed.path:
            path=(output/unquote(parsed.path)).resolve()
            if not path.is_relative_to(output) or not path.is_file():raise ValueError('State index link has no bundled artifact')
        if parsed.fragment and unquote(parsed.fragment) not in parser.ids:raise ValueError('State index link has no exact semantic target')
    return {'links':len(parser.links),'targets':len(parser.ids),'all_internal_links_resolve':True}


def produce(context):
    appearance=context.presentation.as_dict().get('report',{})
    if not isinstance(appearance,dict) or set(appearance)-{'title'} or not isinstance(appearance.get('title',''),str):raise ValueError('Report presentation accepts a title string')
    context.output.mkdir(parents=True,exist_ok=True)
    inventory=copy_evidence(context.dependencies,context.output)
    records=context.output/'execution-records';records.mkdir()
    for step,saved in context.dependencies.items():
        path=records/(step+'.json');path = _write_json(path,result_to_dict(saved.outcome));inventory[step]['record']=path.relative_to(context.output).as_posix()
    navigation=build(context.output,inventory,context.request.as_dict(),**appearance)
    _write_json(context.output/'link-validation.json',validate_links(context.output))
    refs=tuple(ArtifactRef(report_name(path,context.output),path.relative_to(context.output).as_posix(),file_hash(path),context.scientific_id)
        for path in sorted(context.output.rglob('*')) if path.is_file() and path.name!="artefacts.json")
    return StepResult(context.step.name,context.scientific_id,'completed','Saved portable model, state, cell, bout, sample and exact-image navigation with actual branch outcomes',refs,
        provenance=Settings({'analysis_recomputed':False,'source_run':navigation['source_run'],'model_id':navigation['model_id']}))
