"""Readable, searchable local navigation without chart drawing or scientific calls."""
from pymicroglia._results import workings_link
import json
from pymicroglia.pipelines.audit.index import escaped, link

GROUPS=[('cells','Every original cell'),('recordings','Recordings and anchors'),('samples','Biological samples'),
    ('windows','Prepared measurement windows'),('declared_windows','Requested windows awaiting preparation'),('comparisons','Before/after comparisons'),
    ('effects','Within-cell change evidence'),('timing','Observed response and recovery'),('rhythm_windows','Independent window rhythms'),
    ('rhythm_comparisons','Before/after rhythm detection'),('direct_changes','Direct rhythm parameter changes'),
    ('control_comparisons','Treatment/control sample contrasts'),('cell_pairs','Paired measurements within cells'),('declared_pairs','Requested measurement pairs awaiting results'),
    ('sample_pairs','Paired biological-sample changes'),('associations','Biological-sample associations'),('families','Complete test families')]


def label(node):
    row=node['record'];parts=[]
    if node['id'].startswith('samples-'):parts.append(('Biological sample ' if row['sample_confirmed'] else 'Unconfirmed recording ')+str(row['sample'] or ', '.join(row['movies'])))
    else:
        if row.get('movie'):parts.append(row['movie'])
        if row.get('identity') is not None:parts.append('cell '+str(row['identity']))
        sample=row.get('sample');sample=sample.get('sample') if isinstance(sample,dict) else sample
        if sample:parts.append('sample '+str(sample))
        if row.get('measurement'):parts.append(row['measurement'])
        if isinstance(row.get('reference'),str):parts.append(row['reference']+' / '+str(row.get('target','')))
        if row.get('baseline') and row.get('target_window'):parts.append(row['baseline']+' to '+row['target_window'])
        elif row.get('window'):parts.append(row['window'])
        elif row.get('name'):parts.append(row['name'])
        if row.get('property'):parts.append(row['property'])
        if row.get('member_namespace'):parts.append(row['member_namespace'].replace('_',' ')+' | '+str(row.get('scope','all')))
        if row.get('condition'):parts.append(row['condition'])
    return ' | '.join(parts) or 'Original saved result'


def detail(title,value):return '<details><summary>'+escaped(title)+'</summary><pre>'+escaped(json.dumps(value,ensure_ascii=False,indent=2))+'</pre></details>'


def document(data,title):
    nodes={row['id']:row for group,_ in GROUPS for row in data[group]};pages={row['id']:row for row in data['pages']}
    prefixes={'windows':'Window','comparisons':'Window comparison','effects':'Within-cell evidence','rhythm_windows':'Window rhythm',
        'rhythm_comparisons':'Rhythm detection comparison','direct_changes':'Direct rhythm change','control_comparisons':'Sample control contrast',
        'cell_pairs':'Measurement pair','sample_pairs':'Sample measurement pair','associations':'Sample association','families':'Test family','timing':'Observed timing'}
    def target_label(key):
        if key not in nodes:return pages[key]['title']+' | '+pages[key]['view']
        prefix=prefixes.get(key.split('-',1)[0]);return (prefix+' — ' if prefix else '')+label(nodes[key])
    def targets(ids):return '<ul>'+''.join('<li>'+link('#'+key,target_label(key))+'</li>' for key in dict.fromkeys(ids))+'</ul>'
    def entity(node):
        row=node['record'];outcome=next((str(row[k]).replace('_',' ') for k in ['outcome','joint_response','detection_outcome','status','window_status'] if row.get(k) is not None),'')
        # Values are saved fields, never estimates recalculated for the report.
        fields=[(key,row[key]) for key in ['absolute_change','estimate','effect','interval_low','interval_high','p_value','q_value','period_hours','period_supported','amplitude',
            'reference_value','target_value','response_observed','recovery_observed','response_delay_hours','recovery_delay_hours','requested','tested'] if key in row]
        values='<dl>'+''.join('<dt>'+escaped(k.replace('_',' '))+'</dt><dd>'+escaped('Unavailable' if v is None else v)+'</dd>' for k,v in fields)+'</dl>' if fields else ''
        reason=next((row[k] for k in ['reason','eligible_reason','transition_meaning','meaning'] if row.get(k)),'')
        return '<details class="searchable" id="'+node['id']+'"><summary>'+escaped(label(node))+(' — '+escaped(outcome) if outcome else '')+'</summary><p>'+escaped(reason)+'</p>'+values+(
            '<p>'+link(node['source_table'],'Complete original result table')+'</p>' if node['source_table'] else '')+(
            '<h4>Figures displaying this entry or its original members</h4>'+targets(node['pages']) if node['pages'] else '<p>No saved figure displays this entry; its original inventory and outcome remain available.</p>')+(
            '<h4>Figures carrying related statistical context</h4><p>These links do not mean this entry is pictured on that page.</p>'+targets(node['context_pages']) if node['context_pages'] else '')+(
            '<details><summary>Related original cells, windows, samples and evidence</summary>'+targets(node['links'])+'</details>' if node['links'] else '')+detail('Exact saved values and membership',{k:v for k,v in node.items() if k not in {'links','pages','context_pages'}})+'</details>'
    sections=[]
    for group,heading in GROUPS:
        if not data[group] and group not in {'cells','samples','cell_pairs','effects'}:continue
        sections.append('<section><h2 id="'+group+'">'+heading+' ('+str(len(data[group]))+')</h2>'+(''.join(entity(row) for row in data[group]) or '<p>No saved result was produced for this question. Its requested settings and original execution outcome remain below.</p>')+'</section>')
    gallery=[]
    for page in pages.values():
        gallery.append('<section class="figure" id="'+page['id']+'"><h3>'+escaped(page['title'])+'</h3><p>'+link(page['path'],'Open saved figure')+' | '+' | '.join(link(item['path'],item['label']) for item in page['data'])+'</p><a href="'+escaped(page['path'])+'"><img loading="lazy" src="'+escaped(page['path'])+'" alt="'+escaped(page['title'])+'"></a><details><summary>Exact entries displayed on this page</summary>'+targets(page['displayed_targets'])+'</details>'+(
            '<details><summary>Related statistical context, separately from pictured entries</summary>'+targets(page['context_targets'])+'</details>' if page['context_targets'] else '')+detail('Original page settings and complete semantic membership',page['semantics'])+'</section>')
    branches=[]
    for step,row in data['inputs'].items():
        branches.append('<details><summary>'+escaped(step.replace('-',' ')+' — '+row['status'])+'</summary><p>'+escaped(row['reason'])+'</p><p>'+link(row['record'],'Original execution outcome')+'</p><ul>'+''.join('<li>'+link(item['path'],name)+'</li>' for name,item in row['artifacts'].items())+'</ul></details>')
    nav=[(group,heading) for group,heading in GROUPS if data[group] or group in {'cells','samples','cell_pairs','effects'}]+[('figures','Saved figures'),('settings','Settings and sources')]
    selected=sum(bool(row['record'].get('response_supported')) for row in data['effects'])
    failed=[name for name,item in data['inputs'].items() if item['status']=='failed']
    intro=str(len(data['cells']))+' original cells in '+str(len(data['recordings']))+' recordings; '+str(selected)+' supported within-cell measurement comparisons.'
    if failed:intro+=' Some producers failed. This report preserves their outcomes and does not establish a successful analysis.'
    return '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'''+escaped(title)+'''</title><style>
body{font:16px/1.5 system-ui,sans-serif;color:black;background:whitesmoke;max-width:1180px;margin:auto;padding:28px}h1,h2,h3{line-height:1.25}nav{display:flex;flex-wrap:wrap;gap:12px 20px;padding:16px 0;border-bottom:1px solid whitesmoke}a{color:steelblue}summary{cursor:pointer;font-weight:600}details,.figure{margin:14px 0;padding:16px;background:white;border:1px solid whitesmoke;border-radius:8px}details details{background:whitesmoke}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:32rem;overflow:auto;font-size:13px}img{max-width:100%;height:auto}input{font:inherit;padding:10px;width:min(95%,650px)}dl{display:grid;grid-template-columns:minmax(100px,230px) 1fr;gap:3px 14px}dd{margin:0;overflow-wrap:anywhere}li{overflow-wrap:anywhere}[hidden]{display:none!important}.searchable,.figure{scroll-margin-top:15px}</style></head><body><h1>'''+escaped(title)+'</h1><p>'+escaped(intro)+'''</p><p>Observed changes, conditional model evidence, rhythm estimates and data sufficiency remain separate. An inconclusive result is not evidence of no response; a change in rhythm significance alone is not a detected parameter change. Before/after associations do not establish causality.</p><nav>'''+''.join(link('#'+key,text) for key,text in nav)+'''</nav><p><label for="filter">Find an original recording, cell, sample, measurement or window</label></p><input id="filter" type="search" placeholder="Search complete entry labels">'''+''.join(sections)+'''<h2 id="figures">Saved figures</h2>'''+(''.join(gallery) or '<p>No saved figure is available; original execution outcomes remain below.</p>')+'''<h2 id="settings">Settings, provenance and original execution</h2><p>'''+escaped(data['source_policy'])+'</p>'+detail('Current resolved request',data['requested_settings'])+detail('Original prepared settings',data['original_prepared_settings'])+detail('Original scientific provenance and methods',data['original_scientific_provenance'])+detail('Complete original scientific selections',data['selections'])+''.join(branches)+'<p>'+link(workings_link('navigation.json'),'Complete portable navigation inventory')+'''</p><script>
function reveal(){let e=document.getElementById(decodeURIComponent(location.hash.slice(1)));for(let p=e;p;p=p.parentElement){p.hidden=false;if(p.tagName==='DETAILS')p.open=true}if(e)e.scrollIntoView()}
document.getElementById('filter').addEventListener('input',function(){let q=this.value.toLowerCase();document.querySelectorAll('.searchable').forEach(e=>{e.hidden=!e.querySelector('summary').textContent.toLowerCase().includes(q)})});addEventListener('hashchange',reveal);addEventListener('DOMContentLoaded',reveal);
</script></body></html>'''
