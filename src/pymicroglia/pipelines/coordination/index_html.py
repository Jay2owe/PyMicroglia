"""Plain portable navigation for the saved spatial evidence inventory."""
from pymicroglia._results import workings_link
import json,math
from pymicroglia.pipelines.rhythm.index import escaped, link


def number(value):
    return f'{float(value):.4g}' if isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) else 'Unavailable'


def document(data,title):
    pages={row['id']:row for row in data['pages']};effects={row['id']:row for row in data['effects']};pairs={row['id']:row for row in data['pairs']}
    recordings={row['id']:row for row in data['recordings']};samples={row['id']:row for row in data['samples']};cells={row['id']:row for row in data['cells']}
    def detail(label,value):return '<details><summary>'+escaped(label)+'</summary><pre>'+escaped(json.dumps(value,ensure_ascii=False,indent=2))+'</pre></details>'
    def label(row):return row['movie']+' | cell '+str(row['reference_identity'])+' '+row['reference']+' → cell '+str(row['target_identity'])+' '+row['target']
    def page_links(keys):return '<ul>'+''.join('<li>'+link('#'+key,pages[key]['title']+' | '+pages[key]['view'])+'</li>' for key in dict.fromkeys(keys))+'</ul>'
    def pair_links(keys):return '<ul>'+''.join('<li>'+link('#'+key,label(pairs[key]))+'</li>' for key in dict.fromkeys(keys))+'</ul>'
    def effect_table(keys):
        rows=[]
        for key in keys:
            row=effects[key];meaning=' | '.join(str(row.get(field)) for field in ['question','representation','adjustment','evidence_kind'] if row.get(field))
            rows.append('<tr><td>'+link('#'+key,meaning)+'</td><td>'+escaped(number(row.get('effect')))+'</td><td>'+escaped(number(row.get('q_value')))+'</td><td>'+escaped(row['evidence_level'])+'</td><td>'+escaped(str(row['decision']).replace('_',' '))+'</td></tr>')
        return '<table><thead><tr><th>Saved question</th><th>Effect</th><th>Corrected q</th><th>Evidence level</th><th>Outcome</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table>' if rows else '<p>No scientific effect was produced for this inventory entry.</p>'
    selected=sum(bool(row['decision'].get('supported_effect_ids')) for row in pairs.values())
    pair_sections=[];omitted=set(data['display_omissions'].get('pair-report-cards',{}).get('selection',{}).get('omitted_pair_ids',[]))
    for row in sorted(pairs.values(),key=lambda row:(row['movie'],row['reference_identity'],row['target_identity'],row['reference'],row['target'])):
        note='Selected pair; its card is omitted by the recorded display limit.' if row['pair_id'] in omitted else 'No saved card for this pair; its numerical evidence and original outcome remain available.' if not row['card_pages'] else 'Original paired observations and saved diagnostics:'
        pair_sections.append('<details class="searchable" id="'+row['id']+'"><summary>'+escaped(label(row))+' — '+escaped(str(row['decision'].get('status','Evidence unavailable')).replace('_',' '))+'</summary><p>'+link('#'+row['recording'],'Recording and biological samples')+' | '+' | '.join(link('#'+key,'Original cell '+str(cells[key]['identity'])) for key in row['cells'])+'</p><p>'+escaped(note)+'</p>'+page_links(row['card_pages'])+effect_table(row['effects'])
            +detail('Exact pair identity, scientific decision and complete membership',{key:value for key,value in row.items() if key not in {'pages','card_pages'}})
            +'<details><summary>Overviews and maps containing this pair</summary>'+page_links([key for key in row['pages'] if key not in row['card_pages']])+'</details></details>')
    effect_sections=[]
    for row in effects.values():
        scope=link('#family-'+row['family_id'],'Complete correction family') if row.get('family_id') else 'No significance family for this descriptive result'
        effect_sections.append('<details id="'+row['id']+'"><summary>'+escaped(row['movie']+' | '+row['question']+' | '+str(row['decision']).replace('_',' '))+'</summary><p>'+escaped(row.get('decision_reason',''))+'</p><p>'+scope+'</p>'+effect_table([row['id']])+pair_links(row['pairs'])+page_links(row['pages'])
            +'<p>'+link(row['source_table'],'Complete evidence table')+' | '+link(row['original_source'],'Original branch result, including native details')+'</p>'+detail('Exact saved effect, representation, model and timing reference',row)+'</details>')
    recording_sections=[]
    for row in sorted(recordings.values(),key=lambda row:row['movie']):
        recording_sections.append('<details id="'+row['id']+'"><summary>'+escaped(row['movie'])+' — '+str(len(row['cells']))+' cells; '+str(len(row['pairs']))+' requested measurement pairs</summary><p>'
            +' | '.join(link('#'+key,('Biological sample ' if samples[key]['sample_confirmed'] else 'Unconfirmed recording ')+str(samples[key]['sample'] or row['movie'])) for key in row['samples'])+'</p>'+page_links(row['pages'])+pair_links(row['pairs'])+effect_table([key for key in row['effects'] if effects[key]['evidence_level']=='recording'])+'</details>')
    sample_sections=[]
    for row in sorted(samples.values(),key=lambda row:(str(row.get('condition') or ''),str(row.get('sample') or ''),row['id'])):
        text=('Biological sample ' if row['sample_confirmed'] else 'Descriptive recording ')+str(row['sample'] or ', '.join(row['movies']))
        sample_sections.append('<details id="'+row['id']+'"><summary>'+escaped(text+' | '+str(row.get('condition') or 'No declared condition'))+'</summary><p>'+str(len(row['movies']))+' recordings; shared recordings remain one experimental unit.</p><p>'
            +' | '.join(link('#'+key,recordings[key]['movie']) for key in row['recording_targets'])+'</p>'+page_links(row['pages'])+detail('All saved sample summaries and original membership',row)+'</details>')
    cell_sections=[]
    for row in cells.values():
        cell_sections.append('<details id="'+row['id']+'"><summary>'+escaped(row['movie']+' / cell '+str(row['identity']))+'</summary>'+pair_links(row['pairs'])+page_links(row['pages'])+detail('Original full cell identity',row)+'</details>')
    figure_sections=[]
    for row in pages.values():
        figure_sections.append('<section class="figure" id="'+row['id']+'"><h3>'+escaped(row['title'])+'</h3><p>'+link(row['path'],'Open saved figure')+' | '+' | '.join(link(item['path'],item['label']) for item in row['data'])+'</p><a href="'+escaped(row['path'])+'"><img loading="lazy" src="'+escaped(row['path'])+'" alt="'+escaped(row['title'])+'"></a>'+detail('Original figure population, omitted targets and display meaning',row['semantics'])+'</section>')
    families=[]
    for row in data['families']:
        families.append('<details id="family-'+row['family_id']+'"><summary>'+escaped(row['question']+' | '+row['evidence_level']+' | '+row['correction'])+' — '+str(row['tested'])+' tested / '+str(row['requested'])+' declared hypotheses</summary>'
            +detail('Original complete family and missing-probability policy',row)+effect_table(['effect-'+key for key in row['members']])+'</details>')
    branches=[]
    for step,row in data['inputs'].items():
        branches.append('<details><summary>'+escaped(step.replace('-',' ')+' — '+row['status'])+'</summary><p>'+escaped(row['reason'])+'</p><p>'+link(row['record'],'Original execution outcome')+'</p><ul>'
            +''.join('<li>'+link(item['path'],name)+'</li>' for name,item in row['artifacts'].items())+'</ul></details>')
    coverage='<table><tr><th>Question</th><th>Outcome</th><th>Saved effects</th><th>Declared hypotheses</th></tr>'+''.join('<tr><td>'+escaped(row['question'])+'</td><td>'+escaped(row['status'])+'</td><td>'+str(row.get('effect_rows',0))+'</td><td>'+str(row.get('formal_hypotheses',0))+'</td></tr>' for row in data['branches'])+'</table>'
    return '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'''+escaped(title)+'''</title><style>
body{font:16px/1.5 system-ui,sans-serif;color:black;background:whitesmoke;max-width:1180px;margin:auto;padding:28px}h1,h2,h3{line-height:1.25}nav{display:flex;flex-wrap:wrap;gap:18px;padding:16px 0;border-bottom:1px solid whitesmoke}a{color:steelblue}summary{cursor:pointer;font-weight:600}details,section.figure{margin:14px 0;padding:16px;background:white;border:1px solid whitesmoke;border-radius:8px}details details{background:whitesmoke}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:32rem;overflow:auto;font-size:13px}table{border-collapse:collapse;width:100%;margin:15px 0}th,td{text-align:left;vertical-align:top;padding:9px;border-bottom:1px solid whitesmoke}img{max-width:100%;height:auto}input{font:inherit;padding:10px;width:min(100%,550px)}.intro{max-width:900px}.figure{scroll-margin-top:15px}[hidden]{display:none!important}</style></head><body><h1>'''+escaped(title)+'''</h1><p class="intro">'''+str(selected)+' supported pairs from '+str(len(pairs))+' requested measurement pairs across '+str(len(recordings))+''' recordings. Pair, recording and biological-sample evidence remain separate. Unsupported and unavailable results stay in the complete inventory.</p><nav>'''+''.join(link('#'+key,name) for key,name in [('overview','Evidence overview'),('pairs','Every pair'),('recordings','Recordings'),('samples','Biological samples'),('cells','Original cells'),('figures','Saved figures'),('families','Test families'),('settings','Settings and sources')])+'''</nav><h2 id="overview">What the saved analysis supports</h2><p>Proximity and association do not establish communication or causality. A negative association is not automatically antiphase. Rhythm timing requires its own saved period-comparability evidence.</p>'''+coverage+page_links([row['id'] for row in pages.values() if row['step']=='coordination-overview'])+'''<h2 id="pairs">Every requested pair</h2><label for="filter">Find a recording, cell or measurement</label><p><input id="filter" type="search" placeholder="Search pair labels"></p>'''+(''.join(pair_sections) or '<p>No prepared pair inventory is available. Actual execution reasons and requested inputs remain below.</p>')+'''<h2 id="recordings">Original recordings</h2>'''+''.join(recording_sections)+'''<h2 id="samples">Biological samples and complete effects</h2><p>Pairs sharing cells are not independent samples. Available summaries include non-significant and untestable effects where their saved numeric value is usable.</p>'''+''.join(sample_sections)+detail('Complete saved sample comparisons',data['sample_comparisons'])+'''<h2 id="cells">Original cells</h2>'''+''.join(cell_sections)+'''<h2 id="figures">Saved figures</h2>'''+''.join(figure_sections)+'''<h2 id="families">Scientific effects and declared families</h2>'''+''.join(families)+''.join(effect_sections)+'''<h2 id="settings">Settings, provenance and actual execution</h2><p>'''+escaped(data['external_source_policy'])+'</p>'+detail('Current resolved request',data['requested_settings'])+detail('Original prepared settings',data['original_prepared_settings'])+detail('Complete scientific selections',data['selections'])+detail('Presentation limits and omitted pairs',data['display_omissions'])+''.join(branches)+'<p>'+link(workings_link('navigation.json'),'Complete portable navigation inventory')+'''</p><script>
function reveal(){let e=document.getElementById(decodeURIComponent(location.hash.slice(1)));for(let p=e;p;p=p.parentElement){p.hidden=false;if(p.tagName==='DETAILS')p.open=true}if(e)e.scrollIntoView()}
document.getElementById('filter').addEventListener('input',function(){let q=this.value.toLowerCase();document.querySelectorAll('.searchable').forEach(e=>{e.hidden=!e.querySelector('summary').textContent.toLowerCase().includes(q)})});addEventListener('hashchange',reveal);addEventListener('DOMContentLoaded',reveal);
</script></body></html>'''
