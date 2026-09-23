"""Portable navigation connecting full relationship identities to saved evidence."""
from pymicroglia._results import workings_link
from pymicroglia._results import report_name, read_document
from .._saved_figures import table_name
from pymicroglia._sources import source_file
import json
from pathlib import Path
from urllib.parse import quote

import pandas as pd

from pymicroglia.pipelines.audit.index import copy_evidence, inventory_artifact
from pymicroglia.pipelines._contracts import ArtifactRef, Settings, StepResult, cell_number, content_id, result_to_dict
from pymicroglia.pipelines.relationships.inputs import KEYS, PAIR_KEYS
from pymicroglia.pipelines.rhythm.index import escaped, link
from pymicroglia.pipelines._screening import _json_value, _write_json, file_hash, read_table

MANIFESTS={"relationship-overview":"relationship_overview_manifest.json",
    "relationship-reports":"relationship_reports_manifest.json",
    "relationship-lag-profiles":"relationship_lag_profiles_manifest.json",
    "relationship-populations":"relationship_populations_manifest.json"}
LABELS={"relationship-design":"Requested measurements and questions","paired-inputs":"Original observations and matching",
    "within-cell-association":"Associations within individual cells","lag-association":"Complete lag searches and delay uncertainty",
    "between-cell-association":"Associations between cell summaries","sample-consistency":"Complete cell and sample summaries",
    "relationship-report-selection":"Supported relationship report selection","relationship-overview":"Relationship overview matrices",
    "relationship-reports":"Selected pair reports and cell traces","relationship-lag-profiles":"Full lag profiles and coverage",
    "relationship-populations":"Complete cell and sample populations"}


def version():
    return content_id({name:file_hash(source_file(name)) for name in ("relationship_index.py","rhythm_index.py","audit_index.py")})


def cell_key(row):
    return {"source_run":row["source_run"],"movie":row["movie"],"identity":cell_number(row["identity"])}


def cell_id(row):return "cell-"+content_id(cell_key(row))


def build(output,inventory,requested,title="Measurement relationships"):
    output=Path(output)
    def source(step,name):
        ref=inventory_artifact(output,inventory,step,name)
        if ref is None:return None
        path=(output/ref["path"]).resolve()
        if not path.is_relative_to(output.resolve()) or not path.is_file() or file_hash(path)!=ref["sha256"]:
            raise ValueError(f"Missing or changed index evidence: {step}/{name}")
        return path
    def table(step,name):
        path=source(step,name);return read_table(path) if path else pd.DataFrame()
    def metadata(step,name):
        path=source(step,name);return read_document(path) if path else {}
    def path_for(step,name):
        if source(step,name) is None:raise ValueError(f"An entry references absent evidence: {step}/{name}")
        return inventory_artifact(output,inventory,step,name)["path"]
    prepared=metadata("paired-inputs","provenance")
    original=prepared.get("resolved_request",{})
    if original and original.get("inputs")!=requested.get("inputs"):
        raise ValueError("Index inputs do not describe the requested original recordings")
    representation=requested["request"]["representation"]
    source_run=requested["inputs"]["source_run"]
    pairs={content_id(pair):{"id":"pair-"+content_id(pair),"pair_id":content_id(pair),**pair,
        "entries":[],"pages":[],"results":{}} for pair in requested["request"]["pairs"]}
    cells={cell_id(row):{**cell_key(row),"id":cell_id(row),"entries":[],"pages":[],"results":{}} for row in requested["inputs"]["cells"]}
    def validate_pair(row):
        pair=pairs.get(row.get("pair_id"))
        if pair is None or (row.get("reference"),row.get("target"))!=(pair["reference"],pair["target"]):
            raise ValueError("Evidence refers to an unknown or reversed measurement pair")
        if row.get("source_run")!=source_run:raise ValueError("Evidence belongs to a different source run")
        if row.get("representation",representation)!=representation:raise ValueError("Evidence uses a different prepared representation")
        return pair
    def validate_cell(row):
        key=cell_id(row)
        if key not in cells:raise ValueError("Evidence refers to an unknown full cell identity")
        return key
    for step in ("within-cell-association","lag-association","between-cell-association","sample-consistency"):
        provenance=metadata(step,"provenance")
        prepared_id=provenance.get("prepared_input_id")
        if prepared_id and prepared_id!=inventory.get("paired-inputs",{}).get("scientific_id"):
            raise ValueError("Scientific evidence belongs to different prepared inputs")
        for name,sid in provenance.get("source_results",{}).items():
            if sid!=inventory.get(name,{}).get("scientific_id"):raise ValueError("Sample evidence refers to a different scientific result")
        frame=table(step,"summaries" if step=="sample-consistency" else "results")
        for row in _json_value(frame.to_dict("records")):
            pair=validate_pair(row);pair["results"].setdefault(step,[]).append(row)
            if step in ("within-cell-association","lag-association"):
                cell=validate_cell(row);cells[cell]["results"].setdefault(step,[]).append(row)
    selected=table("relationship-report-selection","report_members")
    selected_keys={tuple(row[k] for k in PAIR_KEYS) for row in selected.to_dict("records")}
    pages=[];entries=[]
    for step,manifest_name in MANIFESTS.items():
        manifest=metadata(step,manifest_name)
        for name,sid in manifest.get("source_results",{}).items():
            if sid!=inventory.get(name,{}).get("scientific_id"):raise ValueError("Overview belongs to different scientific results")
        local_pages={}
        for page in manifest.get("pages",[]):
            master=page["master"]
            if master in local_pages:raise ValueError("Duplicate manifest figure target")
            path=path_for(step,master)
            page_id="figure-"+content_id({"step":step,"master":master,"scientific_id":inventory[step]["scientific_id"]})
            value={"id":page_id,"step":step,"path":path,"master":master,"entries":[],"cells":[],"pairs":[],"semantics":page,
                "title":LABELS[step]+" | "+str(page.get("view","profile")).replace("_"," ")+" | page "+str(page.get("page_number",len(local_pages)+1)),"data":[]}
            for filename,label in ((table_name(master,'values',lambda filename:source(step,filename)),"Plotted values"),
                (table_name(master,'statistics',lambda filename:source(step,filename)),"Saved statistics"),(table_name(master,'display',lambda filename:source(step,filename)),"Display settings")):
                if source(step,filename):value["data"].append({"path":path_for(step,filename),"label":label})
            local_pages[master]=value;pages.append(value)
        local_entries=table(step,"entries.json")
        for entry in _json_value(local_entries.to_dict("records")):
            pair=validate_pair(entry)
            page=local_pages.get(entry["master"])
            if page is None:raise ValueError("A semantic entry has no declared figure page")
            semantic_page=page["semantics"]
            if "view" in semantic_page and semantic_page["view"]!=entry.get("view"):
                raise ValueError("An entry points to a different scientific question on an existing page")
            if "pair_id" in semantic_page and semantic_page["pair_id"]!=entry["pair_id"]:
                raise ValueError("An entry points to a different pair on an existing page")
            if step=="relationship-overview" and (entry["reference"] not in semantic_page["rows"] or entry["target"] not in semantic_page["columns"]):
                raise ValueError("Overview entry is outside its declared matrix block")
            if step=="relationship-populations" and entry["group_id"] not in semantic_page["groups"]:
                raise ValueError("Population entry is outside its declared sample/recording page")
            expected_source=("between-cell-association" if entry.get("view") in {"between_cells"} else "sample-consistency") if step in {"relationship-overview","relationship-populations"} else "lag-association"
            if step=="relationship-reports":
                if entry.get("selection_id")!=inventory.get("relationship-report-selection",{}).get("scientific_id"):
                    raise ValueError("Report entry refers to a different selected population")
                if tuple(entry[k] for k in PAIR_KEYS) not in selected_keys:raise ValueError("A report entry was not selected by the saved union")
            elif entry.get("source_scientific_id")!=inventory.get(expected_source,{}).get("scientific_id"):
                raise ValueError("Figure entry points at the wrong scientific result")
            member_ids=[]
            if set(KEYS)<=entry.keys() and all(entry[k] is not None for k in KEYS):member_ids.append(validate_cell(entry))
            for row in entry.get("members",[]):
                if "pair_id" in row:validate_pair(row)
                member_ids.append(validate_cell(row))
            member_ids=list(dict.fromkeys(member_ids))
            entry_id="entry-"+entry["entry_id"]
            if any(row["id"]==entry_id for row in entries):raise ValueError("Duplicate semantic entry identity")
            item={**entry,"id":entry_id,"page":page["id"],"path":page["path"],"cells":member_ids,"step":step,
                "representation":representation}
            entries.append(item);page["entries"].append(entry_id);pair["entries"].append(entry_id)
            page["cells"]=list(dict.fromkeys([*page["cells"],*member_ids]));page["pairs"]=list(dict.fromkeys([*page["pairs"],entry["pair_id"]]))
            pair["pages"].append(page["id"])
            for member in member_ids:cells[member]["entries"].append(entry_id);cells[member]["pages"].append(page["id"])
        # Exact per-cell report/profile page membership must match the semantic entries.
        if step in {"relationship-reports","relationship-lag-profiles"}:
            for page in local_pages.values():
                declared={tuple(row[k] for k in PAIR_KEYS) for row in page["semantics"].get("members",[])}
                found={tuple(entry[k] for k in PAIR_KEYS) for entry in entries if entry["page"]==page["id"]}
                if declared!=found:raise ValueError("Figure page members differ from its exact linked entries")
    navigation={"schema_version":1,"analysis_recomputed":False,"representation":representation,"source_run":source_run,
        "cells":list(cells.values()),"pairs":list(pairs.values()),"entries":entries,"pages":pages,"inputs":inventory,
        "requested_settings":requested,"original_prepared_settings":original}
    _write_json(output/"navigation.json",navigation)
    (output/"index.html").write_text(document(navigation,title),encoding="utf-8")
    return navigation


def document(data,title):
    by_page={p["id"]:p for p in data["pages"]};by_entry={e["id"]:e for e in data["entries"]}
    def details(label,value):return '<details><summary>'+escaped(label)+'</summary><pre>'+escaped(json.dumps(value,ensure_ascii=False,indent=2))+'</pre></details>'
    def page_links(ids):return '<ul>'+''.join('<li>'+link('#'+key,by_page[key]["title"])+ '</li>' for key in dict.fromkeys(ids))+'</ul>'
    pair_sections=[];overview=[]
    for pair in data["pairs"]:
        comparison=[]
        for view in ("within_cell","between_cells","delay"):
            row=next((entry for entry in data["entries"] if entry["pair_id"]==pair["pair_id"] and entry["step"]=="relationship-overview" and entry["view"]==view),None)
            if row is None:comparison.append("No saved overview")
            else:
                value="Unavailable" if row.get("value") is None else f"{row['value']:.3g}"+(" hours" if view=="delay" else "")
                comparison.append(value+"; "+row["status"].replace("-"," "))
        overview.append('<tr><td>'+link('#'+pair['id'],pair['reference']+' to '+pair['target'])+'</td>'+''.join('<td>'+link('#'+pair['id'],value)+'</td>' for value in comparison)+'</tr>')
        rows=[]
        for key in pair["entries"]:
            entry=by_entry[key];label=entry.get("view","").replace("_"," ")
            if entry.get("movie") is not None:label+=f" | {entry['movie']} / cell {entry['identity']}"
            if entry.get("group_id") is not None:label+=" | "+entry["group_id"].replace("sample:","Biological sample ").replace("recording:","Recording ")
            rows.append('<li id="'+entry['id']+'">'+link('#'+entry['page'],label)+details("Exact linked members and saved source",entry)+'</li>')
        pair_sections.append('<section id="'+pair['id']+'"><h3>'+escaped(pair['reference']+' to '+pair['target'])+'</h3><ul>'+''.join(rows)+'</ul>'+details("Complete pair outcomes and sample evidence",pair['results'])+'</section>')
    cell_rows=[];cell_sections=[]
    for cell in data["cells"]:
        label=f"{cell['movie']} / cell {cell['identity']}"
        values=[]
        for step in ('within-cell-association','lag-association'):
            values.append('; '.join(row['reference']+' to '+row['target']+': '+row['status'] for row in cell['results'].get(step,[])) or 'No saved result')
        cell_rows.append('<tr><td>'+link('#'+cell['id'],label)+'</td>'+''.join('<td>'+escaped(value)+'</td>' for value in values)+'</tr>')
        selected=[by_entry[key] for key in cell['entries'] if by_entry[key]['step']=='relationship-reports']
        report_note='' if selected else '<p>No selected report was produced; the saved outcomes and all available population/profile pages remain linked.</p>'
        cell_sections.append('<section id="'+cell['id']+'"><h3>'+escaped(label)+'</h3>'+report_note+page_links(cell['pages'])+details("Full source identity and question-specific outcomes",cell)+'</section>')
    pages=[]
    for page in data['pages']:
        pages.append('<details id="'+page['id']+'"><summary>'+escaped(page['title'])+'</summary><p>'+link(page['path'],'Open saved figure')+' | '+' | '.join(link(a['path'],a['label']) for a in page['data'])+'</p>'
            +'<img loading="lazy" src="'+quote(page['path'],safe='/')+'" alt="'+escaped(page['title'])+'">'+details("Exact page entries and membership",page)+'</details>')
    branches=[]
    for step,row in data['inputs'].items():
        branches.append('<details><summary>'+escaped(LABELS.get(step,step))+': '+escaped(row['status'])+'</summary><p>'+escaped(row['reason'])+'</p>'+link(row['record'],'Execution outcome and exact selections')+'<ul>'+''.join('<li>'+link(a['path'],name)+'</li>' for name,a in row['artifacts'].items())+'</ul></details>')
    failures=[row for row in data['inputs'].values() if row['status'] in {'unavailable','failed'}]
    banner='<p class="status">Some requested branches are unfinished. Their exact states and reasons are retained below.</p>' if failures else ''
    return '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'''+escaped(title)+'''</title>
<style>body{font:16px Arial,sans-serif;max-width:1250px;margin:30px auto;padding:0 20px;color:black}a{color:steelblue}nav{display:flex;gap:22px;flex-wrap:wrap}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;border:1px solid silver;text-align:left}details{margin:14px 0;padding:9px;border:1px solid silver}summary{cursor:pointer;font-weight:bold}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:whitesmoke;padding:10px}img{width:100%;height:auto}section{border-top:1px solid silver;padding:12px 0}.status{background:whitesmoke;padding:14px}</style></head><body><h1>'''+escaped(title)+'''</h1>'''+banner+'''<nav><a href="#pairs">Measurement pairs</a><a href="#cells">Cells</a><a href="#figures">Figures</a><a href="#settings">Settings</a><a href="#evidence">Execution and evidence</a></nav>
<p>Within-cell association, differences between cell summaries and temporal delay are separate questions. A supported association can have unresolved timing. Correlation does not establish rhythm, phase synchrony or causation.</p>
<h2>Relationship overview</h2><table><tr><th>Reference to target</th><th>Within cells</th><th>Between summaries</th><th>Supported delay</th></tr>'''+''.join(overview)+'''</table>
<h2 id="pairs">Linked relationship evidence</h2>'''+''.join(pair_sections)+'''<h2 id="cells">Every requested cell</h2><table><tr><th>Movie / cell</th><th>Same-time outcomes</th><th>Lag-search outcomes</th></tr>'''+''.join(cell_rows)+'''</table>'''+''.join(cell_sections)+'''<h2 id="figures">Saved figures</h2>'''+''.join(pages)+'''<h2 id="settings">Scientific settings</h2>'''+details('Current resolved request',data['requested_settings'])+details('Original prepared measurements, matching and sample assignments',data['original_prepared_settings'])+'''<h2 id="evidence">Execution and complete evidence</h2>'''+''.join(branches)+'<p>'+link(workings_link('navigation.json'),'Complete portable navigation inventory')+'''</p><script>function reveal(){var e=document.getElementById(decodeURIComponent(location.hash.slice(1)));for(var p=e;p;p=p.parentElement){if(p.tagName==='DETAILS')p.open=true}if(e)e.scrollIntoView()}addEventListener('hashchange',reveal);addEventListener('DOMContentLoaded',reveal)</script></body></html>'''


def produce(context):
    appearance=context.presentation.as_dict().get("report",{})
    if not isinstance(appearance,dict) or set(appearance)-{"title"} or not isinstance(appearance.get("title",""),str):
        raise ValueError("Report presentation accepts a title string")
    context.output.mkdir(parents=True,exist_ok=True)
    inventory=copy_evidence(context.dependencies,context.output)
    records=context.output/"execution-records";records.mkdir()
    for step,saved in context.dependencies.items():
        path=records/(step+".json");path = _write_json(path,result_to_dict(saved.outcome));inventory[step]["record"]=path.relative_to(context.output).as_posix()
    navigation=build(context.output,inventory,context.request.as_dict(),**appearance)
    refs=tuple(ArtifactRef(report_name(path,context.output),path.relative_to(context.output).as_posix(),file_hash(path),context.scientific_id)
        for path in sorted(context.output.rglob("*")) if path.is_file() and path.name!="artefacts.json")
    return StepResult(context.step.name,context.scientific_id,"completed","Saved portable navigation for exact cells, ordered pairs, groups and evidence",refs,
        provenance=Settings({"analysis_recomputed":False,"source_run":navigation["source_run"],"representation":navigation["representation"]}))
