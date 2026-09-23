"""Select relationship reports from immutable same-time OR lag evidence."""
from pymicroglia._sources import source_file
from pathlib import Path

import pandas as pd

from pymicroglia.pipelines._contracts import ArtifactRef, SelectionRecord, Settings, StepResult, content_id
from pymicroglia.pipelines.relationships.inputs import PAIR_KEYS
from pymicroglia.pipelines._screening import _write_json, file_hash, read_table, write_table


SELECTION="supported-relationships"
MEMBER_COLUMNS=PAIR_KEYS+["supporting_questions","within_status","lag_status","within_scientific_id","lag_scientific_id"]


def selection_version():
    return content_id({"code":file_hash(Path(__file__)),"contract":file_hash(source_file("contracts.py"))})


def select_reports(context):
    sources={"within_cell":context.saved("within-cell-association"),"lag":context.saved("lag-association")}
    names={"within_cell":"association-supported","lag":"lag-association-supported"}
    chosen={};definitions={};lookups={}
    for question,saved in sources.items():
        selection=next(item for item in saved.outcome.selections if item.name==names[question])
        definitions[question]={"source_scientific_id":saved.outcome.scientific_id,"selection_name":selection.name,
            "selection_id":selection.record_id,"definition":selection.rule.as_dict()}
        outcomes=read_table(saved.artifact("results"))
        lookups[question]={tuple(row[name] for name in PAIR_KEYS):row for row in outcomes.to_dict("records")}
        if len(lookups[question])!=len(outcomes):raise ValueError("Duplicate full cell/pair keys in saved relationship results")
        for member in selection.members:
            key=tuple(member[name] for name in PAIR_KEYS)
            if key not in lookups[question]:raise ValueError("A selected relationship has no matching saved result")
            chosen.setdefault(key,[]).append(question)
    if set(lookups["within_cell"])!=set(lookups["lag"]):
        raise ValueError("Same-time and lag results describe different requested cell/pair populations")
    rows=[]
    for key,questions in sorted(chosen.items()):
        rows.append({**dict(zip(PAIR_KEYS,key)),"supporting_questions":questions,
            "within_status":lookups["within_cell"][key]["status"],"lag_status":lookups["lag"][key]["status"],
            "within_scientific_id":sources["within_cell"].outcome.scientific_id,
            "lag_scientific_id":sources["lag"].outcome.scientific_id})
    selection=SelectionRecord(SELECTION,context.scientific_id,
        Settings({"rule":"Union of saved supported same-time OR complete-search lag associations",
            "sources":definitions,"union_is_new_test":False,"precise_delay_required":False}),
        tuple(Settings({name:row[name] for name in PAIR_KEYS}) for row in rows))
    context.output.mkdir(parents=True)
    frame=pd.DataFrame(rows,columns=MEMBER_COLUMNS);path=context.output/"report_members.json";path = write_table(path,frame)
    provenance=context.output/"provenance.json";_write_json(provenance,{"schema_version":1,"scientific_id":context.scientific_id,
        "sources":definitions,"requested_cell_pairs":len(lookups["within_cell"]),"selected_cell_pairs":len(rows),
        "rule":selection.rule.as_dict(),"scientific_tests_performed":False})
    return StepResult(context.step.name,context.scientific_id,"completed","Saved the exact supported relationship report union",
        (ArtifactRef("report_members",path.name,file_hash(path),context.scientific_id,columns=tuple(frame.columns)),
         ArtifactRef("provenance",provenance.name,file_hash(provenance),context.scientific_id)),(selection,))
