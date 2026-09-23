"""Saved-only distributions and explicit sample-level sign consistency."""
from __future__ import annotations
from pymicroglia._sources import source_file

from importlib.metadata import version as library_version
from pathlib import Path

import numpy as np
import pandas as pd

from pymicroglia.pipelines._contracts import ArtifactRef, Settings, StepResult, content_id
from pymicroglia.pipelines.relationships.between import correct_aggregate_families
from pymicroglia.pipelines.relationships.inputs import PAIR_KEYS
from pymicroglia.pipelines._screening import _write_json, file_hash, read_table, write_table


MEMBER_COLUMNS = PAIR_KEYS + ["sample", "sample_confirmed", "question", "source_scientific_id", "effect", "eligible",
    "eligibility_reason", "effect_population", "original_status", "p_value", "q_value", "tested", "significant",
    "delay_supported", "delay_hours", "delay_interval_hours", "resolution_status", "association_sign"]
SUMMARY_COLUMNS = ["source_run", "pair_id", "reference", "target", "question", "level", "group_id", "representation",
    "aggregation", "effect_population", "effect", "effect_values", "effect_min", "effect_max", "positive_cells", "negative_cells",
    "zero_cells", "mixed_directions", "requested_cells", "eligible_cells", "tested_cells", "supported_cells", "untestable_cells",
    "not_detected_cells", "disabled_cells", "requested_recordings", "eligible_recordings", "confirmed_samples", "eligible_samples",
    "unconfirmed_eligible_cells", "members", "source_scientific_id", "delay_supported_cells", "delay_unresolved_cells",
    "delay_values_hours", "delay_summary_hours", "delay_summary_status", "delay_summary_reason", "delay_compatible_interval_hours",
    "delay_association_sign", "status", "reason", "positive_samples", "negative_samples", "tied_samples", "tested_samples",
    "positive_fraction", "positive_fraction_interval", "p_value", "q_value", "significant", "family_id", "family_requested",
    "family_tested", "correction", "alpha"]


def implementation_version():
    return content_id({"code": {path.name: file_hash(path) for path in (Path(__file__),
        source_file("relationship_between.py"), source_file('relationship_consistency_statistics.py'),
        source_file('circadian.py'))},
        "libraries": {name: library_version(name) for name in ("numpy", "pandas", "scipy")}})


def _members(context):
    rows=[]
    for question, step in (("within_cell","within-cell-association"),("lag","lag-association")):
        saved=context.saved(step)
        outcomes=read_table(saved.artifact("results"))
        profiles=read_table(saved.artifact("profiles")) if question=="lag" else None
        for item in outcomes.to_dict("records"):
            value=None; reason="The requested question is disabled" if item["status"]=="disabled" else "No eligible saved descriptive effect"
            if question=="within_cell":
                value=item["full_overlap_effect"] if item["support_status"]=="eligible" else None
                population="Full original same-time overlap coefficient; all eligible cells, independent of evidence availability"
            else:
                selected=profiles
                for name in PAIR_KEYS: selected=selected.loc[selected[name].eq(item[name])]
                population="Signed empirical absolute maximum over the complete declared descriptive lag profile; all eligible cells"
                if len(selected) and selected.status.eq("descriptive").all() and selected.effect.notna().all():
                    values=selected.effect.to_numpy(float); maximum=max(abs(values))
                    peaks=values[abs(values)>=maximum-64*np.finfo(float).eps]
                    if len(set(np.sign(peaks)))==1: value=float(peaks[0])
                    else: reason="Equal empirical lag maxima have incompatible association signs"
            finite=value is not None and np.isfinite(value)
            rows.append({**{name:item[name] for name in PAIR_KEYS+["sample","sample_confirmed"]},"question":question,
                "source_scientific_id":saved.outcome.scientific_id,"effect":float(value) if finite else None,"eligible":finite,
                "eligibility_reason":"Eligible saved descriptive effect" if finite else reason,"effect_population":population,
                "original_status":item["status"],"p_value":item.get("p_value"),"q_value":item.get("q_value"),
                "tested":pd.notna(item.get("p_value")),"significant":bool(item.get("significant",False)),
                "delay_supported":bool(item.get("delay_supported",False)),"delay_hours":item.get("delay_hours"),
                "delay_interval_hours":item.get("delay_interval_hours"),"resolution_status":item.get("resolution_status"),
                "association_sign":item.get("association_sign")})
    return pd.DataFrame(rows,columns=MEMBER_COLUMNS)


def _delay_summary(values, intervals, signs, aggregation):
    base={"delay_values_hours":values,"delay_summary_hours":None,"delay_compatible_interval_hours":None,
          "delay_association_sign":next(iter(set(signs))) if len(set(signs))==1 else None}
    if not values:
        return {**base,"delay_summary_status":"unavailable","delay_summary_reason":"No resolved supported delays"}
    if len(set(signs))!=1 or any(sign not in {"positive","negative"} for sign in signs):
        return {**base,"delay_summary_status":"incompatible","delay_summary_reason":"Supported delays have different association signs"}
    if any(not isinstance(bounds,(list,tuple)) or len(bounds)!=2 or not np.all(np.isfinite(bounds)) for bounds in intervals):
        return {**base,"delay_summary_status":"unavailable","delay_summary_reason":"A resolved delay lacks its saved uncertainty interval"}
    low,high=max(bounds[0] for bounds in intervals),min(bounds[1] for bounds in intervals)
    if low>high:
        return {**base,"delay_summary_status":"incompatible","delay_summary_reason":"Resolved delays do not share a compatible uncertainty region"}
    value=float(getattr(np,aggregation)(values))
    return {**base,"delay_summary_hours":value,"delay_compatible_interval_hours":[low,high],
        "delay_summary_status":"descriptive-compatible","delay_summary_reason":"Descriptive aggregate of supported delays sharing sign and an uncertainty region; not a new confidence interval"}


def _group(key,members,level,group_id,request,units=None):
    eligible=members.loc[members.eligible]
    values=(units.effect.dropna().to_numpy(float) if units is not None else eligible.effect.to_numpy(float))
    cell_values=eligible.effect.to_numpy(float)
    aggregation=request.sample_summary["aggregation"]
    supported=members.loc[members.delay_supported]
    if units is None:
        delay=_delay_summary(supported.delay_hours.tolist(),supported.delay_interval_hours.tolist(),supported.association_sign.tolist(),aggregation)
    else:
        delay=_delay_summary(supported.delay_hours.tolist(),supported.delay_interval_hours.tolist(),supported.association_sign.tolist(),aggregation)
        if delay["delay_summary_status"]=="descriptive-compatible":
            supported_units=units.loc[units.delay_summary_status.eq("descriptive-compatible")]
            delay=_delay_summary(supported_units.delay_summary_hours.tolist(),supported_units.delay_compatible_interval_hours.tolist(),
                supported_units.delay_association_sign.tolist(),aggregation)
    return {**key,"level":level,"group_id":group_id,"representation":request.representation,"aggregation":aggregation,
        "effect_population":("Equal-weight confirmed sample aggregates of " if units is not None else "Eligible cell distribution of ")+
            (members.effect_population.iloc[0] if len(members) else "the declared saved effect"),
        "effect":float(getattr(np,aggregation)(values)) if len(values) else None,"effect_values":values.tolist(),
        "effect_min":float(min(values)) if len(values) else None,"effect_max":float(max(values)) if len(values) else None,
        "positive_cells":int(np.sum(cell_values>0)),"negative_cells":int(np.sum(cell_values<0)),"zero_cells":int(np.sum(cell_values==0)),
        "mixed_directions":bool(np.any(cell_values>0) and np.any(cell_values<0)),"requested_cells":len(members),"eligible_cells":len(eligible),
        "tested_cells":int(members.tested.sum()),"supported_cells":int(members.significant.sum()),
        "untestable_cells":int(members.original_status.eq("untestable").sum()),"not_detected_cells":int(members.original_status.eq("no-detected-association").sum()),
        "disabled_cells":int(members.original_status.eq("disabled").sum()),"requested_recordings":members.movie.nunique(),
        "eligible_recordings":eligible.movie.nunique(),"confirmed_samples":members.loc[members.sample_confirmed,"sample"].nunique(),
        "eligible_samples":eligible.loc[eligible.sample_confirmed,"sample"].nunique(),"unconfirmed_eligible_cells":int((~eligible.sample_confirmed).sum()),
        "members":members[PAIR_KEYS+["question","source_scientific_id"]].to_dict("records"),
        "source_scientific_id":members.source_scientific_id.iloc[0] if len(members) else None,
        "delay_supported_cells":len(supported),"delay_unresolved_cells":int((members.significant & ~members.delay_supported).sum()) if key["question"]=="lag" else 0,
        **delay,"status":"descriptive" if len(values) else "unavailable","reason":"All eligible saved effects retained" if len(values) else "No eligible saved effects",
        "p_value":None,"q_value":None,"significant":False}


def produce(context):
    import pymicroglia.measure.relationship_consistency_statistics as statistics
    request=context.request.request
    settings=request.sample_summary
    statistics.validate(settings)
    if settings["enabled"] and settings["evidence"]["method"]!="none" and request.inference["correction_scope"]=="movie_pair":
        raise ValueError("sample_summary inference requires all or pair correction_scope")
    members=_members(context)
    rows, details, across=[],[],[]
    if settings["enabled"]:
        for pair in request.pairs:
            for question in ("within_cell","lag"):
                subset=members.loc[members.pair_id.eq(pair.record_id) & members.question.eq(question)]
                key={"source_run":context.request.inputs.source_run,"pair_id":pair.record_id,**pair.as_dict(),"question":question}
                local=[]
                for level,grouper,population in (("recording","movie",subset),
                        ("biological_sample","sample",subset.loc[subset.sample_confirmed])):
                    for group_id,group in population.groupby(grouper,sort=True):
                        local.append(_group(key,group,level,group_id,request))
                rows.extend(local)
                rows.append(_group(key,subset,"all_cells","all-cells",request))
                units=pd.DataFrame([r for r in local if r["level"]=="biological_sample"],columns=SUMMARY_COLUMNS)
                summary=_group(key,subset,"across_samples","all-confirmed-samples",request,units)
                native={}
                if not getattr(request,question)["enabled"]:
                    summary.update(status="disabled",reason="This relationship question was not requested")
                else:
                    if settings["evidence"]["method"]!="none" and summary["unconfirmed_eligible_cells"]:
                        native=statistics.sign_evidence(units.effect.dropna().to_numpy(float),
                            {**settings.as_dict(),"evidence":{"method":"none"}},None)
                        native.update(status="untestable",reason="Eligible cells lack confirmed sample mapping; biological evidence withheld",
                            p_value=None,positive_fraction_interval=None)
                    else:
                        native=statistics.sign_evidence(units.effect.dropna().to_numpy(float),settings,request.inference.get("alpha"))
                    # Preserve measurement identity when native provenance includes a paper reference.
                    summary.update({name:value for name,value in native.items() if name in SUMMARY_COLUMNS and name not in PAIR_KEYS})
                    across.append(summary)
                rows.append(summary);details.append({**key,"result":native})
        # Correct the sign-probability hypotheses without replacing the saved
        # coefficient summaries with a different effect's sign or interval.
        directional=[{**row,"effect":row.get("positive_fraction",.5)-.5 if row.get("positive_fraction") is not None else 0.} for row in across]
        families=correct_aggregate_families(directional,request,"sample_consistency")
        fields=("q_value","significant","family_id","family_requested","family_tested","correction","alpha")
        for row,corrected in zip(across,directional):
            row.update({name:corrected[name] for name in fields})
            if row["p_value"] is not None:
                row.update(status=("positive-consistency" if row["positive_fraction"]>.5 else "negative-consistency") if row["significant"] else "no-detected-consistency",
                    reason="Exact evidence of a sign imbalance across independent observed sample summaries" if row["significant"] else "The corrected test did not detect a sample-level sign imbalance")
    else:
        families=correct_aggregate_families([],request,"sample_consistency")
    context.output.mkdir(parents=True)
    refs=[]
    summaries=pd.DataFrame(rows,columns=SUMMARY_COLUMNS)
    for name,frame in {"members":members,"summaries":summaries,"families":families,
            "units":summaries.loc[summaries.level.eq("biological_sample")].copy()}.items():
        path=context.output/(name+".json");path = write_table(path,frame)
        refs.append(ArtifactRef(name,path.name,file_hash(path),context.scientific_id,columns=tuple(frame.columns)))
    for name,value in {"engine_details":details,"provenance":{"schema_version":1,"scientific_id":context.scientific_id,
        "settings":settings.as_dict(),"inference":request.inference.as_dict(),"implementation":implementation_version(),
        "source_results":{name:context.saved(name).outcome.scientific_id for name in ("within-cell-association","lag-association")},
        "cell_population":"All eligible saved descriptive coefficients; never restricted by significance",
        "sample_weighting":"Aggregate cells once per confirmed sample across its recordings, then weight samples equally",
        "delay_population":"Only supported resolved delays with matching question, representation, association sign and intersecting uncertainty regions",
        "delay_interval_intersection":"A compatibility check, not a newly estimated confidence interval",
        "test_reference":statistics.REFERENCE,"fresh_association_tests":False,"fresh_lag_searches":False,"preprocessing_repeated":False,
        "branch_status":"enabled" if settings["enabled"] else "disabled"}}.items():
        path=context.output/(name+".json");_write_json(path,value)
        refs.append(ArtifactRef(name,path.name,file_hash(path),context.scientific_id))
    return StepResult(context.step.name,context.scientific_id,"completed",
        "Saved complete cell distributions and explicit confirmed-sample consistency summaries",tuple(refs))
