"""Explicit event, window and experimental-unit design, before analysis."""
from dataclasses import dataclass,replace
from pathlib import Path
import re

from pymicroglia.pipelines._contracts import ArtifactRef, InputIdentity, Measurement, PipelineRecipe, Record, Settings, StepResult, StepSpec, cell_number, content_id, text_key
from pymicroglia.pipelines.relationships.options import _number
from pymicroglia.pipelines.rhythm.discovery import MeasurementChoice, _catalogue, _choices, _known_keys, _object, _population, _resolve_measurement, _shape_problem

OPERATIONS={'mean','median','min','max'}


def measurement(choice,operation,tables,catalogue,grains):
    """A windowed scalar needs its own saved bounds and operation in the table."""
    from pandas.api.types import is_numeric_dtype,is_bool_dtype,is_complex_dtype
    candidates=[];problems=[]
    for name in ([choice.table] if choice.table else tables):
        if name not in tables or choice.column not in tables[name]:continue
        frame=tables[name];grain=tuple(grains.get(name,()))
        if set(grain)-{'stem'}=={'identity','window'}:
            required={'stem','identity','window','window_coordinate','window_start','window_end','summary_operation'}
            if required-set(frame):problems.append(name+': saved window summaries require original bounds, coordinate and summary_operation columns');continue
            if not frame.columns.is_unique or frame[['stem','identity','window']].isna().any().any() or frame.duplicated(['stem','identity','window']).any():
                problems.append(name+': duplicate or missing original window identity');continue
            dtype=frame[choice.column].dtype
            if (not is_numeric_dtype(dtype) and not frame[choice.column].isna().all()) or is_bool_dtype(dtype) or is_complex_dtype(dtype):
                problems.append(name+': window outcome must be real numeric');continue
            if not frame.summary_operation.eq(operation).all():problems.append(name+': requested operation differs from the saved window summary');continue
            if frame.window_coordinate.eq('relative_hours').any() and 'window_anchor_hours' not in frame:
                problems.append(name+': relative window summaries require their original window_anchor_hours');continue
            if choice.column in required:problems.append(name+': window coordinates are not measured outcomes');continue
            for row in frame.to_dict('records'):
                windows([{'name':row['window'],'coordinate':row['window_coordinate'],'start':row['window_start'],'end':row['window_end']}],'saved window')
            column=catalogue.get(choice.column)
            candidates.append(Measurement(choice.column,name,grain,column.label if column else choice.column,column.unit if column else '',column is not None,operation))
        else:
            try:candidates.append(replace(_resolve_measurement(replace(choice,table=name,summary=None),tables,catalogue,grains,testing=True),summary=operation))
            except ValueError as error:problems.append(str(error))
    if len(candidates)>1:raise ValueError(choice.column+': ambiguous measured source; choose its table explicitly')
    if not candidates:raise ValueError(choice.column+': unsuitable measurement; '+'; '.join(problems))
    return candidates[0]


def windows(value,where):
    if not isinstance(value,list):raise ValueError(where+' must be a list of explicit windows')
    result=[];names=set()
    for item in value:
        item=_object(item,where)
        _known_keys(item,{'name','coordinate','start','end','baseline','description','from_hours','to_hours','from_frame','to_frame'},where)
        name=text_key(item.get('name'),where+'.name')
        if name in names:raise ValueError('Repeated window name: '+name)
        names.add(name)
        native_hours=any(key in item for key in ['from_hours','to_hours']);native_frames=any(key in item for key in ['from_frame','to_frame'])
        if native_hours or native_frames:
            if native_hours and native_frames or any(key in item for key in ['start','end','coordinate']):raise ValueError('Conflicting window coordinate definitions')
            coordinate='recording_hours' if native_hours else 'frames';suffix='hours' if native_hours else 'frame'
            start,end=item.get('from_'+suffix),item.get('to_'+suffix)
        else:coordinate=item.get('coordinate');start,end=item.get('start'),item.get('end')
        if coordinate not in {'relative_hours','recording_hours','frames'}:raise ValueError('Window coordinate must explicitly be relative_hours, recording_hours or frames')
        start=_number(start,where+'.start');end=_number(end,where+'.end')
        if end<=start:raise ValueError('A half-open window must end after it starts')
        if coordinate=='frames' and (start<0 or not start.is_integer() or not end.is_integer()):raise ValueError('Frame bounds must be nonnegative integers')
        baseline=item.get('baseline')
        if baseline is not None:text_key(baseline,where+'.baseline')
        result.append({'name':name,'coordinate':coordinate,'start':start,'end':end,'baseline':baseline,'description':str(item.get('description',''))})
    for item in result:
        if item['baseline'] is not None and (item['baseline']==item['name'] or item['baseline'] not in names):raise ValueError('Comparison requires a distinct declared baseline window')
    return tuple(Settings(item) for item in result)


def optional(value,where,keys):
    block=_object(value,where);enabled=block.get('enabled',False)
    if not isinstance(enabled,bool):raise ValueError(where+'.enabled must be boolean')
    _known_keys(block,{'enabled',*keys} if enabled else {'enabled'},where)
    return Settings({**block,'enabled':enabled})


@dataclass(frozen=True)
class InterventionRequest(Record):
    name:str
    declaration:Settings
    measurements:tuple[MeasurementChoice,...]
    summaries:Settings
    anchors:Settings
    windows:tuple[Settings,...]
    recording_windows:Settings
    support:Settings
    evidence:Settings
    relative_effects:Settings
    meaningful_effects:Settings
    inference:Settings
    controls:Settings
    timing:Settings
    rhythms:Settings
    coordinated:Settings
    biological_samples:Settings
    conditions:Settings
    matching:tuple[Settings,...]
    table_grains:Settings
    cells:tuple|None
    pipeline:str='intervention-response'

    @classmethod
    def from_dict(cls,value,groups,*,where='intervention-response'):
        block=_object(value,where)
        if block.get('pipeline')!=cls.pipeline:raise ValueError('Expected intervention-response pipeline')
        _known_keys(block,{'pipeline','name','measurements','summary','anchors','windows','recording_windows','support','evidence','relative_effects',
            'meaningful_effects','inference','controls','timing','rhythms','coordinated','biological_samples','conditions','matching','table_grains','cells'},where)
        name=text_key(block.get('name',cls.pipeline),where+'.name')
        if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',name) is None:raise ValueError('Pipeline name must be a plain directory identifier')
        measurements=_choices(block.get('measurements'),groups,where+'.measurements')
        if not measurements:raise ValueError('At least one measured outcome is required')
        summaries={m.column:m.summary or block.get('summary') for m in measurements}
        if any(value not in OPERATIONS for value in summaries.values()):raise ValueError('Each measurement needs an explicit mean, median, min or max summary')
        anchors=_object(block.get('anchors'),where+'.anchors');normal={}
        for movie,item in anchors.items():
            text_key(movie,'anchor movie');item=_object(item,'anchor');_known_keys(item,{'hours','kind','label'},'anchor')
            if item.get('kind') not in {'intervention','control'}:raise ValueError('Each anchor must explicitly identify intervention or control')
            normal[movie]={'hours':_number(item.get('hours'),'anchor.hours'),'kind':item['kind'],'label':text_key(item.get('label'),'anchor.label')}
        shared=windows(block.get('windows',[]),'windows');per_movie=_object(block.get('recording_windows',{}),'recording_windows')
        per_movie={movie:[row.as_dict() for row in windows(items,'recording_windows.'+movie)] for movie,items in per_movie.items()}
        support=_object(block.get('support'),'support');_known_keys(support,{'min_observations','min_valid_fraction','min_coverage','max_gap_hours'},'support')
        support={'min_observations':4,'min_valid_fraction':.8,'min_coverage':.8,**support}
        if isinstance(support['min_observations'],bool) or not isinstance(support['min_observations'],int) or support['min_observations']<1:raise ValueError('min_observations must be a positive integer')
        for key in ['min_valid_fraction','min_coverage']:
            support[key]=_number(support[key],'support.'+key)
            if not 0<=support[key]<=1:raise ValueError(key+' must lie between zero and one')
        support['max_gap_hours']=_number(support.get('max_gap_hours'),'support.max_gap_hours',positive=True)
        evidence=_object(block.get('evidence',{'method':'none'}),'evidence');text_key(evidence.get('method'),'evidence.method')
        if evidence['method']=='none' and set(evidence)!={'method'}:raise ValueError('Evidence method none has no inferential settings')
        from pymicroglia.pipelines.intervention.settings import settings as evidence_settings
        evidence=evidence_settings(evidence)
        inference=_object(block.get('inference',{}),'inference');_known_keys(inference,{'alpha','multiple_testing','correction_scope'},'inference')
        if inference:
            alpha=_number(inference.get('alpha'),'inference.alpha',positive=True)
            if alpha>=1:raise ValueError('alpha must be below one')
            if inference.get('multiple_testing') not in {'none','bonferroni','sidak','bh'}:raise ValueError('Unknown multiple-testing correction')
            if inference.get('correction_scope') not in {'all','measurement'}:raise ValueError('Correction scope must be all or measurement; evidence levels remain separate')
        if evidence['method']!='none' and not inference:raise ValueError('Within-cell evidence requires declared inference settings')
        controls=optional(block.get('controls',{}),'controls',{'aggregation','comparisons','evidence'})
        if controls['enabled']:
            from pymicroglia.pipelines.intervention.control_statistics import control_policy
            control_policy(controls.as_dict(),summaries,inference)
        timing=optional(block.get('timing',{}),'timing',{'method','window_hours','step_hours','thresholds','persistence_hours','recovery','evidence','baseline_reference','immediate_hours'})
        if timing['enabled']:
            from pymicroglia.pipelines.intervention.timing_rules import policy as timing_policy
            timing_policy(timing.as_dict(),summaries)
        rhythms=optional(block.get('rhythms',{}),'rhythms',{'measurements','analysis_options','settings_profile','comparisons','direct_change','correction_scope'})
        if rhythms['enabled']:
            from pymicroglia.pipelines.intervention.rhythms import policy as rhythm_policy
            rhythm_policy(rhythms.as_dict(),summaries)
        coordinated=optional(block.get('coordinated',{}),'coordinated',{'pairs','aggregation','quantity','evidence'})
        if coordinated['enabled']:
            from pymicroglia.pipelines.intervention.patterns import policy as pattern_policy
            pattern_policy(coordinated.as_dict(),summaries,inference)
        mappings=[]
        for key in ['biological_samples','conditions']:
            mapping=_object(block.get(key,{}),key)
            for movie,label in mapping.items():text_key(movie,key);text_key(label,key+'.'+movie)
            mappings.append(Settings(mapping))
        matching=[];used=set();ids=set()
        for pair in block.get('matching',[]):
            pair=_object(pair,'matching');_known_keys(pair,{'match_id','reference_sample','target_sample'},'matching')
            for key in ['match_id','reference_sample','target_sample']:text_key(pair.get(key),'matching.'+key)
            members={pair['reference_sample'],pair['target_sample']}
            if len(members)!=2 or members&used or pair['match_id'] in ids:raise ValueError('Matching must name unique distinct biological samples and match IDs')
            used.update(members);ids.add(pair['match_id']);matching.append(Settings(pair))
        cells=None
        if block.get('cells') is not None:
            if not isinstance(block['cells'],list):raise ValueError('cells must be movie/identity objects')
            cells=[]
            for item in block['cells']:
                item=_object(item,'cells');_known_keys(item,{'movie','identity'},'cells')
                key=(text_key(item.get('movie'),'cells.movie'),cell_number(item.get('identity')))
                if key not in cells:cells.append(key)
            cells=tuple(cells)
        relative=_object(block.get('relative_effects',{}),'relative_effects');meaningful=_object(block.get('meaningful_effects',{}),'meaningful_effects')
        if (set(relative)|set(meaningful))-set(summaries):raise ValueError('Effect settings name an unrequested measurement')
        from pymicroglia.pipelines.intervention.settings import relative_settings
        relative={key:relative_settings(value) for key,value in relative.items()}
        for key,threshold in meaningful.items():
            if _number(threshold,'meaningful_effects.'+key)<0:raise ValueError('Meaningful-effect magnitude cannot be negative')
        return cls(name,Settings(block),measurements,Settings(summaries),Settings(normal),shared,Settings(per_movie),Settings(support),Settings(evidence),
            Settings(relative),Settings(meaningful),Settings(inference),controls,timing,rhythms,coordinated,*mappings,tuple(matching),
            Settings(_object(block.get('table_grains',{}),'table_grains')),cells)


@dataclass(frozen=True)
class ResolvedInterventionRequest(Record):
    request:InterventionRequest
    inputs:InputIdentity
    measurements:tuple[Measurement,...]
    recordings:Settings
    rhythm_analysis:Settings

    @property
    def scientific_id(self):
        request=self.request.as_dict();request.pop('name');request.pop('declaration')
        return content_id({**self.as_dict(),'request':request})


def resolve_request(request,*,source_run,tables,input_hashes,rhythm_params=None,table_grains=None,recording_windows=None,default_windows=None):
    import pandas as pd
    text_key(source_run,'source_run')
    if any(not isinstance(name,str) or not isinstance(frame,pd.DataFrame) for name,frame in tables.items()):raise ValueError('tables must map names to measured DataFrames')
    declared=request.table_grains.as_dict()
    for name,grain in (table_grains or {}).items():
        if name in declared and tuple(declared[name])!=tuple(grain):raise ValueError('Conflicting table grain: '+name)
        declared[name]=grain
    catalogue,grains=_catalogue(declared)
    measurements=tuple(measurement(choice,request.summaries[choice.column],tables,catalogue,grains) for choice in request.measurements)
    names={m.table for m in measurements}
    if 'cell_summary' in tables:
        problem=_shape_problem(tables['cell_summary'],grains['cell_summary'],summary=None,testing=False)
        if problem:raise ValueError('cell_summary population: '+problem)
        names.add('cell_summary')
    cells,samples=_population(source_run,tables,sorted(names),request);movies={cell.movie for cell in cells}
    known={movie for name in names for movie in tables[name].stem.unique()}
    if movies-set(request.anchors):raise ValueError('Every selected recording needs its actual intervention or control anchor')
    for mapping in [request.anchors,request.recording_windows,request.conditions,recording_windows or {}]:
        if set(mapping)-known:raise ValueError('Design names an unavailable recording')
    sample_lookup={item.movie:item for item in samples};sample_roles={};recordings={}
    for movie in sorted(movies):
        anchor=request.anchors[movie];explicit=request.recording_windows.get(movie)
        inherited=(recording_windows or {}).get(movie)
        inherited=[row.as_dict() for row in windows(inherited,'inherited windows')] if inherited else None
        if explicit and inherited and explicit!=inherited:raise ValueError('Conflicting explicit and inherited per-recording windows: '+movie)
        selected=explicit or inherited or [row.as_dict() for row in request.windows] or [row.as_dict() for row in windows(default_windows or [],'inherited shared windows')]
        if not selected or not any(row['baseline'] is not None for row in selected):raise ValueError('Each recording requires a baseline and at least one comparison window')
        baseline_names={row['baseline'] for row in selected if row['baseline'] is not None}
        if any(row['name'] in baseline_names and row['baseline'] is not None for row in selected):raise ValueError('A baseline cannot itself be a response window')
        if len({row['coordinate']=='frames' for row in selected})>1:raise ValueError('Frame and hour windows cannot mix within one recording; preserve one explicit coordinate contract')
        normal=[]
        for row in selected:
            row=dict(row);offset=anchor['hours'] if row['coordinate']=='relative_hours' else 0
            if row['coordinate']!='frames':
                row.update(start_hours=row['start']+offset,end_hours=row['end']+offset)
                if row['name'] in baseline_names and row['end_hours']>anchor['hours']:raise ValueError('Baseline must end at or before the actual intervention/control anchor')
                if row['baseline'] is not None and row['start_hours']<anchor['hours']:raise ValueError('Response windows must begin at or after their anchor')
            else:
                frames=[tables[name].loc[tables[name].stem.eq(movie),['frame_index','hours']] for name in names if 'frame_index' in tables[name] and 'hours' in tables[name]]
                if not frames:raise ValueError('Frame windows require an original frame/hour clock to verify their anchor relationship')
                observed=pd.concat(frames).drop_duplicates()
                held=observed.loc[observed.frame_index.ge(row['start'])&observed.frame_index.lt(row['end'])]
                if row['name'] in baseline_names and held.hours.ge(anchor['hours']).any():raise ValueError('Baseline frame window contains observations after its anchor')
                if row['baseline'] is not None and held.hours.lt(anchor['hours']).any():raise ValueError('Response frame window contains observations before its anchor')
            normal.append(row)
        key_start='start' if normal[0]['coordinate']=='frames' else 'start_hours';key_end='end' if key_start=='start' else 'end_hours'
        ordered=sorted(normal,key=lambda row:row[key_start])
        if any(a[key_end]>b[key_start] for a,b in zip(ordered,ordered[1:])):raise ValueError('Windows overlap; an observation cannot belong to two comparison windows')
        for measured in measurements:
            if 'window' not in measured.grain:continue
            source=tables[measured.table];source=source.loc[source.stem.eq(movie)]
            for row in source.to_dict('records'):
                original=next((item for item in normal if item['name']==row['window']),None)
                if original is None:continue
                if row['window_coordinate']!=original['coordinate'] or row['window_start']!=original['start'] or row['window_end']!=original['end']:
                    raise ValueError('Saved window summary cannot be assigned to different requested bounds')
                if original['coordinate']=='relative_hours' and row['window_anchor_hours']!=anchor['hours']:
                    raise ValueError('Saved relative window summary belongs to another original anchor')
        assignment=sample_lookup[movie];role=(anchor['kind'],request.conditions.get(movie))
        if assignment.confirmed:
            if assignment.sample in sample_roles and sample_roles[assignment.sample]!=role:raise ValueError('One biological sample has conflicting treatment/control or condition assignments')
            sample_roles[assignment.sample]=role
        recordings[movie]={'anchor':anchor,'windows':normal,'window_source':'recording' if explicit or inherited else 'shared',
            'sample':assignment.as_dict(),'condition':request.conditions.get(movie),'boundary_rule':'start inclusive, end exclusive'}
    for pair in request.matching:
        a,b=pair['reference_sample'],pair['target_sample']
        if a not in sample_roles or b not in sample_roles:raise ValueError('Matching names an unconfirmed or unavailable biological sample')
        if sample_roles[a][0]!='control' or sample_roles[b][0]!='intervention':raise ValueError('Matching reference must be a control sample and target an intervention sample')
    if 'frame_summary' in tables and {'stem','frame_index','hours'}<=set(tables['frame_summary']):names.add('frame_summary')
    fingerprints={}
    for name in sorted(names):
        value=input_hashes.get(name)
        if re.fullmatch('[a-fA-F0-9]{64}',str(value)) is None:raise ValueError('Verified original SHA-256 required for '+name)
        fingerprints[name]=value.lower()
    rhythm={}
    if request.rhythms['enabled']:
        from pymicroglia.pipelines.rhythm.discovery import RhythmDiscoveryRequest, resolve_request as resolve_rhythm
        columns=request.rhythms.get('measurements',[m.column for m in measurements])
        if not isinstance(columns,list) or not columns or set(columns)-{m.column for m in measurements}:raise ValueError('Rhythm measurements must be selected intervention measurements')
        if any('window' in m.grain for m in measurements if m.column in columns):raise ValueError('Optional rhythm changes require original time series, not saved window summaries')
        specification={'pipeline':'rhythm-discovery','test_measurements':[{'column':m.column,'table':m.table} for m in measurements if m.column in columns],
            'analysis_options':request.rhythms.get('analysis_options',{}),'biological_samples':request.biological_samples.as_dict()}
        for key in ['settings_profile','correction_scope']:
            if key in request.rhythms:specification[key]=request.rhythms[key]
        native=resolve_rhythm(RhythmDiscoveryRequest.from_dict(specification,{}),source_run=source_run,tables=tables,input_hashes=input_hashes,rhythm_params=rhythm_params,table_grains=declared)
        rhythm={key:value for key,value in native.as_dict().items() if key not in {'request','inputs','test_measurements','comparison_measurements'}}
    return ResolvedInterventionRequest(request,InputIdentity(source_run,Settings(fingerprints),cells,samples),measurements,Settings(recordings),Settings(rhythm))


SCIENCE=(StepSpec('intervention-design','intervention-design',inputs=('measured-tables',)),
    StepSpec('aligned-windows','aligned-windows',('intervention-design',)),
    StepSpec('response-evidence','response-evidence',('aligned-windows',)),
    StepSpec('control-comparisons','control-comparisons',('aligned-windows','response-evidence')),
    StepSpec('response-timing','response-timing',('aligned-windows','response-evidence')),
    StepSpec('rhythm-changes','rhythm-changes',('aligned-windows',)),
    StepSpec('coordinated-responses','coordinated-responses',('aligned-windows','response-evidence')))
RENDERS=(StepSpec('response-overview','response-overview',tuple(s.name for s in SCIENCE),kind='render'),
    StepSpec('cell-reports-and-grids','cell-reports-and-grids',tuple(s.name for s in SCIENCE),kind='render'),
    StepSpec('timing-and-rhythm-figures','timing-and-rhythm-figures',('aligned-windows','response-timing','rhythm-changes'),kind='render'),
    StepSpec('sample-and-pattern-figures','sample-and-pattern-figures',('aligned-windows','control-comparisons','coordinated-responses'),kind='render'))
RECIPE=PipelineRecipe('intervention-response',1,(*SCIENCE,*RENDERS,StepSpec('linked-results-index','linked-results-index',tuple(s.name for s in (*SCIENCE,*RENDERS)),kind='render')))


def identity(context):
    from pymicroglia.pipelines._screening import read_verified_tables, file_hash
    read_verified_tables(context.table_paths,context.request.inputs.table_hashes)
    return content_id({'request':context.request.scientific_id,'code':file_hash(__file__)})


def freeze_design(context):
    from pymicroglia.pipelines._screening import _write_json, file_hash
    context.output.mkdir(parents=True);path=context.output/'intervention_design.json'
    _write_json(path,{'schema_version':1,'scientific_id':context.scientific_id,'request':context.request.as_dict(),'scientific_evaluation':False,
        'event_meaning':'One explicitly declared intervention or control comparison anchor per recording; control anchors are not treatment administrations',
        'population':'Every requested original cell/measurement/window; report selections do not redefine tests or experimental units'})
    return StepResult(context.step.name,context.scientific_id,'completed','Frozen actual anchors, windows, measurements and biological sample design',
        (ArtifactRef('intervention_design',path.name,file_hash(path),context.scientific_id),))


def _pending(context):
    from pymicroglia.pipelines._runner import Unavailable
    raise Unavailable('Required '+context.step.name+' producer is awaiting its implementation stage')


def producers():
    from pymicroglia.pipelines._runner import Producer
    import pymicroglia.pipelines.intervention.windows as intervention_windows
    import pymicroglia.pipelines.intervention.evidence as intervention_evidence
    import pymicroglia.pipelines.intervention.controls as intervention_controls
    import pymicroglia.pipelines.intervention.timing as intervention_timing
    import pymicroglia.pipelines.intervention.rhythms as intervention_rhythms
    import pymicroglia.pipelines.intervention.patterns as intervention_patterns
    import pymicroglia.pipelines.intervention.overview as intervention_overview
    import pymicroglia.pipelines.intervention.reports as intervention_reports
    import pymicroglia.pipelines.intervention.timing_figures as intervention_timing_figures
    import pymicroglia.pipelines.intervention.sample_figures as intervention_sample_figures
    import pymicroglia.pipelines.intervention.index as intervention_index
    return {**{step.producer:Producer(_pending) for step in RECIPE.steps},'intervention-design':Producer(freeze_design,identity),
        'aligned-windows':Producer(intervention_windows.produce,intervention_windows.identity),
        'response-evidence':Producer(intervention_evidence.produce,intervention_evidence.identity),
        'control-comparisons':Producer(intervention_controls.produce,intervention_controls.identity),
        'response-timing':Producer(intervention_timing.produce,intervention_timing.identity),
        'rhythm-changes':Producer(intervention_rhythms.produce,intervention_rhythms.identity),
        'coordinated-responses':Producer(intervention_patterns.produce,intervention_patterns.identity),
        'response-overview':Producer(intervention_overview.produce,version=intervention_overview.version(),accepts_unavailable_dependencies=True),
        'cell-reports-and-grids':Producer(intervention_reports.produce,version=intervention_reports.version(),accepts_unavailable_dependencies=True),
        'timing-and-rhythm-figures':Producer(intervention_timing_figures.produce,version=intervention_timing_figures.version(),accepts_unavailable_dependencies=True),
        'sample-and-pattern-figures':Producer(intervention_sample_figures.produce,version=intervention_sample_figures.version(),accepts_unavailable_dependencies=True),
        'linked-results-index':Producer(intervention_index.produce,version=intervention_index.version(),accepts_unavailable_dependencies=True)}


def run_request(resolved,table_paths,output,*,presentation=None,only=None):
    from pymicroglia.pipelines._runner import run_pipeline
    return run_pipeline(RECIPE,producers(),request=resolved,scientific_settings={'intervention':resolved.scientific_id},table_paths=table_paths,
        output=output,presentation=presentation,only=only)
