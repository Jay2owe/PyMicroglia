"""Controlled complete state-pipeline example; never biological evidence."""
from __future__ import annotations
from pymicroglia._results import read_document

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import shutil
from tests.pipelines.invoke import invoke, source_identity, write_json as _write_json
import sys
import tempfile
from unittest.mock import patch

from pymicroglia.pipelines._screening import file_hash, read_table


def declaration(samples=None, roles=None, conditions=None):
    """Explicit design for the controlled sample generator below."""
    return {'pipeline':'cell-behaviour-states','name':'synthetic-behaviour-states','features':['custom_signal','custom_shape'],
        'observation':{'kind':'frame'},'representation':'raw',
        'learning':{'balance':'sample_cell','max_observations_per_cell':60,'min_observations':12,'seed':17,
            'scaling':{'method':'standard'},'missing':{'method':'complete_case','max_fraction':0}},
        'candidates':[{'method':'gaussian_mixture','components':count,'covariance_type':'full','reg_covar':.001,'n_init':5,'max_iter':300,'seed':17} for count in [1,2]],
        'validation':{'unit':'biological_sample','split':{'method':'explicit','roles':roles or {}},
            'min_groups':{'learning':2,'development':2,'confirmation':3},
            'independence_justification':'Separately generated independent controlled samples; this is not a justification for experimental data'},
        'support':{'method':'heldout_multimodality','sampling_population':'Independent exchangeable controlled validation samples with the same feature-generating distribution'},
        'assignment':{'min_probability':.8,'outlier_quantile':.01,'min_observed_fraction':1},
        'statistics':{'interval_rule':'adjacent_midpoint','max_gap_hours':2,'transition_interval_hours':[.4,.6],'sample_aggregation':'mean',
            'comparison':{'method':'independent_sample_permutation','contrasts':[{'reference':'control','target':'treated'}],
                'metrics':['occupancy_observed','unknown_fraction','switch_rate'],'min_samples':4,'resamples':999,'seed':3,'alpha':.05,'multiple_testing':'bonferroni',
                'independent_samples':'Separately generated assignment-only samples unused for learning or choosing the state vocabulary',
                'interval':{'method':'independent_bootstrap','confidence':.95,'resamples':1000,'seed':31}}},
        'biological_samples':samples or {},'conditions':conditions or {}}


def prepare(output,scenario='accepted',images=False):
    """Create original clocks, protected populations and labelled software controls."""
    import numpy as np
    import pandas as pd
    if scenario not in {'accepted','continuous','cloud','insufficient'}:raise ValueError('Unknown controlled state scenario')
    output=Path(output).resolve()
    if output.exists() and any(output.iterdir()):raise ValueError('Prepare requires a new empty folder; --verify-existing reuses unchanged files')
    output.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(77013);rows=[];samples={};roles={};conditions={};truth=[]
    centres=np.array([[-4.,-2.],[4.,2.]])
    def add(movie,values,hours,role,sample=None,condition=None,frame_indices=None,purpose=None):
        samples[movie]=sample or movie;roles[samples[movie]]=role
        if condition:conditions[movie]=condition
        indices=list(range(len(values))) if frame_indices is None else frame_indices
        for index,hour,value in zip(indices,hours,values):
            rows.append({'stem':movie,'identity':7,'frame_index':index,'hours':hour,'custom_signal':float(value[0]),'custom_shape':float(value[1]),
                'imagej_frame':index+1,'source_imagej_frame':index+2})
        truth.append({'movie':movie,'identity':7,'purpose':purpose or role,'observations':len(values),'sample':samples[movie],'role':role})
    groups=8 if scenario=='insufficient' else 60
    for role,count in [('learning',4),('development',groups),('confirmation',groups)]:
        for group in range(count):
            n=120 if role=='learning' else 4
            if scenario=='continuous':
                x=rng.uniform(-4,4,n);values=np.column_stack([x,x+rng.normal(0,.12,n)])
            elif scenario=='cloud':values=rng.normal(0,1,(n,2))
            else:values=rng.normal(0,.25,(n,2))+centres[rng.integers(0,2,n)]
            add(role+'-'+str(group),values,50.+np.arange(n)*.5,role)
    if scenario=='accepted':
        independent=np.random.default_rng(9071)
        for condition in ['control','treated']:
            for group in range(6):
                n=40;fraction=(.1 if condition=='control' else .7)+.025*group
                labels=np.where(np.arange(n)<round(fraction*n),0,1)
                values=centres[labels]+independent.normal(0,.04,(n,2))
                movie=condition+'-independent-'+str(group)
                add(movie,values,50.+np.arange(n)*.5,'assignment_only',condition=condition,purpose='Independent condition comparison')
                if condition=='control' and group==0:
                    add(movie+'-second-recording',values.copy(),150.+np.arange(n)*.5,'assignment_only',sample=movie,condition=condition,purpose='Second recording of the same biological sample')
        for movie,n,start,spacing in [('gap',14,200.,.5),('unknown',12,350.,.5),('faster',14,500.,.25),('point',1,600.,.5)]:
            values=centres[np.arange(n)%2]+independent.normal(0,.04,(n,2));indices=list(range(n))
            if movie=='gap':indices.remove(5);values=values[indices]
            if movie=='unknown':values[3:5,0]=np.nan;values[8]=1e6
            add('timeline-'+movie,values,start+np.asarray(indices)*spacing,'assignment_only',frame_indices=indices,purpose='Original-time '+movie+' control')
        add('timeline-clock-reset',centres[[0,0,1,1]],[700.,700.5,700.25,701.],'assignment_only',purpose='Nonincreasing physical clock; duration unavailable')
    frame=pd.DataFrame(rows);tables=output/'run/pooled/tables';tables.mkdir(parents=True)
    frame.to_csv(tables/'cell_frame.csv',index=False)
    summary=frame[['stem','identity']].drop_duplicates()
    if scenario=='accepted':summary=pd.concat([summary,pd.DataFrame([{'stem':'timeline-point','identity':8}])],ignore_index=True)
    summary.to_csv(tables/'cell_summary.csv',index=False)
    movies=[]
    for ordinal,(movie,group) in enumerate(frame.groupby('stem',sort=True)):
        record={'stem':movie,'modules':[]}
        if images:
            import tifffile
            directory=output/'run/images';directory.mkdir(exist_ok=True)
            n=int(group.frame_index.max())+1;labels=np.zeros((n,24,24),dtype=np.uint16)
            for index in range(n):labels[index,8:16,8+index%3:16+index%3]=7
            raw=np.arange((n+1)*24*24,dtype=np.float32).reshape(n+1,24,24)+ordinal*100000
            inputs={}
            for kind,array in [('labels',labels),('raw',raw)]:
                path=directory/(movie+'-'+kind+'.tif');tifffile.imwrite(path,array,metadata={'axes':'TYX'},photometric='minisblack')
                inputs[kind]={'path':path.relative_to(output/'run').as_posix(),'sha256':file_hash(path)}
            record['provenance']={'inputs':inputs}
        movies.append(record)
    _write_json(output/'run/manifest.json',{'synthetic':True,'scenario':scenario,'description':'Controlled software-test measurements and optional procedural pixel arrays; not biological cells or findings','movies':movies})
    _write_json(output/'behaviour.json',declaration(samples,roles,conditions))
    _write_json(output/'presentation.json',{'report':{'title':'Controlled cell behaviour states: '+scenario},'state_timelines':{'state_cells_per_page':12},
        'state_summaries':{'state_summary_cells_per_page':32},'state_cards':{'state_card_images':images,'state_card_low_membership':True,
            'text':{'subtitle':'Controlled software fixture; no biological findings'}}})
    _write_json(output/'truth.json',{'biological_result':False,'scenario':scenario,'seed':77013,'independent_seed':9071,'cells':truth,
        'expected':'accepted' if scenario=='accepted' else 'inconclusive' if scenario=='insufficient' else 'no_supported_states',
        'scope':'Controlled separated or continuous feature distributions; numerical labels have no biological meaning. No periodic signal was assumed.'})
    _write_json(output/'fixture.json',{'files':{path.relative_to(output).as_posix():file_hash(path) for path in sorted(output.rglob('*')) if path.is_file()}})
    return output


def resolve(output):
    import pandas as pd
    from pymicroglia.pipelines import parse
    from pymicroglia.pipelines.behaviour.options import resolve_request
    output=Path(output);paths={path.stem:path for path in (output/'run/pooled/tables').glob('*.csv')}
    request=parse([read_document(output/'behaviour.json')])[0]
    return resolve_request(request,source_run=source_identity(output / "run"),tables={name:pd.read_csv(path) for name,path in paths.items()},
        input_hashes={name:file_hash(path) for name,path in paths.items()}),paths


def scientific_operations():
    import pymicroglia.states.behaviour_models as behaviour_models
    import pymicroglia.states.behaviour_support as behaviour_support
    import pymicroglia.workbench as circadian
    import pymicroglia.pipelines.behaviour.durations as behaviour_durations
    import pymicroglia.pipelines.behaviour.samples as behaviour_samples
    return [(behaviour_models,'fit_candidate'),(behaviour_models,'predict_candidate'),(behaviour_support,'measure'),
        (behaviour_durations,'statistics'),(behaviour_samples,'aggregate'),(behaviour_samples,'compare_samples'),(circadian,'detrend_trace'),(circadian,'adjust_pvalues')]


def verify(output):
    """Run all nodes, relocate every link, then reopen without scientific work."""
    from tests.pipelines.audit.demo import _latest
    from pymicroglia.pipelines.behaviour.options import run_request
    from pymicroglia.pipelines.behaviour.index import validate_links
    from pymicroglia.pipelines.rhythm.index import openable_report
    output=Path(output).resolve();fixture=read_document(output/'fixture.json')
    for name,fingerprint in fixture['files'].items():
        if file_hash(output/name)!=fingerprint:raise ValueError('Prepared fixture changed: '+name)
    invoke([sys.executable,'-m','analysis','pipeline',str(output/'run'),'--request',str(output/'behaviour.json'),
        '--presentation',str(output/'presentation.json'),'--out',str(output/'pipeline')],check=True)
    root=output/'pipeline/synthetic-behaviour-states';record,saved=_latest(root)
    assert len(saved)==13
    decision=read_document(saved['state-support'].artifact('support_decision'))
    truth=read_document(output/'truth.json');assert decision['status']==truth['expected'],decision
    report=saved['linked-results-index'];navigation=read_document(report.artifact('navigation.json'))
    if decision['status']=='accepted':
        assert all(item.outcome.status in {'completed','reused'} for item in saved.values())
        assert len(navigation['states'])==2 and navigation['examples']
        assignments=read_table(saved['state-assignments'].artifact('assignments'))
        assert assignments.observation_id.is_unique
        assert {'missing_features','outside_training_distribution'}<=set(assignments.status)
        assert len(navigation['cells'])==len(resolve(output)[0].inputs.cells)
        units=read_table(saved['state-sample-comparisons'].artifact('unit_inventory'))
        paired=units.loc[units['sample'].eq('control-independent-0')].iloc[0]
        assert paired.recordings==2 and paired.cells==2 and paired.independent_of_state_choice
        expected_units={row['sample'] for row in truth['cells'] if row['role']=='assignment_only'}
        assert set(units.loc[units.independent_of_state_choice,'sample'])==expected_units
    else:
        assert not navigation['states'] and not navigation['examples']
        assert saved['state-report-cards'].outcome.status=='skipped-empty'
    before={str(item.artifact(ref.name)):file_hash(item.artifact(ref.name)) for item in saved.values() for ref in item.outcome.artifacts}
    with tempfile.TemporaryDirectory(prefix='motion-state-report-') as temporary:
        moved=Path(temporary)/'report';shutil.copytree(report.root,moved);links=validate_links(moved)
    resolved,paths=resolve(output);appearance=read_document(output/'presentation.json')
    with ExitStack() as stack:
        def forbidden(*a,**k):raise AssertionError('Reopening a saved state report attempted scientific calculations')
        for owner,name in scientific_operations():stack.enter_context(patch.object(owner,name,forbidden))
        reopened=run_request(resolved,paths,root,presentation=appearance)
        assert reopened.successful,{name:item.outcome.reason for name,item in reopened.results.items()}
        # Diagnostic snapshots can record completed versus reused prerequisite
        # outcomes anew; they still consume only saved scientific evidence.
        assert all(item.outcome.status=='reused' for name,item in reopened.results.items() if name not in {'linked-results-index','state-support-figures'} and saved[name].outcome.status in {'completed','reused'})
    assert all(file_hash(Path(path))==fingerprint for path,fingerprint in before.items())
    opened=openable_report(reopened.results['linked-results-index'])
    proof={'biological_result':False,'execution':str(record),'reopened_execution':str(reopened.record_path),'decision':decision['status'],
        'nodes':len(saved),'cells':len(navigation['cells']),'states':len(navigation['states']),'bouts':len(navigation['bouts']),'examples':len(navigation['examples']),
        'masters':len(navigation['pages']),'moved_links':links,'source_artifacts_unchanged':len(before),'reopening_performed_no_science':True,'openable_report':str(opened)}
    _write_json(output/'verification.json',proof);print(json.dumps(proof,indent=2));return proof


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path)
    parser.add_argument('--scenario',choices=['accepted','continuous','cloud','insufficient'],default='accepted')
    parser.add_argument('--images',action='store_true',help='Include procedural software-test pixel arrays, not biological cell images')
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--verify-existing',action='store_true');args=parser.parse_args(argv)
    if not args.verify_existing:prepare(args.output,args.scenario,args.images)
    if not args.prepare_only:verify(args.output)


if __name__=='__main__':main()
