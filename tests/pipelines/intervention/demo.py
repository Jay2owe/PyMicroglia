"""Reproducible intervention software controls through the public pipeline command."""
from pymicroglia._results import read_document
from pathlib import Path
from tests.pipelines.invoke import invoke, source_identity, write_json as _write_json
from copy import deepcopy
from contextlib import ExitStack
from unittest.mock import patch
import argparse,json,subprocess,sys,tempfile,shutil
import numpy as np
import pandas as pd
from pymicroglia.pipelines._screening import file_hash

CASES=('responses','timing','rhythms','insufficient')


def generated(hours,origin,scenarios,*,label,missing=None):
    """All waveform, drift, noise and disturbance generation belongs to Workbench."""
    import pymicroglia.workbench as circadian
    design={'replicates':1,'scenarios':scenarios,'truth_policy':{'min_observations':24,'min_cycles':3.,
        'period_min_hours':2.,'period_max_hours':16.,'relative_tolerance':.1,'absolute_tolerance_hours':0.,'target':'all','extra_components':'penalize'}}
    profile={'id':label,'hours':list(map(float,hours)),'origin_hours':float(origin),'metadata':{'synthetic':True}}
    if missing is not None:profile['missing']=list(map(bool,missing))
    source=circadian.generate_benchmark_cases(design,[profile],partition='development',seed=88317029)
    traces={row['case_id']:row for row in source['traces']}
    return {row['scenario']:traces[row['case_id']]['values'] for row in source['cases']},source


def windows(width):return [{'name':'baseline','coordinate':'relative_hours','start':-width,'end':0.},
    {'name':'followup','coordinate':'relative_hours','start':0.,'end':width,'baseline':'baseline'}]


def response_inputs():
    rows=[];clocks=[];sources=[];anchors={};samples={};conditions={};matching=[]
    recordings=[(role+str(i),role,i,[7]) for i in range(8) for role in ['reference','target']]+[('reference-repeat','reference',0,[7,8])]
    for movie,role,index,identities in recordings:
        anchor=100.+13.*index;anchors[movie]={'hours':anchor,'kind':'control' if role=='reference' else 'intervention','label':'Declared software comparison'}
        samples[movie]=role+'-sample-'+str(index);conditions[movie]=role
        relative=(np.arange(192)-96)*.5;hours=anchor+relative
        clocks.extend({'stem':movie,'frame_index':i,'hours':float(t)} for i,t in enumerate(hours))
        if movie=='target1':identities=[7,8]
        for identity in identities:
            missing=(relative>=0) if movie=='target0' else (relative<-3) if movie=='target1' and identity==8 else (relative==-3) if movie=='target2' else np.zeros(len(hours),bool)
            changes=[0.,0.] if role=='reference' else [4.+.4*index,-2.-.2*index]
            scenarios=[{'id':metric,'components':[],'baseline':baseline,'drift':{'linear_per_hour':trend},
                'noise':{'kind':'exponential','sd':.25,'correlation_hours':.55},
                'disturbances':[{'kind':'step','start_hours':0.,'amplitude':change}]}
                for metric,baseline,trend,change in zip(['reporter_custom','shape_custom'],[10.,20.],[.012,-.006],changes)]
            values,source=generated(hours,anchor,scenarios,label=movie+'-cell-'+str(identity),missing=missing);sources.append(source)
            for frame,t in enumerate(hours):rows.append({'stem':movie,'identity':identity,'frame_index':frame,'hours':float(t),**{metric:value[frame] for metric,value in values.items()}})
    matching=[{'match_id':'declared-pair-'+str(i),'reference_sample':'reference-sample-'+str(i),'target_sample':'target-sample-'+str(i)} for i in range(8)]
    request={'pipeline':'intervention-response','measurements':['reporter_custom','shape_custom'],'summary':'mean','anchors':anchors,'windows':windows(48.),
        'support':{'max_gap_hours':.6},'biological_samples':samples,'conditions':conditions,'matching':matching,
        'evidence':{'method':'segmented_glsar','estimand':'baseline_trend_adjusted_change','baseline_trend':'linear','response_model':'level','ar_order':1,
            'model_justification':'Workbench-generated linear baseline plus a declared constant level step; no oscillatory mean model is asserted',
            'error_model_justification':'Workbench-generated stationary Gaussian exponential covariance sampled uniformly; one-lag autoregressive errors'},
        'inference':{'alpha':.05,'multiple_testing':'bonferroni','correction_scope':'all'},
        'controls':{'enabled':True,'aggregation':'mean','evidence':{'method':'matched_sample_permutation',
            'matched_samples':'Eight declared independent software pairs, with label exchangeability within each pair under the null',
            'interval':{'method':'paired_bootstrap','confidence':.95,'resamples':999,'seed':85}},
            'comparisons':[{'name':metric+'-change','measurement':metric,'baseline':'baseline','target_window':'followup',
                'reference_condition':'reference','target_condition':'target','quantity':'absolute_change','design':'matched'} for metric in ['reporter_custom','shape_custom']]},
        'coordinated':{'enabled':True,'pairs':{'mode':'explicit','pairs':[['reporter_custom','shape_custom']]},'aggregation':'mean',
            'evidence':{'method':'independent_sample_permutation','resamples':999,'seed':29,
                'independence_justification':'Original software samples are independent within each separately evaluated condition',
                'exchangeability_justification':'Paired sample changes are exchangeable within their condition under the software association null'}}}
    truth={'cells':19,'recordings':17,'samples':16,'matched_pairs':8,'complete_control_matches':7,
        'positive_measurement':'reporter_custom','negative_measurement':'shape_custom','null_condition':'reference',
        'missing_followup':'target0','incomplete_baseline':'target1/cell8','gapped_trace':'target2/cell7',
        'common_time_trend':'Each condition has the same declared linear baseline trend, distinct from the treatment step',
        'noise':'Native stationary exponential-covariance Gaussian errors; recording repetitions do not create new biological samples'}
    return pd.DataFrame(rows),pd.DataFrame(clocks),request,sources,truth


def timing_inputs():
    rows=[];clocks=[];sources=[];anchors={}
    for index,movie in enumerate(['increase','decrease','recovered','never','lost','gapped','short-baseline']):
        anchor=50.+7*index;relative=np.arange(-8.,20.);hours=relative+anchor
        anchors[movie]={'hours':anchor,'kind':'intervention','label':'Declared software event'}
        events=[] if movie=='never' else [{'kind':'pulse' if movie in {'recovered','gapped'} else 'step','start_hours':2. if movie!='decrease' else 3.,
            'amplitude':-10. if movie=='decrease' else 10.,**({'duration_hours':8.} if movie in {'recovered','gapped'} else {})}]
        missing=(relative>8) if movie=='lost' else (relative==4) if movie=='gapped' else (relative<-1) if movie=='short-baseline' else np.zeros(len(hours),bool)
        values,source=generated(hours,anchor,[{'id':'reporter_custom','components':[],'baseline':10.,'disturbances':events}],label=movie,missing=missing);sources.append(source)
        for frame,t in enumerate(hours):
            clocks.append({'stem':movie,'frame_index':frame,'hours':float(t)})
            if movie=='lost' and relative[frame]>8:continue
            rows.append({'stem':movie,'identity':7,'frame_index':frame,'hours':float(t),'reporter_custom':values['reporter_custom'][frame]})
    request={'pipeline':'intervention-response','measurements':['reporter_custom'],'summary':'mean','anchors':anchors,
        'windows':[windows(8.)[0],{**windows(20.)[1]}],'support':{'max_gap_hours':1.1},'evidence':{'method':'none'},
        'timing':{'enabled':True,'method':'observed_threshold_episodes','baseline_reference':'saved_window_summary',
            'thresholds':{'reporter_custom':{'change':5.,'direction':'either','unit':'software units'}},'persistence_hours':2.,
            'recovery':{'enabled':True,'reference':'same_baseline','tolerances':{'reporter_custom':{'value':1.,'unit':'software units'}},'persistence_hours':2.}}}
    return pd.DataFrame(rows),pd.DataFrame(clocks),request,sources,{'cells':7,'recordings':7,'samples':7,
        'delay_hours':{'increase':2.,'decrease':3.,'recovered':2.,'gapped':5.},'recovery_hours':{'recovered':10.,'gapped':10.},
        'lost_last_observation':8.,'interpretation':'Observed threshold episodes and sampling brackets, not statistical onset or equivalence tests'}


def rhythm_inputs():
    rows=[];clocks=[];sources=[];anchors={};overrides={}
    for movie in ['amplitude-phase','period','same','short','missing','gapped','detection-only']:
        width=8. if movie=='short' else 96.;anchors[movie]={'hours':width,'kind':'intervention','label':'Declared software boundary'}
        if width==8.:overrides[movie]=windows(width)
        for side in range(2):
            hours=side*width+np.arange(int(width/.25))*.25;period=10. if movie=='period' and side else 8.
            amplitude=4. if movie=='amplitude-phase' and side else 3.;phase=1. if movie=='amplitude-phase' and side else 0.
            components=[] if movie=='detection-only' and side else [
                {'id':'fundamental','waveform':'cosine','period_hours':period,'amplitude':amplitude,'phase_hours':phase},
                {'id':'harmonic','waveform':'cosine','period_hours':period/2.,'amplitude':1.,'phase_hours':phase}]
            missing=np.ones(len(hours),bool) if movie=='missing' and side else (np.arange(len(hours))==20) if movie=='gapped' and side else np.zeros(len(hours),bool)
            values,source=generated(hours,0.,[{'id':'reporter_custom','components':components,'baseline':10.,'noise':{'kind':'white','sd':.25}}],
                label=movie+'-'+str(side),missing=missing);sources.append(source)
            for index,t in enumerate(hours):
                frame=index+side*len(hours);clocks.append({'stem':movie,'frame_index':frame,'hours':float(t)})
                rows.append({'stem':movie,'identity':7,'frame_index':frame,'hours':float(t),'reporter_custom':values['reporter_custom'][index]})
    request={'pipeline':'intervention-response','measurements':['reporter_custom'],'summary':'mean','anchors':anchors,'windows':windows(96.),
        'recording_windows':overrides,'support':{'max_gap_hours':.3},'evidence':{'method':'none'},
        'rhythms':{'enabled':True,'measurements':['reporter_custom'],
            'analysis_options':{'fit_method':'fft_nlls','significance_method':'lomb','detrend':'none','period_min_hours':2.,'period_max_hours':16.,
                'min_observations':24,'min_cycles':3.,'multiple_testing':'bh','rhythmic_alpha':.05,
                'period_config':{'nlls_max_components':2,'nlls_circadian_min':6.,'nlls_circadian_max':12.}},
            'comparisons':[{'measurement':'reporter_custom','baseline':'baseline','target_window':'followup','component_period_band':[6.,12.]}],
            'direct_change':{'method':'native_fit_covariance','properties':['period','amplitude','phase'],
                'independent_window_errors':'Workbench generates separate independent white-noise streams for the two original windows',
                'residual_model_justification':'Declared two-component waveform plus homoskedastic independent Gaussian errors; negative control has no generated component',
                'phase_reference':'recording_anchor','period_equivalence_fraction':.05,'max_phase_extrapolation_cycles':.05}}}
    return pd.DataFrame(rows),pd.DataFrame(clocks),request,sources,{'cells':7,'recordings':7,'samples':7,'period_change_hours':2.,
        'amplitude_change':1.,'phase_change_cycles':.125,'short_window_hours':8.,'original_period_hours':8.,
        'interpretation':'Constructed non-daily components are software truths, not priors on microglia or a general method calibration'}


def appearance(case):
    return {'overview':{'views':['coverage','status_matrix','controls'] if case in {'responses','insufficient'} else ['coverage']},
        'reports':{'views':['trace_grid'] if case=='responses' else ['cell_report','trace_grid']},
        'timing_figures':{'views':['coverage','response_delay','recovery'] if case=='timing' else ['coverage','window_period','direct_change'] if case=='rhythms' else ['coverage']},
        'sample_figures':{'views':['coverage','control_effects','sample_pairs','recurrence'] if case=='responses' else ['coverage']},
        'report':{'title':'Intervention response | '+case+' software example'}}


def prepare(output):
    output=Path(output).resolve()
    if output.exists() and any(output.iterdir()):raise ValueError('Use a new empty directory, or --verify-existing to reopen the unchanged example')
    for case,make in [('responses',response_inputs),('timing',timing_inputs),('rhythms',rhythm_inputs)]:
        frame,clock,request,sources,truth=make();directory=output/case/'run/pooled/tables';directory.mkdir(parents=True)
        frame.to_csv(directory/'cell_frame.csv',index=False);clock.to_csv(directory/'frame_summary.csv',index=False)
        frame[['stem','identity']].drop_duplicates().to_csv(directory/'cell_summary.csv',index=False)
        generator=output/case/'native-generation.json';_write_json(generator,{'sources':sources,'local_changes':'Declared identity/recording labels and omission of lost-cell rows only; numerical generation belongs to Workbench'})
        _write_json(output/case/'run/manifest.json',{'synthetic':True,'description':'Constructed intervention software controls; no biological findings',
            'generator_sha256':file_hash(generator),'movies':[{'stem':movie,'modules':[]} for movie in request['anchors']]})
        _write_json(output/case/'request.json',request);_write_json(output/case/'presentation.json',appearance(case))
        _write_json(output/case/'software-truth.json',{'biological_result':False,'seed':88317029,**truth})
    insufficient=deepcopy(read_document(output/'responses/request.json'));insufficient['support']['min_observations']=10000
    (output/'insufficient').mkdir();_write_json(output/'insufficient/request.json',insufficient);_write_json(output/'insufficient/presentation.json',appearance('insufficient'))
    return output


def verify(output,cases=CASES):
    from pymicroglia.pipelines import parse
    import pymicroglia.pipelines.intervention.windows as intervention_windows
    import pymicroglia.pipelines.intervention.evidence as intervention_evidence
    import pymicroglia.pipelines.intervention.controls as intervention_controls
    import pymicroglia.pipelines.intervention.patterns as intervention_patterns
    import pymicroglia.pipelines.intervention.timing as intervention_timing
    import pymicroglia.pipelines.intervention.rhythms as intervention_rhythms
    import pymicroglia.pipelines.intervention.rhythm_windows as intervention_rhythm_windows
    from pymicroglia.pipelines.intervention.options import resolve_request, run_request
    from pymicroglia.pipelines.intervention.index import build, validate_links
    from tests.pipelines.audit.demo import _latest
    import pymicroglia.workbench as circadian
    output=Path(output).resolve();proof_path=output/'verification.json';proof=read_document(proof_path) if proof_path.exists() else []
    def forbidden(*a,**k):raise AssertionError('Saved display repeated scientific calculations')
    for case in cases:
        source=output/('responses' if case=='insufficient' else case);run=source/'run';manifest=run/'manifest.json'
        paths={p.stem:p for p in (run/'pooled/tables').glob('*.csv')};tables={key:pd.read_csv(path,dtype={'stem':str}) for key,path in paths.items()}
        request=read_document(output/case/'request.json');presentation=read_document(output/case/'presentation.json')
        original={str(p):file_hash(p) for p in [*paths.values(),manifest,source/'native-generation.json',output/case/'request.json']}
        invoke([sys.executable,'-m','analysis','pipeline',str(run),'--request',str(output/case/'request.json'),
            '--out',str(output/case/'pipeline'),'--presentation',str(output/case/'presentation.json')],check=True)
        target=output/case/'pipeline/intervention-response';record,saved=_latest(target);report=saved['linked-results-index']
        nav=read_document(report.artifact('navigation.json'));truth=read_document(source/'software-truth.json')
        assert len(nav['cells'])==truth['cells'] and len(nav['samples'])==truth['samples']
        effects=intervention_evidence.read_evidence(saved['response-evidence'])
        if case=='responses':
            changed=effects['effects'].loc[lambda x:x.movie.eq('target7')].set_index('measurement')
            assert changed.loc['reporter_custom','outcome']=='increase' and changed.loc['shape_custom','outcome']=='decrease'
            assert effects['effects'].loc[lambda x:x.movie.str.startswith('reference'),'outcome'].eq('no_detected_change').any()
            assert effects['families'].requested.sum()==len(effects['effects'])
            assert all(r['record']['complete_matches']==7 for r in nav['control_comparisons'])
            shared=next(r for r in nav['samples'] if r['record']['sample']=='reference-sample-0');assert len(shared['record']['movies'])==2
        elif case=='insufficient':
            assert not effects['effects'].response_supported.any() and effects['effects'].p_value.isna().all()
            assert saved['cell-reports-and-grids'].outcome.status=='skipped-empty'
        elif case=='timing':
            timing=intervention_timing.read_timing(saved['response-timing'])['timing'].set_index('movie')
            for movie,hours in truth['delay_hours'].items():assert timing.loc[movie,'response_delay_hours']==hours
            for movie,hours in truth['recovery_hours'].items():assert timing.loc[movie,'recovery_delay_hours']==hours
            assert pd.isna(timing.loc['lost','recovery_delay_hours']) and timing.loc['lost','observed_endpoint_relative_hours']==8.
            assert not timing.loc['never','response_observed'] and not timing.loc['short-baseline','response_observed']
        else:
            rhythm=intervention_rhythms.read_rhythms(saved['rhythm-changes']);direct=rhythm['direct_comparisons']
            selected=direct.loc[direct.movie.eq('amplitude-phase')].set_index('property');changed=direct.loc[direct.movie.eq('period')].set_index('property')
            assert selected.loc['amplitude','change_supported'] and abs(selected.loc['amplitude','effect']-1.)<.15
            assert selected.loc['phase','change_supported'] and abs(selected.loc['phase','effect']-.125)<.02
            assert changed.loc['period','change_supported'] and abs(changed.loc['period','effect']-2.)<.1 and pd.isna(changed.loc['phase','p_value'])
            assert direct.loc[direct.movie.isin(['short','missing','gapped','detection-only']),'p_value'].isna().all()
            assert not rhythm['window_results'].loc[lambda x:x.movie.eq('short'),'period_supported'].any()
            transition=rhythm['comparisons'].loc[lambda x:x.movie.eq('detection-only')].iloc[0]
            assert transition.baseline_detected and not transition.target_detected
        frozen={str(value.artifact(ref.name)):file_hash(value.artifact(ref.name)) for value in saved.values() for ref in value.outcome.artifacts}
        resolved=resolve_request(parse([request])[0],source_run=source_identity(manifest.parent),tables=tables,input_hashes={key:file_hash(path) for key,path in paths.items()})
        extra=[]
        with ExitStack() as stack:
            for module,name in [(intervention_windows,'prepare'),(intervention_evidence,'analyse'),(intervention_controls,'compare_units'),(intervention_patterns,'analyse'),
                (intervention_timing,'scan'),(intervention_rhythms,'analyse'),(intervention_rhythm_windows,'analyse'),(circadian,'adjust_pvalues'),(circadian,'rhythm_window_comparison')]:stack.enter_context(patch.object(module,name,forbidden))
            reopened=run_request(resolved,paths,target,presentation=presentation,only=['linked-results-index']);assert reopened.successful
            assert reopened.results['linked-results-index'].outcome.status=='reused'
            with tempfile.TemporaryDirectory(prefix='intervention-portable-') as temporary:
                copied=Path(temporary)/'report';shutil.copytree(report.root,copied)
                assert build(copied,nav['inputs'],resolved.as_dict())==nav and validate_links(copied)==validate_links(report.root)
            if case=='responses':
                display=deepcopy(presentation);display['overview']['rows_per_page']=8
                shown=run_request(resolved,paths,target,presentation=display,only=['response-overview']);assert shown.successful
                assert shown.results['response-evidence'].outcome.scientific_id==saved['response-evidence'].outcome.scientific_id
                assert shown.results['response-overview'].root!=saved['response-overview'].root
                extra.append(shown.results['response-overview'])
        changed_science=None
        if case=='responses':
            changed=deepcopy(request);changed['inference']['alpha']=.025
            different=resolve_request(parse([changed])[0],source_run=source_identity(manifest.parent),tables=tables,input_hashes={key:file_hash(path) for key,path in paths.items()})
            assert different.scientific_id!=resolved.scientific_id
            altered=run_request(different,paths,target,only=['response-evidence']);assert altered.successful
            changed_science=altered.results['response-evidence'].outcome.scientific_id
            assert changed_science!=saved['response-evidence'].outcome.scientific_id
        assert all(file_hash(Path(p))==sha for p,sha in {**original,**frozen}.items())
        batches=[]
        for value in [*saved.values(),*extra]:
            if any(ref.name=='intervention_display_manifest.json' for ref in value.outcome.artifacts):
                display=read_document(value.artifact('intervention_display_manifest.json'));batches.append({'case':case+'-'+value.outcome.step,'bundle':str(value.root),
                    'masters':len(display['pages']),'check_output':display['check_output']})
        result={'case':case,'source_execution':str(record),'reopen_execution':str(reopened.record_path),'report':str(report.root),'cells':len(nav['cells']),
            'samples':len(nav['samples']),'pages':len(nav['pages']),'link_validation':validate_links(report.root),'batches':batches,
            'original_sources_unchanged':len(original),'saved_artifacts_unchanged':len(frozen),'scientific_calls_forbidden_on_reopen':True,'changed_scientific_id':changed_science,
            'source_hashes':original,'saved_artifact_hashes':frozen}
        proof=[row for row in proof if row['case']!=case]+[result];_write_json(proof_path,proof)
        print(json.dumps({key:result[key] for key in ['case','report','cells','samples','pages','link_validation','original_sources_unchanged','saved_artifacts_unchanged']}),flush=True)
    return proof


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--verify-existing',action='store_true');parser.add_argument('--case',choices=CASES,action='append')
    args=parser.parse_args(argv)
    if not args.verify_existing:prepare(args.out)
    if not args.prepare_only:verify(args.out,args.case or CASES)
    return 0


if __name__=='__main__':raise SystemExit(main())
