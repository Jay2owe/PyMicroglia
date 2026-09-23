"""A bounded reproducible spatial-coordination software example, not cell data."""
from pymicroglia._results import read_document
from pathlib import Path
from tests.pipelines.invoke import invoke, source_identity, write_json as _write_json
from contextlib import ExitStack
from unittest.mock import patch
import argparse,json,shutil,subprocess,sys,tempfile

from pymicroglia.pipelines._screening import file_hash


def declaration():
    """Fixed questions and null assumptions for these constructed controls only."""
    return {'pipeline':'spatial-coordination','reference_measurements':['custom_signal'],
        'target_measurements':['custom_signal'],'pairs':{'mode':'cartesian'},'representation':'raw',
        'questions':{'simultaneous':{'enabled':True,'statistic':'pearson','evidence':{
            'method':'truncated_time_shift','radius_hours':24.,'stationary_series':'target',
            'stationarity_justification':'Constructed stationary noise and a separately recorded software reference; not an assumption about biological cells'}}},
        'geometry':{'x':'centroid_x','y':'centroid_y','definition':'centroid','input_unit':'px','unit':'px',
            'scale':[1,1],'neighbourhood':{'method':'all'}},
        'support':{'matching':'exact','matching_tolerance_hours':0,'max_gap_hours':.75,'min_observations':24,'min_span_hours':12.},
        'shared_reference':{'method':'external','measurements':{'custom_signal':'measured_reference'},
            'adjustment':'subtract','units':{'custom_signal':'controlled units','measured_reference':'controlled units'}},
        'table_grains':{'recording_reference':['frame_index']},
        'inference':{'alpha':.05,'multiple_testing':'bh','correction_scope':'question'},
        'biological_samples':{movie:'software-sample-'+movie for movie in ['same','opposed','common','independent','gapped','short']}}


def prepare(output):
    import numpy as np
    import pandas as pd
    from copy import deepcopy
    output=Path(output).resolve()
    if output.exists() and any(output.iterdir()):raise ValueError('Use a new empty directory, or --verify-existing to reopen the unchanged example')
    directory=output/'run/pooled/tables';directory.mkdir(parents=True)
    rng=np.random.default_rng(928211);rows=[];references=[]
    movies=['same','opposed','common','independent','gapped','short']
    for movie in movies:
        n=6 if movie=='short' else 180
        x=rng.normal(size=n);y=x.copy() if movie in ['same','gapped'] else -x if movie=='opposed' else rng.normal(size=n)
        common=rng.normal(scale=10.,size=n) if movie=='common' else np.zeros(n)
        for frame,value in enumerate(common):references.append({'stem':movie,'frame_index':frame,'hours':50.+frame*.5,'measured_reference':value})
        for index,values in enumerate([x+common,y+common]):
            for frame,value in enumerate(values):
                rows.append({'stem':movie,'identity':index+1,'frame_index':frame,'hours':50.+frame*.5,
                    'centroid_x':float(index*4),'centroid_y':0.,
                    'custom_signal':np.nan if movie=='gapped' and index==1 and frame==35 else float(value)})
    frame=pd.DataFrame(rows);frame.to_csv(directory/'cell_frame.csv',index=False)
    frame[['stem','identity']].drop_duplicates().to_csv(directory/'cell_summary.csv',index=False)
    pd.DataFrame(references).to_csv(directory/'recording_reference.csv',index=False)
    _write_json(output/'run/manifest.json',{'synthetic':True,'description':'Constructed spatial/temporal controls; no biological findings',
        'movies':[{'stem':movie,'modules':[]} for movie in movies]})
    request=declaration();_write_json(output/'supported-request.json',request)
    no_support=deepcopy(request);no_support['inference']['multiple_testing']='bonferroni';_write_json(output/'no-support-request.json',no_support)
    insufficient=deepcopy(request);insufficient['support']['min_observations']=10000;_write_json(output/'insufficient-request.json',insufficient)
    _write_json(output/'presentation.json',{'overview':{'views':['coverage','effects']},'maps':{'edge_limit':2},'pair_cards':{'pair_limit':2},
        'report':{'title':'Spatial coordination software example'}})
    _write_json(output/'software-truth.json',{'biological_result':False,'seed':928211,'recordings':6,'cells':12,'observations':len(frame),
        'known_relationships':{'same':'positive association','opposed':'negative association, not a phase claim',
            'common':'raw shared signal; its actual recorded reference explains the raw association',
            'independent':'independently generated processes','gapped':'one original unknown observation','short':'six original observations'},
        'prespecified_variants':'BH support, a more conservative Bonferroni family and deliberately insufficient declared support; these are software controls, not an outcome-driven choice for biological data'})
    return output


def verify(output):
    import pandas as pd
    from pymicroglia.pipelines import parse
    import pymicroglia.pipelines.coordination.inputs as coordination_inputs
    import pymicroglia.pipelines.coordination.simultaneous as coordination_simultaneous
    import pymicroglia.pipelines.coordination.temporal_inputs as coordination_temporal_inputs
    import pymicroglia.pipelines.coordination.evidence as coordination_evidence
    from pymicroglia.pipelines.coordination.options import resolve_request, run_request
    from pymicroglia.pipelines.coordination.index import validate_links
    from tests.pipelines.audit.demo import _latest
    import pymicroglia.workbench as circadian
    output=Path(output).resolve();manifest=output/'run/manifest.json'
    paths={path.stem:path for path in (output/'run/pooled/tables').glob('*.csv')}
    original={str(path):file_hash(path) for path in [*paths.values(),manifest]}
    tables={key:pd.read_csv(path) for key,path in paths.items()};appearance=read_document(output/'presentation.json')
    proofs=[]
    def forbidden(*a,**k):raise AssertionError('Presentation-only execution repeated scientific calculations')
    for case in ['supported','no-support','insufficient']:
        request=read_document(output/(case+'-request.json'))
        invoke([sys.executable,'-m','analysis','pipeline',str(output/'run'),'--request',str(output/(case+'-request.json')),
            '--out',str(output/'pipeline'),'--presentation',str(output/'presentation.json')],check=True)
        root=output/'pipeline/spatial-coordination';record,saved=_latest(root)
        evidence=coordination_evidence.read_evidence(saved['coordination-evidence']);effects=evidence['effects'];pairs=evidence['pair_decisions']
        assert len(pairs)==6 and len(effects)==12
        assert effects.loc[effects.movie.isin(['gapped','short']),'p_value'].isna().all()
        if case=='supported':
            selected=pairs.loc[pairs.supported_effect_ids.map(bool)];assert {'same','opposed','common'}==set(selected.movie)
            common=effects.loc[effects.movie.eq('common')].set_index('adjustment')
            assert common.loc['none','effect']>.98 and abs(common.loc['shared_reference','effect'])<.2
            assert common.loc['none','supported'] and not common.loc['shared_reference','supported']
        else:
            assert not pairs.supported_effect_ids.map(bool).any()
            assert saved['pair-report-cards'].outcome.status=='skipped-empty' and saved['connection-maps'].outcome.status=='skipped-empty'
        report=saved['linked-results-index'];checked=validate_links(report.root);navigation=read_document(report.artifact('navigation.json'))
        assert len(navigation['pairs'])==6 and len(navigation['cells'])==12
        before={str(item.artifact(ref.name)):file_hash(item.artifact(ref.name)) for item in saved.values() for ref in item.outcome.artifacts}
        with tempfile.TemporaryDirectory(prefix='spatial-demo-report-') as temporary:
            target=Path(temporary)/'report';shutil.copytree(report.root,target);assert validate_links(target)==checked
        resolved=resolve_request(parse([request])[0],source_run=source_identity(manifest.parent),tables=tables,input_hashes={key:file_hash(path) for key,path in paths.items()})
        with ExitStack() as stack:
            for owner,name in [(coordination_inputs,'prepare'),(coordination_simultaneous,'analyse'),(coordination_temporal_inputs,'adjusted_pair'),
                (coordination_evidence,'resolve_evidence'),(circadian,'adjust_pvalues')]:stack.enter_context(patch.object(owner,name,forbidden))
            for _ in range(3):
                reopened=run_request(resolved,paths,root,only=['linked-results-index'],presentation={**appearance,'report':{'title':'Reopened '+case+' software example'}})
                assert reopened.successful
            assert reopened.results['linked-results-index'].outcome.status=='reused'
        assert all(file_hash(Path(path))==value for path,value in {**original,**before}.items())
        proof={'case':case,'biological_result':False,'execution':str(record),'reopened_execution':str(reopened.record_path),
            'report':str(report.root),'pages':len(navigation['pages']),**checked,'source_files_unchanged':len(original),
            'artifacts_unchanged':len(before),'presentation_repeated_no_science':True,'relocated_links_verified':True}
        proofs.append(proof);_write_json(output/'verification.json',proofs);print(json.dumps(proof),flush=True)
    return proofs


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--verify-existing',action='store_true');parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    if not args.verify_existing:prepare(args.out)
    if not args.prepare_only:verify(args.out)


if __name__=='__main__':main()
