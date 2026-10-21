"""Write explicit serial executions for outstanding action families."""
import json
from benchmark_pipeline_execution import ROOT

base=ROOT/'development/speed-pass-20261011/complete'
plan=[]
def add(name,*args):
    plan.append({'id':name,'args':list(args)})
    if '--baseline-ledger' in args:
        current=[arg for arg in args if arg!='--baseline-ledger']
        index=current.index('--label')+1
        current[index]=current[index].replace('-before','-after')
        if current[index]==args[args.index('--label')+1]:raise ValueError('Pair needs distinct output labels')
        plan.append({'id':name.replace('-before','-after'),'args':current})
add('remaining-core-before','tools/benchmark_core_actions.py','--label','extra-before','--baseline-ledger',
    '--actions','automatic_scn_outline','publication_workbook','cell_image_grid','cell_video_grid','trace_panel','follow','cry1_dluc_photon','track')
add('dataset-before','tools/benchmark_dataset_actions.py','--label','dataset-before','--baseline-ledger')
failed=['clock_face','cell_analysis_grid','all_cell_trace_grid','metric_rhythm_matrix',
        'second_verse','cd68_reporter_rhythm','period_method_audit','reporter_shape_timing',
        'breakout_triggered_average','stable_traits_versus_states','cell_group_comparison',
        'neighbour_coordination','phase_compass']
add('figures-selected-before','tools/benchmark_figure_actions.py','--label','selected-before',
    '--baseline-ledger','--options',str(base/'figure-inputs.json'),'--actions',*failed)
measured=base/'dataset-before/dataset/measurement/measured'
add('figures-long-before','tools/benchmark_figure_actions.py','--label','long-before',
    '--baseline-ledger','--run',str(measured),'--options',str(base/'figure-inputs.json'),
    '--actions','own_clock_composite','rhythm_strength','active_span','contact_ledger','independence_map')
add('figures-states-before','tools/benchmark_figure_actions.py','--label','states-before',
    '--baseline-ledger','--run',str(base/'dataset-before/dataset/states/states'),'--stem','',
    '--actions','state_histories')
original_figures=[]
for receipt in (base/'before1/figures').glob('*/receipt.json'):
    row=json.loads(receipt.read_text())
    if row.get('status')=='measured':original_figures.append(row['id'].split(':',1)[1])
add('figures-complete-before','tools/benchmark_figure_actions.py','--label','figures-complete-before',
    '--baseline-ledger','--actions',*sorted(original_figures))
original_core=[]
for folder in ('before1/core','media-before1/core'):
    for receipt in (base/folder).glob('*/receipt.json'):
        row=json.loads(receipt.read_text());name=row['id'].split(':',1)[1]
        if row.get('status')=='measured' and name!='follow':original_core.append(name)
add('core-complete-before','tools/benchmark_core_actions.py','--label','core-complete-before',
    '--baseline-ledger','--actions',*sorted(set(original_core)))
for family in ('rhythm','audit','relationships','behaviour','coordination','intervention'):
    add('pipeline-'+family+'-before','tools/benchmark_pipeline_execution.py',family,
        '--label','pipeline-'+family+'-before','--baseline-ledger')
for family,label in [('rhythm','before2'),('audit','before1'),('relationships','before1'),
                     ('behaviour','before1'),('coordination','before1'),('intervention','before1')]:
    source=base/label/family/'fixture'
    run=source/'responses/run' if family=='intervention' else source/'run'
    add('saved-'+family+'-before','tools/benchmark_figure_actions.py','--label','saved-'+family+'-before',
        '--baseline-ledger','--saved-run',str(run))
for case in ('timing','rhythms'):
    label='intervention-'+case+'-before'
    add(label,'tools/benchmark_pipeline_execution.py','intervention','--case',case,'--label',label,'--baseline-ledger')
    add('saved-'+label,'tools/benchmark_figure_actions.py','--label','saved-'+label,'--baseline-ledger',
        '--saved-run',str(base/label/'intervention/fixture'/case/'run'))
add('boundaries','tools/benchmark_workbench_boundaries.py','--label','boundaries1')
path=base/'remaining-executions.json';path.write_text(json.dumps(plan,indent=2),encoding='utf-8')
print(path)
