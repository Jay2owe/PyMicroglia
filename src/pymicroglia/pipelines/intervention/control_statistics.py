"""Declared treatment/control measurements with Workbench sample statistics."""
from copy import deepcopy
from pymicroglia import workbench
policy = workbench.sample_contrasts.matched_sample_policy
compare = workbench.sample_contrasts.matched_sample_contrast

def control_policy(block,measurements,inference):
    """Validate declarations without importing Workbench or fitting anything."""
    if not block['enabled']:return {'enabled':False}
    if block.get('aggregation') not in {'mean','median'}:raise ValueError('Control comparisons require explicit mean or median aggregation')
    comparisons=block.get('comparisons');default=block.get('evidence',{'method':'none'})
    if not isinstance(comparisons,list) or not comparisons:raise ValueError('Declare at least one treatment/control comparison')
    output=[];seen=set();names=set()
    from pymicroglia.pipelines._contracts import content_id, text_key
    for item in comparisons:
        keys={'name','measurement','baseline','target_window','reference_condition','target_condition','quantity','design','evidence'}
        if not isinstance(item,dict) or set(item)-keys:raise ValueError('Unknown treatment/control comparison field')
        item=deepcopy(item)
        for key in ['measurement','baseline','target_window','reference_condition','target_condition']:text_key(item.get(key),'controls.comparisons.'+key)
        if item['measurement'] not in measurements:raise ValueError('Control comparison uses an unrequested measurement')
        if item['baseline']==item['target_window'] or item['reference_condition']==item['target_condition']:raise ValueError('Control contrast requires distinct windows and conditions')
        if item.get('quantity') not in {'absolute_change','estimate','relative_change'}:raise ValueError('Choose original absolute_change, conditional-model estimate or declared relative_change')
        if item.get('design') not in {'independent','matched'}:raise ValueError('Declare independent or matched biological-sample comparison')
        item['evidence']=policy(item.get('evidence',default));method=item['evidence']['method']
        if method!='none' and not inference:raise ValueError('Formal control comparisons require declared inference settings')
        if method!='none' and (method=='matched_sample_permutation')!=(item['design']=='matched'):raise ValueError('Sample evidence method differs from the declared independent/matched design')
        definition={key:item[key] for key in ['measurement','baseline','target_window','reference_condition','target_condition','quantity','design']};key=content_id(definition)
        if key in seen:raise ValueError('Repeated treatment/control question')
        seen.add(key);item['name']=text_key(item.get('name',key[:16]),'comparison.name')
        if item['name'] in names:raise ValueError('Control comparison names must be unique')
        names.add(item['name']);item['comparison_spec_id']=key;output.append(item)
    return {'enabled':True,'aggregation':block['aggregation'],'comparisons':output}
