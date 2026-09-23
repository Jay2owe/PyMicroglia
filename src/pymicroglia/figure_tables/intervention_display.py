"""Prepare original saved intervention rows and categorical matrices."""
import numpy as np
OUTCOMES=('increase','decrease','no_detected_change','inconclusive','detected_below_meaningful_threshold')

def prepare(values,settings,panel):
    all_rows=values.to_dict('records')
    rows=[row for row in all_rows if row['entry_id'] in settings['plotted_entry_ids']] if panel=='intervention_samples' else all_rows
    result={'rows':rows,'all_rows':all_rows}
    if settings['view']=='status_matrix':
        ids,metrics=settings['cell_ids'],settings['measurements']
        matrix=np.full((len(ids),len(metrics)),np.nan)
        seen=set()
        for row in rows:
            position=(ids.index(row['cell_id']),metrics.index(row['measurement']))
            if position in seen:raise ValueError('Saved status matrix repeats an original cell/measurement')
            seen.add(position)
            matrix[position]=OUTCOMES.index(row['outcome']) if row['outcome'] in OUTCOMES else OUTCOMES.index('inconclusive')
        result['matrix']=np.ma.masked_invalid(matrix)
    elif settings['view']=='joint_outcomes':
        result['matrix']=np.asarray([[OUTCOMES.index(row[role+'_outcome']) for role in ('reference','target')] for row in rows],dtype=float).reshape(len(rows),2)
    return result
