"""Prepare state criteria and observed profiles before plotting."""
import math
from pymicroglia.visualisation.panels.behaviour_profiles import criterion,observed,wrap
from pymicroglia.visualisation.panels._format import present,number

def prepare(values,settings):
    meta=settings['metadata']
    result={}
    if settings['view']=='support':
        rows=[];refit=0
        for row in values.to_dict('records'):
            name=str(row['criterion'])
            if name.startswith('grouped_refit:'):refit+=1
            label=wrap(criterion(name,refit),43)
            measured=wrap(observed(row['observed'],name),29)
            required=wrap(observed(row['required'],name),29)
            reason=row.get('reason')
            if isinstance(reason,str) and reason.strip():measured=wrap(reason,63);required=''
            lines=max(label.count('\n'),measured.count('\n'),required.count('\n'))+1
            rows.append(dict(label=label,measured=measured,required=required,status=row['status'],height=max(.72,.22*lines+.25)))
        result.update(rows=rows,total=sum(row['height'] for row in rows))
    else:
        info={row['state_id']:row for row in meta['states']}
        features=[]
        for feature in settings['features']:
            selected=values.loc[values.measurement.eq(feature['measurement']) & values.source_table.eq(feature['source_table']) & values.representation.eq(feature['representation'])]
            lookup=selected.set_index('state_id')
            rows=[]
            for state in settings['states']:
                row=lookup.loc[state]
                median,lower,upper=[number(row[name]) for name in ('median','q25','q75')]
                counts=f"{int(row['observed_values'])} values; {int(row['missing_values'])} missing\n{int(row['cells'])} cells; {int(row['movies'])} recordings; {int(row['confirmed_samples'])} samples"
                rows.append(dict(label=settings['state_display_names'].get(state,info[state]['label']),component=info[state]['component'],median=median,lower=lower,upper=upper,valid=all(math.isfinite(v) for v in (median,lower,upper)),counts=counts))
            unit=str(selected.unit.iloc[0]) if present(selected.unit.iloc[0]) and str(selected.unit.iloc[0]) else 'unit not recorded'
            features.append(dict(rows=rows,measurement=feature['measurement'],representation=feature['representation'],unit=unit))
        result['features']=features
        counts=meta.get('assignment_status_counts',{})
        total=sum(counts.values());assigned=counts.get('assigned',0)
        result['assignment_note']=f'Assigned: {assigned:,} / {total:,} original observations; {total-assigned:,} without a state' if counts else ''
    return result
