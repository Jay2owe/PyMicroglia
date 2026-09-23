"""Original observation segments, windows and saved evidence for intervention reports."""
from pymicroglia.visualisation.panels.intervention_reports import finite,truth,evidence_lines

def trace(rows,coordinate,anchor,max_gap):
    points = sorted([row for row in rows if row['kind'] == 'trace'], key=lambda row: row['sequence_index'])
    current = []
    segments = []
    previous = None
    missing = 0
    for row in points:
        valid = truth(row.get('raw_valid')) and truth(row.get('clock_valid')) and finite(row.get(coordinate)) and finite(row.get('raw_value'))
        adjacent = previous is not None and valid and truth(previous.get('raw_valid')) and truth(previous.get('clock_valid'))
        if adjacent:
            dt = row['hours'] - previous['hours']
            adjacent = 0 < dt <= max_gap and row['sequence_index'] == previous['sequence_index'] + 1
            if finite(row.get('frame_index')) and finite(previous.get('frame_index')):
                adjacent &= row['frame_index'] == previous['frame_index'] + 1
        if not adjacent and current:
            segments.append(current)
            current = []
        if valid:
            current.append(row)
        else:
            missing += 1
        previous = row
    if current:
        segments.append(current)
    windows=[]
    for row in rows:
        if row['kind']!='window':continue
        low,high=row.get('start_hours'),row.get('end_hours')
        if finite(low) and finite(high):
            shift=anchor['hours'] if coordinate=='relative_hours' else 0.
            windows.append(dict(low=low-shift,high=high-shift,baseline=row.get('baseline') is None))
    return dict(segments=[([r[coordinate] for r in segment],[r['raw_value'] for r in segment]) for segment in segments],windows=windows,counts={'original_points':len(points),'unplotted_invalid_points':missing,'segments':len(segments)})

def prepare(values,settings):
    rows=values.to_dict('records')
    if settings['view']=='cell_report':
        return dict(lines=evidence_lines(rows,settings),traces={clock:trace(rows,clock,settings['anchor'],settings['max_gap_hours']) for clock in ('hours','relative_hours')})
    cells={}
    for key in settings['cell_ids']:
        members=[row for row in rows if row['cell_id']==key]
        cells[key]=dict(label=members[0]['cell_label'],supported=sum(truth(row.get('response_supported')) for row in members if row['kind']=='effect'),changed=sum(truth(row.get('change_supported')) for row in members if row['kind']=='rhythm_change'),traces={clock:trace(members,clock,settings['anchors'][key],settings['max_gap_hours']) for clock in ('hours','relative_hours')})
    return dict(cells=cells)
