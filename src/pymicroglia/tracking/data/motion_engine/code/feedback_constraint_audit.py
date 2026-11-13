"""Evaluation only: preserve individual feedback relationships, even in failed cues.

Generated persistence positions are inferred path controls, not independent
human clicks. Report their provenance; never pass these positions to producers.
"""
from feedback_ground_truth_scoring import evaluate_at_feedback_cues


def compare_constraints(mark, parent, candidate, source):
    before=evaluate_at_feedback_cues(mark,parent,source)
    after=evaluate_at_feedback_cues(mark,candidate,source)
    a=before.get('candidate_labels',[]);b=after.get('candidate_labels',[])
    if len(a)!=len(mark.anchors) or len(b)!=len(mark.anchors):
        return [dict(mark=mark.mark_id,kind=mark.kind,constraint='unresolved_positions',parent='undecidable',candidate='undecidable',regression=False,gain=False,provenance='saved feedback could not be resolved')]
    constraints=[('exists',(i,)) for i in mark.must_exist]
    constraints += [('match',tuple(pair)) for pair in mark.must_match]
    constraints += [('differ',tuple(pair)) for pair in mark.must_differ]
    if mark.kind in ('persist','ok'):
        bodies=[i for i,q in enumerate(mark.anchors) if q.role in ('body','point')]
        if len(bodies)==1:
            reference=bodies[0]
            constraints += [('exists',(i,)) for i in range(len(mark.anchors))]
            constraints += [('match',(reference,i)) for i in range(len(mark.anchors)) if i!=reference]
        else:
            return [dict(mark=mark.mark_id,kind=mark.kind,constraint='ambiguous_persistence_reference',parent='undecidable',candidate='undecidable',regression=False,gain=False,provenance='no unique human reference body')]
    def status(names,kind,indices):
        values=[int(names[i]) for i in indices]
        if kind=='exists':return 'satisfied' if values[0]>0 else 'violated'
        if kind=='differ' and any(i in mark.zero_ok for i in indices):
            return 'satisfied' if 0 in values or values[0]!=values[1] else 'violated'
        if 0 in values:return 'undecidable'
        yes=(values[0]==values[1]) if kind=='match' else (values[0]!=values[1])
        return 'satisfied' if yes else 'violated'
    rows=[]
    for kind,indices in sorted(set(constraints)):
        old=status(a,kind,indices);new=status(b,kind,indices)
        anchors=[mark.anchors[i] for i in indices]
        rows.append(dict(mark=mark.mark_id,kind=mark.kind,constraint=kind,indices=list(indices),
            frames=[q.review_frame for q in anchors],parent=old,candidate=new,
            parent_names=[a[i] for i in indices],candidate_names=[b[i] for i in indices],
            regression=old=='satisfied' and new!='satisfied',gain=old!='satisfied' and new=='satisfied',
            provenance='inferred persistence path from human body mark' if any(q.role=='control' for q in anchors) else 'saved human feedback relationship',
            aggregate_parent=before['status'],aggregate_candidate=after['status']))
    return rows
