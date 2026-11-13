"""Evaluation-only checks of the name requested for a clicked biological body.

Keep relational cue scores unchanged. Two matching wrong names do not establish
the requested identity. Adjudicated names may override saved intermediate names;
this module and those declarations must never be consumed by a producer.
"""
from __future__ import annotations

from feedback_ground_truth_scoring import evaluate_at_feedback_cues


def evaluate_explicit_identity(mark, candidate, source, *, expected_identity=None,
                               cue_result=None):
    expected = mark.expected_identity if expected_identity is None else expected_identity
    result = dict(mark_id=mark.mark_id, expected_identity=expected,
                  actual_identity=None, status="not_declared")
    targets = {}
    if expected is not None:
        bodies = [i for i, anchor in enumerate(mark.anchors) if anchor.role in ("body", "point")]
        if len(bodies) == 1:
            targets[bodies[0]] = int(expected)
    elif getattr(mark, "kind", None) == "swap":
        # The reference position can be wrong; its saved identity is the name
        # explicitly requested for the paired body in the swap annotation.
        for pair in mark.must_match:
            if len(pair) != 2:
                continue
            bodies = [i for i in pair if mark.anchors[i].role == "body"]
            refs = [i for i in pair if mark.anchors[i].role == "reference"]
            if len(bodies) == len(refs) == 1:
                targets[bodies[0]] = int(mark.anchors[refs[0]].identity)
    else:
        return result
    if expected is not None and int(expected) <= 0:
        raise ValueError("An explicit cell identity must be positive")
    cue = cue_result if cue_result is not None else evaluate_at_feedback_cues(mark, candidate, source)
    names = cue.get("candidate_labels", [])
    if not targets or any(i >= len(names) for i in targets):
        result.update(status="undecidable", detail="clicked body could not be resolved uniquely")
        return result
    actual = [int(names[i]) for i in targets]
    requested = list(targets.values())
    result.update(actual_identity=actual[0] if len(actual) == 1 else actual,
                  expected_identity=requested[0] if len(requested) == 1 else requested,
                  status="satisfied" if actual == requested else "violated",
                  detail="explicit identity at clicked body; reference name is not authority")
    return result
