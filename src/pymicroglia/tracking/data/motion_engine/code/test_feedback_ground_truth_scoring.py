"""Saved identity centers may differ from the bodies Jamie clicked."""
import numpy as np

from feedback_ground_truth_scoring import evaluate_at_feedback_cues
from feedback_marks import Anchor, Mark, evaluate


def test_clicked_body_controls_join_score_without_rewriting_mark():
    source = np.zeros((1, 25, 35), dtype=np.uint16)
    source[0, 3:8, 3:8] = 7
    source[0, 17:22, 17:22] = 7
    source[0, 17:22, 26:31] = 9
    candidate = source.copy()
    candidate[0, 17:22, 17:22] = 9
    mark = Mark("M1", "95_TEST", "join", "", "", 0,
                anchors=[Anchor(1, 1, 7, 5, 5, 0, "body", 19, 19),
                         Anchor(1, 1, 9, 19, 28, 0, "body", 19, 28)],
                must_match=[[0, 1]])

    assert evaluate(mark, candidate)["status"] == "violated"
    result = evaluate_at_feedback_cues(mark, candidate, source)
    assert result["status"] == "satisfied"
    assert result["resolved_anchor_indices"] == [0]
    assert (mark.anchors[0].y, mark.anchors[0].x) == (5, 5)


def test_split_with_click_on_other_body_is_undecidable():
    source = np.zeros((1, 30, 30), dtype=np.uint16)
    source[0, 3:9, 3:9] = 7
    source[0, 20:27, 20:27] = 7
    mark = Mark("M2", "95_TEST", "split", "", "", 0,
                anchors=[Anchor(1, 1, 7, 5, 5, 0, "body", 23, 23),
                         Anchor(1, 1, 7, 4, 4, 0, "side"),
                         Anchor(1, 1, 7, 7, 7, 0, "side")],
                must_differ=[[1, 2]], line=[[4, 5], [8, 5]])

    result = evaluate_at_feedback_cues(mark, source, source)
    assert result["status"] == "undecidable"
    assert "different bodies" in result["detail"]
