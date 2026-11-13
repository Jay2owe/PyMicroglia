"""Score saved feedback against the physical body that was clicked.

This is evaluation-only code. Producers must never import feedback or receive
mark identities, frames, coordinates, or regions.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
from scipy import ndimage

from feedback_marks import Mark, evaluate


def _nearest_source_pixel(plane: np.ndarray, identity: int,
                          y: int, x: int) -> tuple[int, int, float, int, np.ndarray]:
    mask = plane == identity
    if not mask.any():
        raise ValueError(f"identity {identity} is absent from its source frame")
    components, _ = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    distance, nearest = ndimage.distance_transform_edt(~mask, return_indices=True)
    ny, nx = map(int, nearest[:, y, x])
    return ny, nx, float(distance[y, x]), int(components[ny, nx]), components


def evaluate_at_feedback_cues(mark: Mark, candidate: np.ndarray,
                              source: np.ndarray, maximum_click_distance_px: float = 5.0) -> dict:
    """Use saved clicks for body relations; refuse ambiguous split locations.

    The source is the exact accepted label stack whose hash is stored in the
    mark. The original mark object and JSON-lines record remain untouched.
    """
    if candidate.shape != source.shape:
        raise ValueError("candidate and source label stacks have different shapes")
    resolved = deepcopy(mark)
    repaired_indices = []
    for index, anchor in enumerate(resolved.anchors):
        if anchor.identity <= 0 or anchor.click_y is None or anchor.click_x is None:
            continue
        ny, nx, distance, click_component, components = _nearest_source_pixel(
            source[anchor.frame_idx], anchor.identity, anchor.click_y, anchor.click_x)
        if distance > maximum_click_distance_px:
            return {"mark_id": mark.mark_id, "kind": mark.kind,
                    "status": "undecidable", "detail": "click is too far from its recorded source identity",
                    "candidate_labels": [], "sentence": mark.sentence,
                    "resolved_anchor_indices": repaired_indices}
        if int(components[anchor.y, anchor.x]) != click_component:
            repaired_indices.append(index)
        anchor.y, anchor.x = ny, nx

    if mark.kind == "split" and mark.line and mark.anchors:
        body = mark.anchors[0]
        (y0, x0), (y1, x1) = mark.line
        mid_y, mid_x = round((y0 + y1) / 2), round((x0 + x1) / 2)
        _, _, distance, cut_component, components = _nearest_source_pixel(
            source[body.frame_idx], body.identity, mid_y, mid_x)
        side_components = [int(components[a.y, a.x]) for a in mark.anchors[1:]
                           if a.role == "side"]
        clicked_component = int(components[resolved.anchors[0].y, resolved.anchors[0].x])
        if (distance > maximum_click_distance_px or clicked_component != cut_component
                or any(component != cut_component for component in side_components)):
            return {"mark_id": mark.mark_id, "kind": mark.kind,
                    "status": "undecidable",
                    "detail": "saved click, cut, or split-side points refer to different bodies",
                    "candidate_labels": [], "sentence": mark.sentence,
                    "resolved_anchor_indices": repaired_indices}
    result = evaluate(resolved, candidate)
    result["resolved_anchor_indices"] = repaired_indices
    return result
