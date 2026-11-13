from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi


CONNECTIVITY = np.ones((3, 3), bool)


def centroid(mask: np.ndarray) -> np.ndarray:
    points = np.column_stack(np.nonzero(mask))
    if not len(points):
        raise ValueError("cannot measure an empty mask")
    return points.mean(axis=0)


def largest_core_px(mask: np.ndarray) -> int:
    eroded = ndi.binary_erosion(mask, structure=CONNECTIVITY)
    components, count = ndi.label(eroded, structure=CONNECTIVITY)
    if not count:
        return 0
    return int(np.bincount(components.ravel())[1:].max())


def component_for_identity(components: np.ndarray, frame: np.ndarray,
                           identity: int) -> int:
    values, counts = np.unique(components[frame == identity], return_counts=True)
    choices = [(int(count), int(value))
               for value, count in zip(values, counts) if int(value) > 0]
    return max(choices)[1] if choices else 0


def derive_container(previous_labels: np.ndarray,
                     previous_foreground: np.ndarray,
                     donor: int) -> tuple[int, np.ndarray, dict]:
    """Find the labelled co-resident in the donor's predecessor blob.

    The takeover host is deliberately not considered. The container is a member
    of the same physical predecessor blob as the donor, chosen by immediate
    boundary contact and then shared-blob area.
    """
    components, count = ndi.label(previous_foreground, structure=CONNECTIVITY)
    donor_component = component_for_identity(components, previous_labels, donor)
    if not donor_component or donor_component > count:
        raise ValueError(f"identity {donor} has no predecessor foreground blob")
    branch = components == donor_component
    donor_mask = previous_labels == donor
    neighbours = sorted(set(map(int, np.unique(previous_labels[branch]))) - {0, donor})
    if not neighbours:
        raise ValueError(f"identity {donor} has no co-resident predecessor label")
    donor_edge = ndi.binary_dilation(donor_mask, structure=CONNECTIVITY) & branch
    rows = []
    for identity in neighbours:
        identity_mask = branch & (previous_labels == identity)
        rows.append({
            "identity": identity,
            "contact_px": int(np.count_nonzero(donor_edge & identity_mask)),
            "area_px": int(identity_mask.sum()),
        })
    best = max(rows, key=lambda row: (row["contact_px"], row["area_px"]))
    return int(best["identity"]), branch, {
        "branch_area_px": int(branch.sum()),
        "branch_members": "|".join(map(str, sorted(
            set(map(int, np.unique(previous_labels[branch]))) - {0}))),
        "container_contact_px": int(best["contact_px"]),
    }


def choose_carrier(current_components: np.ndarray, component_count: int,
                   current_labels: np.ndarray, container: int,
                   previous_branch: np.ndarray, dilation_px: int) -> int:
    visible = component_for_identity(current_components, current_labels, container)
    if visible:
        return visible
    expanded = ndi.binary_dilation(
        previous_branch, structure=CONNECTIVITY, iterations=dilation_px)
    previous_centre = centroid(previous_branch)
    scored = []
    for number in range(1, component_count + 1):
        mask = current_components == number
        direct = int(np.count_nonzero(previous_branch & mask))
        near = int(np.count_nonzero(expanded & mask))
        distance = float(np.linalg.norm(centroid(mask) - previous_centre))
        scored.append((direct + 0.25 * near - 0.01 * distance, number))
    if not scored:
        raise ValueError("current frame has no foreground component")
    return int(max(scored)[1])


def project_branch(domain: np.ndarray, container_seed: np.ndarray,
                   previous_branch: np.ndarray, target_area: int,
                   dilation_px: int) -> np.ndarray:
    """Keep an internal branch mask while several blobs share one component."""
    seed = container_seed & domain
    if not np.any(seed):
        seed = (ndi.binary_dilation(
            previous_branch, structure=CONNECTIVITY,
            iterations=dilation_px) & domain)
    if not np.any(seed):
        pixels = np.column_stack(np.nonzero(domain))
        point = centroid(previous_branch)
        nearest = pixels[int(np.argmin(np.sum((pixels - point) ** 2, axis=1)))]
        seed = np.zeros(domain.shape, bool)
        seed[tuple(nearest)] = True
    distance = ndi.distance_transform_edt(~seed)
    domain_pixels = np.flatnonzero(domain)
    keep = min(int(target_area), len(domain_pixels))
    result = np.zeros(domain.shape, bool)
    if keep:
        costs = distance.ravel()[domain_pixels]
        selected = domain_pixels[np.argpartition(costs, keep - 1)[:keep]]
        result.ravel()[selected] = True
    return result


def release_candidates(current_components: np.ndarray, component_count: int,
                       carrier: int, current_labels: np.ndarray,
                       previous_branch: np.ndarray, donor: int,
                       expected_area: int, expected_core: int,
                       params: dict) -> list[dict]:
    expanded = ndi.binary_dilation(
        previous_branch, structure=CONNECTIVITY,
        iterations=int(params["branch_projection_dilation_px"]))
    rows = []
    for number in range(1, component_count + 1):
        if number == carrier:
            continue
        mask = current_components == number
        area = int(mask.sum())
        donor_px = int(np.count_nonzero(mask & (current_labels == donor)))
        if not donor_px:
            continue
        direct = int(np.count_nonzero(previous_branch & mask))
        contact = int(np.count_nonzero(expanded & mask))
        core = largest_core_px(mask)
        area_fraction = area / max(expected_area, 1)
        core_fraction = core / max(expected_core, 1)
        donor_fraction = donor_px / max(area, 1)
        passed = bool(
            contact >= int(params["minimum_branch_contact_px"])
            and float(params["minimum_release_area_fraction"]) <= area_fraction
            <= float(params["maximum_release_area_fraction"])
            and core >= int(params["minimum_release_core_px"])
            and core_fraction >= float(params["minimum_release_core_fraction"])
            and donor_fraction >= float(params["minimum_release_donor_fraction"]))
        score = (donor_fraction + min(core_fraction, 1.0)
                 - abs(np.log2(max(area_fraction, 1e-6)))
                 + 0.01 * min(contact, 25))
        rows.append({
            "component": number,
            "area_px": area,
            "core_px": core,
            "direct_branch_overlap_px": direct,
            "branch_contact_px": contact,
            "donor_px": donor_px,
            "donor_fraction": donor_fraction,
            "area_fraction": area_fraction,
            "core_fraction": core_fraction,
            "score": float(score),
            "passed": passed,
        })
    return sorted(rows, key=lambda row: row["score"], reverse=True)


def suppress_dormant_identity(frame: np.ndarray, unclaimed: np.ndarray,
                              donor: int, container: int,
                              stored_branch: np.ndarray,
                              allowed_mask: np.ndarray | None = None
                              ) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Remove dormant-name appearances while conserving foreground support."""
    del container, stored_branch
    result = frame.copy()
    result_unclaimed = unclaimed.copy()
    forbidden = result == donor
    if allowed_mask is not None:
        forbidden &= ~allowed_mask
    actions = []
    if np.any(forbidden):
        result[forbidden] = 0
        result_unclaimed[forbidden] = donor
        actions.append({
            "action": "dormant_pixels_made_unclaimed_until_branch_split",
            "foreground_component": 0,
            "changed_pixels": int(forbidden.sum()),
        })
    return result, result_unclaimed, actions


def merge_residency_registry(parent: pd.DataFrame,
                             ledger: pd.DataFrame) -> pd.DataFrame:
    """Carry upstream residents forward and add newly dormant branch residents."""
    columns = ["source_imagej_frame", "identity", "host_identity", "held_px"]
    additions = []
    if not ledger.empty:
        retained = ledger[ledger.decision.astype(str).isin({
            "retained_inside_stored_branch",
            "retained_inside_stored_branch_at_movie_end",
        })]
        for row in retained.itertuples(index=False):
            additions.append({
                "source_imagej_frame": int(row.source_frame),
                "identity": int(row.resident_identity),
                "host_identity": int(row.container_identity),
                "held_px": int(row.projected_branch_area_px),
            })
    combined = pd.concat(
        [parent.reindex(columns=columns), pd.DataFrame(additions, columns=columns)],
        ignore_index=True)
    if combined.empty:
        return pd.DataFrame(columns=columns)
    for column in columns:
        combined[column] = combined[column].astype(int)
    return (combined.drop_duplicates(
        ["source_imagej_frame", "identity"], keep="last")
        .sort_values(["source_imagej_frame", "identity"])
        .reset_index(drop=True))


def process_event(result: np.ndarray, result_unclaimed: np.ndarray,
                  baseline: np.ndarray, foreground: np.ndarray,
                  event: dict, offset: int, params: dict
                  ) -> tuple[list[dict], list[dict]]:
    donor = int(event["donor_identity"])
    takeover_host = int(event["host_identity"])
    start = int(event["source_frame"])
    start_index = start - offset - 1
    derived_container, previous_branch, branch_info = derive_container(
        baseline[start_index - 1], foreground[start_index - 1], donor)
    configured_container = int(event.get("container_identity", derived_container))
    if configured_container != derived_container:
        raise AssertionError(
            f"{event['case_id']}: configured container {configured_container} "
            f"does not match predecessor blob {derived_container}")
    expected_area = int(np.count_nonzero(baseline[start_index - 1] == donor))
    expected_core = largest_core_px(baseline[start_index - 1] == donor)
    target_branch_area = int(previous_branch.sum())
    activation_value = event.get("residency_activation_source_frame", start)
    activation_source = (None if activation_value is None
                         else int(activation_value))
    if activation_source is not None and activation_source < start:
        raise ValueError(
            f"{event['case_id']}: residency activation precedes branch origin")
    ledger_rows: list[dict] = []
    action_rows: list[dict] = []
    released = False
    residency_active = False

    for index in range(start_index, len(baseline)):
        source = index + offset + 1
        components, count = ndi.label(
            foreground[index], structure=CONNECTIVITY)
        carrier = choose_carrier(
            components, count, baseline[index], configured_container,
            previous_branch, int(params["branch_projection_dilation_px"]))
        carrier_mask = components == carrier
        current_branch = project_branch(
            carrier_mask,
            (baseline[index] == configured_container) & carrier_mask,
            previous_branch, target_branch_area,
            int(params["branch_projection_dilation_px"]))

        # REGRESSION GUARD: a newer upstream base can move or remove the
        # reviewed takeover. Remember the branch from contact, but activate
        # suppression only at that base's separately measured takeover frame.
        # Global visibility is not the trigger: the old base could show the
        # same name on a different body after its reviewed body was absorbed.
        visible_px = int(np.count_nonzero(baseline[index] == donor))
        if (activation_source is None
                or (not residency_active and source < activation_source)):
            if not visible_px:
                raise AssertionError(
                    f"{event['case_id']}: identity {donor} disappeared at "
                    f"source {source} before configured activation "
                    f"{activation_source}")
            ledger_rows.append({
                "case_id": event["case_id"],
                "source_frame": source,
                "resident_identity": donor,
                "takeover_host_identity": takeover_host,
                "container_identity": configured_container,
                "predecessor_branch_members": branch_info["branch_members"],
                "carrier_component": carrier,
                "carrier_area_px": int(carrier_mask.sum()),
                "projected_branch_area_px": target_branch_area,
                "baseline_resident_px": visible_px,
                "release_candidate_count": 0,
                "best_candidate_component": 0,
                "best_candidate_area_px": 0,
                "best_candidate_core_px": 0,
                "best_candidate_branch_contact_px": 0,
                "decision": "visible_before_residency_activation",
            })
            previous_branch = current_branch
            continue

        residency_active = True
        candidates = release_candidates(
            components, count, carrier, baseline[index], previous_branch,
            donor, expected_area, expected_core, params)
        accepted = next((row for row in candidates if row["passed"]), None)
        allowed = components == int(accepted["component"]) if accepted else None
        result[index], result_unclaimed[index], suppression = (
            suppress_dormant_identity(
                result[index], result_unclaimed[index], donor,
                configured_container, current_branch, allowed))
        for row in suppression:
            action_rows.append({
                "case_id": event["case_id"],
                "source_frame": source,
                "resident_identity": donor,
                "container_identity": configured_container,
                **row,
            })
        ledger_rows.append({
            "case_id": event["case_id"],
            "source_frame": source,
            "resident_identity": donor,
            "takeover_host_identity": takeover_host,
            "container_identity": configured_container,
            "predecessor_branch_members": branch_info["branch_members"],
            "carrier_component": carrier,
            "carrier_area_px": int(carrier_mask.sum()),
            "projected_branch_area_px": target_branch_area,
            "baseline_resident_px": visible_px,
            "release_candidate_count": len(candidates),
            "best_candidate_component": (
                int(candidates[0]["component"]) if candidates else 0),
            "best_candidate_area_px": (
                int(candidates[0]["area_px"]) if candidates else 0),
            "best_candidate_core_px": (
                int(candidates[0]["core_px"]) if candidates else 0),
            "best_candidate_branch_contact_px": (
                int(candidates[0]["branch_contact_px"]) if candidates else 0),
            "decision": ("released_from_stored_branch_split" if accepted
                         else "retained_inside_stored_branch"),
        })
        if accepted:
            released = True
            break
        previous_branch = current_branch

    if not released:
        ledger_rows[-1]["decision"] = (
            "retained_inside_stored_branch_at_movie_end"
            if residency_active
            else "upstream_kept_identity_visible_no_residency_needed")
    return ledger_rows, action_rows

