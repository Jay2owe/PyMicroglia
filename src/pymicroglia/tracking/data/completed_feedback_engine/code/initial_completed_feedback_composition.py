"""The original three accepted operations as a reusable array-to-array step.

No saved feedback, recording name, or numeric target is accepted by this function.
Parameters are supplied by the recorded accepted run, without recalibration.
"""
import density_reciprocal_overlap
import recurrent_cyclic_alias
import high_confidence_lineage_union

def _apply_original(labels, points, density_parameters):
    density,da,ds=density_reciprocal_overlap.correct(labels,density_parameters)
    cyclic,cp,mh,th,ca,cs=recurrent_cyclic_alias.correct(density,points)
    final,ua,um,ut,uapp,us=high_confidence_lineage_union.correct(cyclic,points)
    tables=dict(density_overlap_audit=da,cyclic_alias_proposals=cp,cyclic_mask_handoffs=mh,
        cyclic_track_handoffs=th,cyclic_alias_applications=ca,lineage_union_audit=ua,
        lineage_union_mask_handoffs=um,lineage_union_track_handoffs=ut,lineage_union_applications=uapp)
    return final,tables,dict(density=ds,cyclic=cs,lineage_union=us)


def apply(labels, points, density_parameters, *, exact_geometry_speedups=True):
    """Same accepted operations; optionally group per-cell pixel coordinates once."""
    if not exact_geometry_speedups:
        return _apply_original(labels, points, density_parameters)
    # Code provenance safeguard, not a scientific threshold or data selector.
    # A future change to either measured implementation uses its original path
    # until the grouped implementation has been checked against that revision.
    import hashlib
    from pathlib import Path
    sources = (
        (density_reciprocal_overlap, '5b2649db96a293aab84369b194cb80613fc94d69a8587e66b03f647627f6175f'),
        (recurrent_cyclic_alias, 'a77e023938de9a276d860eeac0f80066ca307f665fcca9059139eeeFB2257bcc'.lower()),
    )
    if any(hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != digest
           for module, digest in sources):
        return _apply_original(labels, points, density_parameters)
    from grouped_cell_geometry import grouped_geometry
    from contextlib import ExitStack
    import resident_takeover
    import owner_sampling_speed
    from resident_sampling_speed import batched_resident_owners
    sampling_sources = (
        (resident_takeover, '29e786ddd157429831438bc0d62ce6d43c1e95f1d0ab3458da040206551ba42f'),
        (owner_sampling_speed, '7bb2eadfb2fd7ad88a42e5c874e7b2cfd72b43435b00a300ac939a2869943146'),
    )
    with ExitStack() as stack:
        stack.enter_context(grouped_geometry(density_reciprocal_overlap, recurrent_cyclic_alias))
        from overlap_pair_encoding import encoded_overlap_pairs
        stack.enter_context(encoded_overlap_pairs(density_reciprocal_overlap, recurrent_cyclic_alias))
        if all(hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == digest
               for module, digest in sampling_sources):
            stack.enter_context(batched_resident_owners(resident_takeover, (high_confidence_lineage_union,)))
        return _apply_original(labels, points, density_parameters)
