"""Validate the physical soma evidence behind whole spatial-overlap paths.

Repeated detector or recovery masks do not become independent soma evidence.
A single independently supported body anywhere protects its entire connected
path. Biological names are carried through but never determine connectivity.
"""
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from recovery_temporal_soma_support import temporally_supported_points


def tracker_support_frames(raw, points, *, internal_peak=False, peak_quantisation=False, ownership_labels=None, require_foreground_majority=False, retain_named_centre=False, prefer_centred_owner=False):
    """Reuse accepted centred raw coverage and whole-disk dim-path protection."""
    selected = temporally_supported_points(raw, points, centre_connected=True, internal_peak=internal_peak, peak_quantisation=peak_quantisation)
    selected = selected[selected.strong.astype(bool) & selected.state.eq('observed')]
    if 'physically_visible' in selected:
        selected = selected[selected.physically_visible.astype(bool)]
    elif 'visibility' in selected:
        selected = selected[selected.visibility.gt(0)]
    result = {}
    ownership_cache = {}
    cross = ndi.generate_binary_structure(2, 1)
    for q in selected.itertuples(index=False):
        t = int(q.frame); x = float(q.x); y = float(q.y); r = float(q.radius_px)
        if not 0 <= t < len(raw) or not np.isfinite([x, y, r]).all() or r < 0:
            continue
        y0 = max(0, int(np.floor(y-r))); y1 = min(raw.shape[1], int(np.ceil(y+r))+1)
        x0 = max(0, int(np.floor(x-r))); x1 = min(raw.shape[2], int(np.ceil(x+r))+1)
        if y0 >= y1 or x0 >= x1:
            continue
        yy, xx = np.ogrid[y0:y1, x0:x1]
        disk = (yy-y)**2 + (xx-x)**2 <= r*r
        cy, cx = int(np.floor(y+.5))-y0, int(np.floor(x+.5))-x0
        if 0 <= cy < len(disk) and 0 <= cx < disk.shape[1]:
            disk[cy, cx] = True
        support = disk
        resolved_core = False
        interior = ndi.binary_erosion(disk, cross)
        if interior.any():
            patch = raw[t, y0:y1, x0:x1]
            bright = disk & (patch > np.median(patch[disk & ~interior]))
            components, _ = ndi.label(bright, np.ones((3, 3), bool))
            seed = int(components[cy, cx]) if 0 <= cy < len(disk) and 0 <= cx < disk.shape[1] else 0
            if seed and ndi.binary_erosion(components == seed, cross).any():
                support = components == seed
                resolved_core = True
        if resolved_core and ownership_labels is not None:
            from tracker_core_ownership import owned_core_support, component_owners
            if t not in ownership_cache:
                ownership_cache[t] = component_owners(ownership_labels[t])
            support = owned_core_support(support, ownership_cache[t][y0:y1, x0:x1], require_foreground_majority=require_foreground_majority, centre=(cy,cx) if retain_named_centre else None, prefer_centred_owner=prefer_centred_owner)
        result.setdefault(t, np.zeros(raw.shape[1:], bool))[y0:y1, x0:x1] |= support
    return result


def constrain_spatial_soma_paths(observed, proposals, raw, points, params=None, *, proposal_history=None, resolve_thin_enclosure=False, coherent_measured_core=False, enclose_measured_core=False, require_tracker_peak=False, tracker_peak_measured_only=False, tracker_peak_quantisation=False, reciprocal_overlap=False, require_shared_tracker_core=False, exclusive_tracker_core=False, require_soma_profile=False, qualify_inferred_paths=False, tracker_foreground_majority=False, retain_named_centre=False, inferred_only=False, prefer_centred_owner=False, repeated_thin_measured_only=False, repeated_template_gap_copies=False, repeated_measured_templates_only=False):
    """Withdraw unsupported complete paths; caller preserves approved revisions."""
    if observed.shape != proposals.shape or observed.shape != raw.shape or raw.ndim != 3:
        raise ValueError('Expected aligned complete movies')
    if np.any((observed > 0) & (observed != proposals)):
        raise ValueError('Expected fixed additive proposals')
    # Empty registration padding carries no acquisition evidence. Evaluate
    # point footprints in the same observed support extent after translations.
    support_extent = np.any(raw != 0, axis=0)
    ey, ex = np.nonzero(support_extent)
    if not len(ey):
        y0, y1, x0, x1 = 0, raw.shape[1], 0, raw.shape[2]
    else:
        y0, y1, x0, x1 = int(ey.min()), int(ey.max())+1, int(ex.min()), int(ex.max())+1
    local_points = points.copy()
    local_points['y'] = local_points['y']-y0
    local_points['x'] = local_points['x']-x0
    ownership = proposals[:,y0:y1,x0:x1] if exclusive_tracker_core else None
    local_footprints = tracker_support_frames(raw[:,y0:y1,x0:x1], local_points, internal_peak=require_tracker_peak, peak_quantisation=tracker_peak_quantisation, ownership_labels=ownership, require_foreground_majority=tracker_foreground_majority, retain_named_centre=retain_named_centre, prefer_centred_owner=prefer_centred_owner)
    footprint = {t: np.pad(mask, ((y0,raw.shape[1]-y1),(x0,raw.shape[2]-x1))) for t,mask in local_footprints.items()}
    legacy_footprint = {}
    if require_tracker_peak and tracker_peak_measured_only:
        legacy_local = tracker_support_frames(raw[:,y0:y1,x0:x1], local_points)
        legacy_footprint = {t: np.pad(mask, ((y0,raw.shape[1]-y1),(x0,raw.shape[2]-x1))) for t,mask in legacy_local.items()}
    clipped = set()
    cross = ndi.generate_binary_structure(2, 1)
    eight = np.ones((3, 3), bool)
    nodes = np.zeros(proposals.shape, np.int32)
    records = [None]; parents = [0]; touching = set()
    for t, frame in enumerate(proposals):
        for identity, roi in enumerate(ndi.find_objects(frame), start=1):
            if roi is None:
                continue
            components, count = ndi.label(frame[roi] == identity, eight)
            for part in range(1, count+1):
                body = components == part
                interior = ndi.binary_erosion(body, cross)
                measured = body & (observed[t][roi] == identity)
                measured_interior = ndi.binary_erosion(measured, cross)
                coherent = False
                measured_enclosure_core = False
                if measured_interior.any():
                    patch = raw[t][roi]
                    level = np.median(patch[measured & ~measured_interior])
                    if coherent_measured_core:
                        coherent = bool(ndi.binary_erosion(measured & (patch > level), cross).any())
                    else:
                        coherent = bool(np.any(measured_interior & (patch > level)))
                    if coherent_measured_core and enclose_measured_core and not coherent:
                        from skimage.morphology import convex_hull_image
                        enclosure = convex_hull_image(measured)
                        enclosed_inner = ndi.binary_erosion(enclosure, cross)
                        level = np.median(patch[enclosure & ~enclosed_inner])
                        measured_enclosure_core = bool(ndi.binary_erosion(enclosure & (patch > level), cross).any())
                        coherent = measured_enclosure_core
                ambiguous = bool(measured.any() and not measured_interior.any())
                enclosure_resolved = False; enclosure_core = False
                enclosure_area = 0; enclosure_inner_pixels = 0
                if resolve_thin_enclosure and ambiguous:
                    # Two axial erosions need at least five rows and columns.
                    # This exact geometric early exit avoids needless hull work.
                    if min(measured.shape) >= 5:
                        from skimage.morphology import convex_hull_image
                        enclosure = convex_hull_image(measured)
                        enclosed_inner = ndi.binary_erosion(enclosure, cross)
                        enclosure_resolved = bool(ndi.binary_erosion(enclosed_inner, cross).any())
                        enclosure_area = int(enclosure.sum())
                        enclosure_inner_pixels = int(enclosed_inner.sum())
                        if enclosure_resolved:
                            patch = raw[t][roi]
                            level = np.median(patch[enclosure & ~enclosed_inner])
                            enclosure_core = bool(ndi.binary_erosion(enclosure & (patch > level), cross).any())
                            ambiguous = False
                            coherent = enclosure_core
                if qualify_inferred_paths and not measured.any():
                    from inferred_raw_soma_evidence import has_raw_soma
                    coherent = has_raw_soma(body, raw[t][roi])
                witness = bool(t in footprint and np.any(body & footprint[t][roi]))
                profile_supported = True
                if require_soma_profile and (coherent or witness) and not ambiguous:
                    from soma_profile_evidence import soma_profile_supported
                    profile_supported = soma_profile_supported(body, raw[t][roi])
                    coherent = coherent and profile_supported
                    witness = witness and profile_supported
                edge_only_tracker = False
                if witness and require_shared_tracker_core:
                    witness = bool(ndi.binary_erosion(body & footprint[t][roi], cross).any())
                    edge_only_tracker = not witness
                node = len(records); parents.append(node)
                nodes[t][roi][body] = node
                yy, xx = np.nonzero(body)
                gy, gx = yy+roi[0].start, xx+roi[1].start
                if np.any((gy <= y0) | (gy >= y1-1) | (gx <= x0) | (gx >= x1-1)):
                    clipped.add(node)
                records.append(dict(node=node, identity=identity, frame=t+1,
                    seed_y=int(yy[0])+roi[0].start, seed_x=int(xx[0])+roi[1].start,
                    proposed_pixels=int(body.sum()), measured_pixels=int(measured.sum()),
                    resolved=bool(interior.any()) or enclosure_resolved, raw_soma=coherent, tracker_soma=witness, measured_ambiguous=ambiguous))
                if require_shared_tracker_core:
                    records[-1]['edge_only_tracker_contact'] = edge_only_tracker
                if require_soma_profile:
                    records[-1]['soma_profile_supported'] = profile_supported
                if resolve_thin_enclosure:
                    records[-1].update(enclosure_resolved=enclosure_resolved, enclosure_raw_soma=enclosure_core, enclosure_area=enclosure_area, enclosure_inner_pixels=enclosure_inner_pixels)
                if enclose_measured_core:
                    records[-1]['measured_enclosure_core'] = measured_enclosure_core
                if tracker_peak_measured_only:
                    records[-1]['legacy_tracker_soma'] = bool(t in legacy_footprint and np.any(body & legacy_footprint[t][roi]))
        # A touching outline can be another part of a real cell. Preserve that
        # ownership ambiguity rather than interpreting it as absent foreground.
        field = nodes[t]; padded = np.pad(field, 1)
        for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):
            neighbour = padded[1+dy:1+dy+field.shape[0], 1+dx:1+dx+field.shape[1]]
            hit = (field > 0) & (neighbour > 0) & (field != neighbour)
            touching.update(map(int, np.unique(field[hit])))
            touching.update(map(int, np.unique(neighbour[hit])))

    def find(a):
        while parents[a] != a:
            parents[a] = parents[parents[a]]
            a = parents[a]
        return a

    if reciprocal_overlap:
        from reciprocal_soma_overlap import reciprocal_overlap_pairs
        areas=np.array([0]+[q['proposed_pixels'] for q in records[1:]])
    for t in range(1, len(nodes)):
        shared = (nodes[t-1] > 0) & (nodes[t] > 0)
        if reciprocal_overlap:
            pairs, counts = np.unique(np.column_stack((nodes[t-1][shared], nodes[t][shared])), axis=0, return_counts=True)
            pairs = reciprocal_overlap_pairs(pairs, counts, areas)
        else:
            pairs = np.unique(np.column_stack((nodes[t-1][shared], nodes[t][shared])), axis=0)
        for a, b in pairs:
            ra, rb = find(int(a)), find(int(b))
            if ra != rb:
                parents[rb] = ra
    groups = {}
    for node in range(1, len(records)):
        groups.setdefault(find(node), []).append(node)
    rejected = []; audit = []
    for path, members in groups.items():
        rec = [records[node] for node in members]
        if inferred_only and any(q['measured_pixels'] for q in rec):
            continue
        if any(q['raw_soma'] or q['tracker_soma'] or q['measured_ambiguous'] for q in rec):
            continue
        # Peak qualification refines paths anchored by an actual detection.
        # Pure gap recovery retains its preceding independent tracker evidence;
        # a stricter local contrast test alone cannot overturn that layer.
        if tracker_peak_measured_only and not any(q['measured_pixels'] for q in rec) and any(q['legacy_tracker_soma'] for q in rec):
            continue
        if (not any(q['resolved'] for q in rec) and (not qualify_inferred_paths or any(q['measured_pixels'] for q in rec))) or any(node in touching or node in clipped for node in members):
            continue
        if repeated_thin_measured_only:
            # Repeated outlines are not independent observations of a soma.
            # Compare complete measured geometry up to translation. A native
            # resolved interior, any shape change, gap-only observation, or
            # a single observation keeps its existing protection.
            if len(rec) < 2 or len({q['frame'] for q in rec}) != len(rec):
                continue
            if repeated_template_gap_copies:
                # A copied gap outline is not an additional measurement. Its
                # geometry must equal a fully measured thin template exactly;
                # partially measured shapes keep their existing protection.
                measured_rec = [q for q in rec if q['measured_pixels']]
                if (not measured_rec or (repeated_measured_templates_only and len(measured_rec)<2)) or any(q['measured_pixels'] != q['proposed_pixels'] or not q.get('enclosure_resolved', False) for q in measured_rec):
                    continue
            elif any(q['measured_pixels'] != q['proposed_pixels'] or not q.get('enclosure_resolved', False) for q in rec):
                continue
            template = None
            repeated = True
            for q in rec:
                if repeated_measured_templates_only and not q['measured_pixels']:
                    continue
                yy, xx = np.nonzero(nodes[q['frame']-1] == q['node'])
                geometry = np.column_stack((yy-yy.min(), xx-xx.min()))
                if template is None:
                    template = geometry
                elif not np.array_equal(template, geometry):
                    repeated = False
                    break
            if not repeated:
                continue
        rejected.extend(members)
        first = min(q['frame'] for q in rec); last = max(q['frame'] for q in rec)
        for q in rec:
            audit.append(dict(q, path=path, path_first_frame=first, path_last_frame=last,
                path_node_count=len(members), path_resolved_nodes=sum(x['resolved'] for x in rec),
                path_measured_nodes=sum(x['measured_pixels'] > 0 for x in rec),
                touches_another_outline=False, withdrawn_pixels=q['proposed_pixels'],
                decision='spatial_path_without_independent_soma_evidence'))
    output = proposals.copy()
    lookup = np.zeros(len(records), bool); lookup[rejected] = True
    output[lookup[nodes]] = 0
    columns = ['node','identity','frame','seed_y','seed_x','proposed_pixels','measured_pixels',
               'resolved','raw_soma','tracker_soma','measured_ambiguous','path','path_first_frame','path_last_frame',
               'path_node_count','path_resolved_nodes','path_measured_nodes',
               'touches_another_outline','withdrawn_pixels','decision']
    if resolve_thin_enclosure:
        columns += ['enclosure_resolved','enclosure_raw_soma','enclosure_area','enclosure_inner_pixels']
    if enclose_measured_core:
        columns += ['measured_enclosure_core']
    if tracker_peak_measured_only:
        columns += ['legacy_tracker_soma']
    if require_shared_tracker_core:
        columns += ['edge_only_tracker_contact']
    if require_soma_profile:
        columns += ['soma_profile_supported']
    return output, pd.DataFrame(audit, columns=columns)

