"""Keep demonstrated paired soma ownership across subsequently restored outlines."""
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from measured_restoration_continuity import geometry_by_frame
from measured_restoration_continuity_a002 import CoupledRestoredContinuity
from transport_restored_ownership_a002 import carry_restorations as carry_root_restorations


class MeasuredBodies:
    def __init__(self, frame, cores):
        self.frame, self.cores = frame, cores
        self.components, self.matches = {}, {}

    def numbered(self, name):
        if name not in self.components:
            self.components[name] = ndi.label(self.frame == name, np.ones((3, 3), bool))[0]
        return self.components[name]

    def locate(self, index):
        if index in self.matches:
            return self.matches[index]
        y, x, radius = self.cores[index]
        iy, ix = np.floor(np.array([y, x])+.5).astype(int)
        h, w = self.frame.shape
        key = None
        if 0 <= iy < h and 0 <= ix < w:
            name = int(self.frame[iy, ix])
            if name:
                key = (name, int(self.numbered(name)[iy, ix]))
            else:
                y0, y1 = max(0, int(np.floor(y-radius))), min(h, int(np.ceil(y+radius))+1)
                x0, x1 = max(0, int(np.floor(x-radius))), min(w, int(np.ceil(x+radius))+1)
                yy, xx = np.ogrid[y0:y1, x0:x1]
                disk = (yy-y)**2+(xx-x)**2 <= radius**2
                local = self.frame[y0:y1, x0:x1]
                choices = set()
                for name in np.unique(local[disk]):
                    if not name:
                        continue
                    parts = self.numbered(int(name))[y0:y1, x0:x1]
                    for number in np.unique(parts[disk & (local == name)]):
                        if number:
                            choices.add((int(name), int(number)))
                if len(choices) == 1:
                    key = next(iter(choices))
        self.matches[index] = key
        return key

    def unique_body(self, index):
        key = self.locate(index)
        if key is None:
            return None
        # A body's measurement is reciprocal too: two measured centres must not
        # acquire the same outline merely because their displayed names differ.
        if sum(self.locate(i) == key for i in range(len(self.cores))) != 1:
            return None
        return self.numbered(key[0]) == key[1]


def carry_restorations(parent, original, root_before, root_after, points, events):
    result, original_audit = carry_root_restorations(parent, original, root_before, root_after, points, events)
    geometry = geometry_by_frame(points)
    correspondence = CoupledRestoredContinuity(geometry, [])
    groups = {}
    for e in events:
        if e['decision'] == 'measured_exchange_applied':
            groups.setdefault(tuple(e['exchange_group']), []).append(e)
    requests = {}; audit = []
    for members in groups.values():
        # Repeated receipts cannot create extra votes for a biological identity.
        unique = {(int(e['owner']), tuple(e['target_core'])): e for e in members}
        if len(unique) != 2:
            continue
        members = list(unique.values())
        frame = int(members[0]['frame'])-1
        if any(int(e['frame'])-1 != frame for e in members):
            continue
        owners = [int(e['owner']) for e in members]
        if len(set(owners)) != 2:
            continue
        if any(int(parent[int(e['origin_frame'])-1,
                          *np.floor(np.asarray(e['origin_core'][:2])+.5).astype(int)]) != int(e['owner'])
               for e in members):
            continue
        cores = geometry.get(frame, np.empty((0, 3)))
        indices = [np.flatnonzero(np.all(cores == np.asarray(e['target_core']), axis=1)) for e in members]
        if any(len(i) != 1 for i in indices):
            continue
        indices = [int(i[0]) for i in indices]
        if indices[0] == indices[1]:
            continue
        for f in range(frame, len(parent)):
            if f > frame:
                links = correspondence.pair(f-1)[0]
                if any(i not in links for i in indices):
                    audit.append(dict(frame=f+1, decision='paired_measurement_correspondence_ended'))
                    break
                indices = [links[i] for i in indices]
            requests.setdefault(f, []).append((owners, list(indices), frame))

    for f, proposals in requests.items():
        bodies = MeasuredBodies(result[f], geometry.get(f, np.empty((0, 3))))
        transactions = []
        for owners, indices, origin in proposals:
            regions = [bodies.unique_body(i) for i in indices]
            record = dict(frame=f+1, origin_exchange_frame=origin+1,
                          owners=';'.join(map(str, owners)))
            if any(b is None for b in regions) or np.any(regions[0] & regions[1]):
                audit.append({**record, 'decision': 'paired_outlines_absent_shared_or_ambiguous'})
                continue
            union = regions[0] | regions[1]
            if any(np.any(b & (parent[f] != original[f])) and not np.all(result[f][b] == owner)
                   for b, owner in zip(regions, owners)):
                audit.append({**record, 'decision': 'paired_previous_accepted_change_preserved'})
                continue
            if any(np.any((result[f] == owner) & ~union) for owner in owners):
                audit.append({**record, 'decision': 'paired_unresolved_other_name_holder_preserved'})
                continue
            transactions.append((regions, owners, record))
        claims = np.zeros(result[f].shape, result.dtype)
        conflict = np.zeros(result[f].shape, bool)
        for regions, owners, _ in transactions:
            for b, owner in zip(regions, owners):
                conflict |= b & (claims > 0) & (claims != owner)
                claims[b] = owner
        # Remove complete transactions, including consistent overlap with a
        # rejected transaction, so a conflicting pair cannot leave a half move.
        while True:
            expanded = conflict.copy()
            for regions, _, _ in transactions:
                union = regions[0] | regions[1]
                if np.any(conflict & union):
                    expanded |= union
            if np.array_equal(expanded, conflict):
                break
            conflict = expanded
        for regions, owners, record in transactions:
            if np.any(conflict & (regions[0] | regions[1])):
                audit.append({**record, 'decision': 'paired_competing_certificates_held'})
                continue
            for b, owner in zip(regions, owners):
                count = int(np.count_nonzero(result[f][b] != owner))
                result[f][b] = owner
                audit.append({**record, 'owner': owner, 'pixels': int(b.sum()),
                              'changed_pixels': count, 'decision': 'paired_measured_owner_retained_in_restored_outline'})
    assert np.array_equal(result > 0, parent > 0)
    assert not np.any((result != parent) & (parent != original))
    return result, pd.concat([original_audit, pd.DataFrame(audit)], ignore_index=True)
