"""All states of one assignment against its observations in one array pass.

Batch.build measured one state against every observation, once per state, and
materialised a fifteen-value tuple for every pair although most pairs are later
rejected by distance alone. Here every per-state scalar is computed by the same
Python expression as Batch.build; only the element-wise per-observation
arithmetic runs on (state, observation) arrays, whose elements are the same
IEEE operations. Pair tuples are made only when asked for, and prepare runs on
the state's array rows with the frozen prepare statements.

Without retained-source constraints, a state's eligible observations are its
row of the whole-batch empty mask. The nonzero entries of every row are taken
once per whole-batch result (row-major, so each state's in ascending order)
and their values converted to Python lists once; each state's prepare is then
a slice of those lists, the same values in the same order as its own
flatnonzero, fancy index and tolist.
"""
from collections.abc import Sequence

import numpy as np


def _norm2(x, axis):
    """np.linalg.norm(x, axis=axis) for float64 coordinate pairs on the last axis.

    For a real float64 array norm squares the whole array in one multiply and
    adds along the axis; with two elements that is the one sum s0 + s1 (an
    IEEE sum of two terms has one result), then the square root. A sum that
    overflows is taken by norm's own reduction of the same squares, so it is
    reported as norm reports it; other inputs keep norm.
    """
    if type(x) is np.ndarray and x.dtype == np.float64 and x.ndim >= 1 and axis == x.ndim - 1 and x.shape[-1] == 2:
        squares = x * x
        try:
            with np.errstate(over='raise'):
                total = squares[..., 0] + squares[..., 1]
        except FloatingPointError:
            return np.sqrt(np.add.reduce(squares, axis=axis))
        return np.sqrt(total)
    return np.linalg.norm(x, axis=axis)


class Rows(Sequence):
    """Batch.rows for one state, built on access with Batch.build's types."""

    def __init__(self, matrix, s):
        self.matrix = matrix
        self.s = s
        self.columns = None

    def __len__(self):
        return self.matrix.n

    def __getitem__(self, i):
        m, s = self.matrix, self.s
        if not isinstance(i, (int, np.integer)):
            return [self[j] for j in range(m.n)[i]]
        if i < 0:
            i += m.n
        if not 0 <= i < m.n:
            raise IndexError(i)
        if self.columns is None:
            # This state's row as Python values, converted once from the same elements.
            if m.preferred_values is None:
                m.preferred_values = [int(v) for v in m.preferred]
            self.columns = (m.preferred_values, m.distance[s].tolist(), m.displacement[s].tolist(),
                            m.area_ratio[s].tolist(), m.area_change[s].tolist(), m.mean_change[s].tolist(),
                            m.lineage_area[s].tolist(), m.lineage_mean[s].tolist(),
                            m.lineage_shape[s].tolist(), m.reversal[s].tolist())
        (preferred, distance, displacement, area_ratio, area_change, mean_change,
         lineage_area, lineage_mean, lineage_shape, reversal) = self.columns
        i = int(i)
        return (m.steps[s], m.predicted[s, i], preferred[i], m.reserved,
                distance[i], displacement[i], m.maximum_distance[s],
                m.maximum_recovery, area_ratio[i], area_change[i],
                mean_change[i], lineage_area[i], lineage_mean[i],
                lineage_shape[i], reversal[i])


class LazyPlain:
    """prepare's eligible-pair dictionaries, each formed when looked up.

    The columns are Python floats, booleans and integers converted from the
    same array elements the dictionary values were converted from.
    """
    __slots__ = ('position', 'steps', 'columns')

    def __init__(self, eligible, steps, columns, start=0):
        # With start, the columns are every state's lists and eligible's
        # entries sit at start, start+1, ... in them, so no column is sliced.
        self.position = dict(zip(eligible, range(start, start + len(eligible))))
        self.steps = steps
        self.columns = columns

    def __len__(self):
        return len(self.position)

    def __contains__(self, index):
        return index in self.position

    def __iter__(self):
        return iter(self.position)

    def keys(self):
        return self.position.keys()

    def __getitem__(self, index):
        (allowed, cost, distance, displacement, predicted_y, predicted_x, area_ratio, mean_change,
         lineage_area, lineage_mean, lineage_shape, reversal, shared) = self.columns
        k = self.position[index]
        return {'allowed': allowed[k], 'cost': cost[k], 'distance_px': distance[k],
                'observed_displacement_px': displacement[k],
                'predicted_y': predicted_y[k], 'predicted_x': predicted_x[k],
                'area_ratio': area_ratio[k], 'mean_log2_change': mean_change[k],
                'lineage_area_log2_change': lineage_area[k], 'lineage_mean_log2_change': lineage_mean[k],
                'lineage_shape_log2_change': lineage_shape[k], 'direction_reversal_penalty': reversal[k],
                'raw_direction_reversal_penalty': reversal[k], 'shared_green_tracks': shared[k],
                'fast_motion_handoff': False, 'reversal_aware_handoff': False,
                'motion_prediction_model': 'constant_velocity', 'gap_steps': self.steps,
                'motion_support': 0.0, 'supportive_px': 0, 'contradictory_px': 0, 'union_px': 0,
                'source_loss_coverage': 0.0, 'destination_gain_coverage': 0.0}

    def get(self, index, default=None):
        return self[index] if index in self.position else default

    def values(self):
        return [self[i] for i in self.position]

    def items(self):
        return [(i, self[i]) for i in self.position]


class Matrix:
    def __init__(self, states, batch):
        p = batch.params
        n = len(batch.observations)
        positions, areas, means, shapes = batch.positions, batch.areas, batch.means, batch.shapes
        if not (positions.dtype == np.float64 and areas.dtype == np.float64
                and means.dtype == np.float64 and shapes.dtype == np.float64):
            raise TypeError('observation arrays are not float64')
        reserved = p.get('_motion_reservation_identity_set')
        if reserved is None:
            reserved = {int(v) for v in p.get('motion_reservation_identities', [])}
        maximum_recovery = float(p.get('maximum_recovery_distance_px', np.inf))
        minimum = float(p.get('minimum_direction_speed_px', 2.0))
        steps, span, base, origin, velocity_s = [], [], [], [], []
        area_d, mean_d, ref_area_d, ref_mean_d, ref_shape, old_speed, maximum_distance = [], [], [], [], [], [], []
        for state in states:
            step = abs(int(batch.target) - int(state.t))
            point = state.position + state.velocity * max(step, 1)
            position = np.asarray(state.position)
            velocity = np.asarray(state.velocity)
            if (np.asarray(point).dtype != np.float64 or np.shape(point) != (2,)
                    or position.dtype.kind not in 'fiu' or position.shape != (2,)
                    or velocity.dtype.kind not in 'fiu' or velocity.shape != (2,)):
                raise TypeError('state vectors differ from the float64 pair layout')
            steps.append(step)
            span.append(max(step, 1))
            base.append(point)
            origin.append(position)
            velocity_s.append(velocity)
            maximum_distance.append(float(p['max_link_px']) * np.sqrt(max(step, 1)))
            area_d.append(max(state.area, 1))
            mean_d.append(state.mean + 1.0)
            ref_area = float(state.reference_area) if np.isfinite(state.reference_area) else float(state.area)
            ref_mean = float(state.reference_mean) if np.isfinite(state.reference_mean) else float(state.mean)
            shape = state.reference_shape if state.reference_shape is not None else batch.module._shape_descriptor(state.mask)
            ref_area_d.append(max(ref_area, 1.0))
            ref_mean_d.append(ref_mean + 1.0)
            ref_shape.append(np.asarray(shape, float) + 1e-3)
            old_speed.append(float(np.linalg.norm(state.velocity)))
        ref_shape = np.asarray(ref_shape)
        if ref_shape.ndim != 2 or ref_shape.shape[1] != shapes.shape[1]:
            raise TypeError('shape descriptors differ in length')

        def column(values):
            values = np.asarray(values, dtype=np.float64)
            # Integer and float32 scalars widen exactly; anything else is unsupported.
            return values[:, None]

        for values in (area_d, mean_d):
            for value in values:
                if not isinstance(value, (int, float, np.integer, np.floating)) or (
                        isinstance(value, np.floating) and np.dtype(type(value)).itemsize > 8):
                    raise TypeError('unsupported state scalar')
                if isinstance(value, (int, np.integer)) and abs(int(value)) > 2**53:
                    raise TypeError('integer state scalar is not exactly representable')
        predicted = np.broadcast_to(np.asarray(base)[:, None, :], (len(states), n, 2)).copy()
        for s, state in enumerate(states):
            if state.identity in reserved:
                predicted[s][batch.preferred == state.identity] = state.position
        origin = np.asarray(origin, dtype=np.float64)
        velocity_s = np.asarray(velocity_s, dtype=np.float64)
        self.distance = _norm2(predicted - positions[None], 2)
        self.displacement = _norm2(origin[:, None, :] - positions[None], 2)
        # The same ufuncs on the same values, each written over its own fresh temporary.
        means_plus = means[None] + 1.0
        self.area_ratio = areas[None] / column(area_d)
        self.area_change = np.maximum(self.area_ratio, 1e-12)
        np.abs(np.log2(self.area_change, out=self.area_change), out=self.area_change)
        self.mean_change = means_plus / column(mean_d)
        np.abs(np.log2(self.mean_change, out=self.mean_change), out=self.mean_change)
        self.lineage_area = np.maximum(areas[None] / column(ref_area_d), 1e-12)
        np.abs(np.log2(self.lineage_area, out=self.lineage_area), out=self.lineage_area)
        self.lineage_mean = means_plus / column(ref_mean_d)
        np.abs(np.log2(self.lineage_mean, out=self.lineage_mean), out=self.lineage_mean)
        shape_change = (shapes[None] + 1e-3) / ref_shape[:, None, :]
        np.abs(np.log2(shape_change, out=shape_change), out=shape_change)
        self.lineage_shape = np.mean(shape_change, axis=2)
        velocity = (positions[None] - origin[:, None, :]) / np.asarray(span, dtype=np.float64)[:, None, None]
        new_speed = _norm2(velocity, 2)
        old = np.asarray(old_speed, dtype=np.float64)
        # Every state's cosines at once; pairs where either speed is below the minimum stay 0.0.
        with np.errstate(all='ignore'):
            cosine = np.sum(velocity_s[:, None, :] * velocity, axis=2) / np.maximum(old[:, None] * new_speed, 1e-12)
        active = (old >= minimum)[:, None] & (new_speed >= minimum)
        self.reversal = np.where(active, np.where(-cosine > 0, -cosine, 0.0), 0.0)
        self.predicted = predicted
        self.steps = steps
        self.maximum_distance = maximum_distance
        self.maximum_recovery = maximum_recovery
        self.reserved = reserved
        self.preferred = batch.preferred
        self.preferred_values = None
        self.plain_rows = None
        self.track_index = None
        self.preferred_reserved = None
        self.columns_for = None
        self.whole_for = None
        self.whole = None
        self.split = None
        self.n = n
        self.states = states
        self.index = {id(state): s for s, state in enumerate(states)}
        self.far = self.displacement > maximum_recovery
        # Whether each state's nine pair rows are all finite (prepare's first test).
        finite = np.ones(len(states), bool)
        for values in (self.distance, self.displacement, self.area_ratio, self.area_change, self.mean_change,
                       self.lineage_area, self.lineage_mean, self.lineage_shape, self.reversal):
            finite &= np.isfinite(values).all(axis=1)
        self.finite = finite

    def _whole(self, batch):
        """_prepare_rowwise's per-pair arrays for every state at once, or None.

        Each per-state scalar is the same Python value _prepare_rowwise forms,
        held as a float64 column, so every pair sees the same IEEE operations
        in the same order. Requires plain observations and states, set-valued
        reservations and track sets, and a stacked table within 32 MiB.
        """
        p = batch.params; n = self.n; states = self.states; S = len(states)
        if 11 * S * n * 8 > 32 * 1024**2 or self.preferred.dtype.kind not in 'iu':
            return None
        observations = batch.observations
        if self.plain_rows is None:
            self.plain_rows = all(type(o) is dict and isinstance(o.get('local_id'), (int, np.integer))
                                  and type(o.get('green_track_ids')) is set for o in observations)
        reserved = self.reserved
        if (not self.plain_rows or type(reserved) not in (set, frozenset)
                or any(type(state.identity) is not int or type(state.green_track_ids) not in (set, frozenset)
                       for state in states)):
            return None
        recovery = self.maximum_recovery
        steps = np.asarray(self.steps)
        one = steps == 1
        displacement, reversal = self.displacement, self.reversal
        fast_bound = float(p.get('fast_motion_max_link_px', recovery))
        empty = np.ones((S, n), bool)
        empty[one] = displacement[one] > fast_bound
        if getattr(batch, 'defer_far', False): empty &= ~(displacement > recovery)
        # _prepare_rowwise answers {} for a one-frame state under these settings.
        skip = one & bool(not bool(p.get('red_led_motion', False))
            and 0.0 >= float(p.get('fast_motion_minimum_support', np.inf))
            and 0 >= int(p.get('fast_motion_minimum_px', 1))
            and 0.0 >= float(p.get('fast_motion_minimum_source_loss_coverage', 0.0))
            and 0.0 >= float(p.get('fast_motion_minimum_destination_gain_coverage', 0.0)))
        if (bool(p.get('reversal_aware_motion', False))
                and 0.0 >= float(p.get('reversal_aware_minimum_motion_support', .7))
                and 0.0 >= float(p.get('reversal_aware_minimum_source_loss_coverage', .7))
                and 0.0 >= float(p.get('reversal_aware_minimum_destination_gain_coverage', .7))
                and 0 >= int(p.get('reversal_aware_minimum_supportive_px', 8))):
            empty[one] &= ~((reversal[one] > 0) & (displacement[one] <= float(
                p.get('reversal_aware_max_link_px', p['max_link_px']))))
        if self.track_index is None:
            index = {}
            for i, o in enumerate(observations):
                for track in o['green_track_ids']:
                    index.setdefault(track, []).append(i)
            self.track_index = {track: np.array(v, dtype=np.intp) for track, v in index.items()}
        # Each track's holders are distinct observations, so the count of a
        # (state, observation) cell is the number of (track, holder) entries for it.
        held, owners = [], []
        for s, state in enumerate(states):
            for track in state.green_track_ids:
                holders = self.track_index.get(track)
                if holders is not None:
                    held.append(holders)
                    owners.append(s)
        if held:
            sizes = np.fromiter((len(h) for h in held), np.intp, len(held))
            cells = np.repeat(np.asarray(owners, np.intp) * n, sizes) + np.concatenate(held)
            shared = np.bincount(cells, minlength=S * n).reshape(S, n).astype(np.asarray([0]).dtype, copy=False)
        else:
            shared = np.zeros((S, n), dtype=np.asarray([0]).dtype)
        if self.columns_for is not p:
            with np.errstate(all='ignore'):
                self.basic = (self.displacement <= recovery) & (float(p['minimum_area_ratio']) <= self.area_ratio) & (self.area_ratio <= float(p['maximum_area_ratio'])) & (self.mean_change <= float(p['maximum_mean_log2_change']))
                self.leading_cost = (self.distance / max(float(p['max_link_px']), 1e-6)
                    + float(p.get('area_weight', .45)) * self.area_change
                    + float(p.get('mean_weight', .25)) * self.mean_change
                    + float(p.get('lineage_area_weight', 0.0)) * self.lineage_area
                    + float(p.get('lineage_mean_weight', 0.0)) * self.lineage_mean
                    + float(p.get('lineage_shape_weight', 0.0)) * self.lineage_shape
                    + float(p.get('direction_reversal_weight', 0.0)) * self.reversal
                    - float(p['motion_weight']) * 0.0)
            self.columns_for = p
        # Rows that are not finite are never read; they may raise floating-point warnings.
        with np.errstate(all='ignore'):
            extend = (shared != 0) & (not bool(p.get('shared_motion_requires_support', False))
                or 0.0 >= float(p.get('shared_motion_minimum_support', 0.0)))
            maximum = np.asarray(self.maximum_distance, dtype=np.float64)[:, None]
            permitted = maximum * np.where(extend, float(p.get('shared_motion_distance_multiplier', 1.5)), 1.0)
            allowed = (self.distance <= permitted) & self.basic
            identities = np.asarray([state.identity for state in states])
            if identities.dtype.kind not in 'iu':
                return None
            same = self.preferred[None] == identities[:, None]
            if reserved:
                if self.preferred_reserved is None:
                    self.preferred_reserved = np.asarray([int(v) in reserved for v in self.preferred], bool)
                reserved_pairs = np.broadcast_to(self.preferred_reserved, (S, n)).copy()
                reserved_pairs[[state.identity in reserved for state in states]] = True
                limit = np.asarray([max(m, recovery * np.sqrt(max(k, 1)))
                                    for m, k in zip(self.maximum_distance, self.steps)], dtype=np.float64)[:, None]
                allowed[reserved_pairs] = (same & (self.distance <= limit) & self.basic)[reserved_pairs]
            high = p.get('_high_confidence_identity_set')
            if high is None: high = {int(v) for v in p.get('high_confidence_identity_ids', [])}
            base_reward = float(p.get('preferred_identity_reward', 0.0))
            low_reward = float(p.get('low_confidence_preferred_identity_reward', base_reward))
            reward = np.asarray([low_reward if high and state.identity not in high else base_reward
                                 for state in states], dtype=np.float64)[:, None]
            late = np.asarray([.15 * max(k - 1, 0) for k in self.steps], dtype=np.float64)[:, None]
            cost = (self.leading_cost
                - np.where(shared != 0, float(p.get('shared_motion_reward', 1.5)), 0.0)
                - np.where(same, reward, 0.0) + late)
        floats = np.stack([cost, self.distance, displacement, self.predicted[..., 0], self.predicted[..., 1],
                           self.area_ratio, self.mean_change, self.lineage_area, self.lineage_mean,
                           self.lineage_shape, reversal])
        return dict(empty=empty, skip=skip, allowed=allowed, shared=shared, floats=floats)

    def prepare(self, batch, state, s):
        """_prepare_rowwise's output, read from arrays formed for every state at once."""
        p = batch.params; n = self.n
        if not n: return {}
        if self.whole_for is not p:
            self.whole = self._whole(batch)
            self.whole_for = p
        whole = self.whole
        if whole is None or self.states[s] is not state:
            return self._prepare_rowwise(batch, state, s)
        if not self.finite[s] or whole['skip'][s]: return {}
        constraints = p.get('_retained_source_constraints', {})
        if type(constraints) is dict and not constraints:
            split = self.split
            if split is None or split[0] is not whole:
                split = self.split = (whole, *self._split(whole))
            _, offsets, columns_of, columns = split
            start, stop = offsets[s], offsets[s + 1]
            if start == stop: return {}
            return LazyPlain(columns_of[start:stop], int(self.steps[s]), columns, start)
        else:
            observations = batch.observations
            eligible = [i for i in np.flatnonzero(whole['empty'][s]).tolist()
                        if constraints.get((state.identity, int(observations[i]['local_id']))) is None]
        if not eligible: return {}
        e = np.array(eligible, dtype=np.intp)
        return LazyPlain(eligible, int(self.steps[s]), (whole['allowed'][s, e].tolist(),
            *whole['floats'][:, s, e].tolist(), whole['shared'][s, e].tolist()))

    def _split(self, whole):
        """Every state's eligible observations and their prepare columns, as Python lists."""
        rows, cols = np.nonzero(whole['empty'])
        offsets = np.searchsorted(rows, np.arange(len(self.states) + 1)).tolist()
        return offsets, cols.tolist(), (whole['allowed'][rows, cols].tolist(),
            *whole['floats'][:, rows, cols].tolist(), whole['shared'][rows, cols].tolist())

    def _prepare_rowwise(self, batch, state, s):
        """tracking_plain_batch.prepare on this state's rows.

        Every per-pair value is element-wise, so it is computed only at the
        eligible observations, the only entries the original output kept; the
        output dictionaries are formed when looked up.
        """
        p = batch.params; n = self.n
        if not n: return {}
        steps, maximum, recovery, reserved = self.steps[s], self.maximum_distance[s], self.maximum_recovery, self.reserved
        if not self.finite[s]: return {}
        displacement, reversal = self.displacement[s], self.reversal[s]
        fast_bound = float(p.get('fast_motion_max_link_px', recovery))
        empty = np.ones(n, bool) if steps != 1 else displacement > fast_bound
        if getattr(batch, 'defer_far', False): empty &= ~(displacement > recovery)
        fast_empty = (not bool(p.get('red_led_motion', False)) and steps == 1
            and 0.0 >= float(p.get('fast_motion_minimum_support', np.inf))
            and 0 >= int(p.get('fast_motion_minimum_px', 1))
            and 0.0 >= float(p.get('fast_motion_minimum_source_loss_coverage', 0.0))
            and 0.0 >= float(p.get('fast_motion_minimum_destination_gain_coverage', 0.0)))
        if fast_empty: return {}
        if (bool(p.get('reversal_aware_motion', False)) and steps == 1
                and 0.0 >= float(p.get('reversal_aware_minimum_motion_support', .7))
                and 0.0 >= float(p.get('reversal_aware_minimum_source_loss_coverage', .7))
                and 0.0 >= float(p.get('reversal_aware_minimum_destination_gain_coverage', .7))
                and 0 >= int(p.get('reversal_aware_minimum_supportive_px', 8))):
            empty &= ~((reversal > 0) & (displacement <= float(p.get('reversal_aware_max_link_px', p['max_link_px']))))
        constraints = p.get('_retained_source_constraints', {})
        observations = batch.observations
        if self.plain_rows is None:
            # Every observation holds an integer local_id and a set of green tracks.
            self.plain_rows = all(type(o) is dict and isinstance(o.get('local_id'), (int, np.integer))
                                  and type(o.get('green_track_ids')) is set for o in observations)
        if self.plain_rows and type(constraints) is dict and not constraints:
            # An empty constraint table answers None for every pair.
            eligible = np.flatnonzero(empty).tolist()
        else:
            eligible = [i for i in np.flatnonzero(empty).tolist()
                        if constraints.get((state.identity, int(observations[i]['local_id']))) is None]
        if not eligible: return {}
        e = np.array(eligible, dtype=np.intp)
        distance, displacement, area_ratio = self.distance[s][e], displacement[e], self.area_ratio[s][e]
        area_change, mean_change = self.area_change[s][e], self.mean_change[s][e]
        lineage_area, lineage_mean, lineage_shape = self.lineage_area[s][e], self.lineage_mean[s][e], self.lineage_shape[s][e]
        reversal = reversal[e]
        preferred = self.preferred[e]
        if self.plain_rows and type(state.green_track_ids) is set and not state.green_track_ids:
            # An empty set shares no track with any observation's set.
            shared = np.zeros(len(eligible), dtype=np.asarray([0]).dtype)
        elif self.plain_rows and type(state.green_track_ids) in (set, frozenset):
            # Each shared track adds one to every observation whose set holds it.
            if self.track_index is None:
                index = {}
                for i, o in enumerate(observations):
                    for track in o['green_track_ids']:
                        index.setdefault(track, []).append(i)
                self.track_index = {track: np.array(v, dtype=np.intp) for track, v in index.items()}
            counts = np.zeros(n, dtype=np.asarray([0]).dtype)
            for track in state.green_track_ids:
                holders = self.track_index.get(track)
                if holders is not None:
                    counts[holders] += 1
            shared = counts[e]
        else:
            shared = np.asarray([len(state.green_track_ids & observations[i]['green_track_ids']) for i in eligible])
        extend = (shared != 0) & (not bool(p.get('shared_motion_requires_support', False))
            or 0.0 >= float(p.get('shared_motion_minimum_support', 0.0)))
        permitted = maximum * np.where(extend, float(p.get('shared_motion_distance_multiplier', 1.5)), 1.0)
        if self.columns_for is not p:
            # The basic test and the state-independent leading cost terms, for every pair at once.
            with np.errstate(all='ignore'):
                self.basic = (self.displacement <= recovery) & (float(p['minimum_area_ratio']) <= self.area_ratio) & (self.area_ratio <= float(p['maximum_area_ratio'])) & (self.mean_change <= float(p['maximum_mean_log2_change']))
                self.leading_cost = (self.distance / max(float(p['max_link_px']), 1e-6)
                    + float(p.get('area_weight', .45)) * self.area_change
                    + float(p.get('mean_weight', .25)) * self.mean_change
                    + float(p.get('lineage_area_weight', 0.0)) * self.lineage_area
                    + float(p.get('lineage_mean_weight', 0.0)) * self.lineage_mean
                    + float(p.get('lineage_shape_weight', 0.0)) * self.lineage_shape
                    + float(p.get('direction_reversal_weight', 0.0)) * self.reversal
                    - float(p['motion_weight']) * 0.0)
            self.columns_for = p
        basic = self.basic[s][e]
        allowed = (distance <= permitted) & basic
        same = preferred == state.identity
        if type(reserved) is set and not reserved:
            reserved_pairs = np.zeros(len(eligible), bool)
        elif type(reserved) in (set, frozenset) and self.preferred.dtype.kind in 'iu':
            if state.identity in reserved:
                reserved_pairs = np.ones(len(eligible), bool)
            else:
                if self.preferred_reserved is None:
                    self.preferred_reserved = np.asarray([int(v) in reserved for v in self.preferred], bool)
                reserved_pairs = self.preferred_reserved[e]
        else:
            reserved_pairs = np.asarray([state.identity in reserved or int(v) in reserved for v in preferred])
        if np.any(reserved_pairs):
            limit = max(maximum, recovery * np.sqrt(max(steps, 1)))
            allowed[reserved_pairs] = (same & (distance <= limit) & basic)[reserved_pairs]
        high = p.get('_high_confidence_identity_set')
        if high is None: high = {int(v) for v in p.get('high_confidence_identity_ids', [])}
        reward = float(p.get('preferred_identity_reward', 0.0))
        if high and state.identity not in high: reward = float(p.get('low_confidence_preferred_identity_reward', reward))
        cost = (self.leading_cost[s][e]
            - np.where(shared != 0, float(p.get('shared_motion_reward', 1.5)), 0.0)
            - np.where(same, reward, 0.0) + .15 * max(steps - 1, 0))
        predicted = self.predicted[s][e]
        return LazyPlain(eligible, int(steps), (
            allowed.tolist(), cost.tolist(), distance.tolist(), displacement.tolist(),
            predicted[:, 0].tolist(), predicted[:, 1].tolist(), area_ratio.tolist(), mean_change.tolist(),
            lineage_area.tolist(), lineage_mean.tolist(), lineage_shape.tolist(), reversal.tolist(),
            shared.tolist()))

    def _prepare_before(self, batch, state, s):
        """The previous whole-row prepare, kept for the equality check."""
        rows = Rows(self, s)
        p = batch.params; n = self.n
        if not n: return {}
        steps, maximum, recovery, reserved = self.steps[s], self.maximum_distance[s], self.maximum_recovery, self.reserved
        values = np.stack([self.distance[s], self.displacement[s], self.area_ratio[s], self.area_change[s],
                           self.mean_change[s], self.lineage_area[s], self.lineage_mean[s],
                           self.lineage_shape[s], self.reversal[s]], axis=1)
        if not np.isfinite(values).all(): return {}
        distance, displacement, area_ratio, area_change, mean_change, lineage_area, lineage_mean, lineage_shape, reversal = values.T
        fast_bound = float(p.get('fast_motion_max_link_px', recovery))
        empty = np.ones(n, bool) if steps != 1 else displacement > fast_bound
        if getattr(batch, 'defer_far', False): empty &= ~(displacement > recovery)
        fast_empty = (not bool(p.get('red_led_motion', False)) and steps == 1
            and 0.0 >= float(p.get('fast_motion_minimum_support', np.inf))
            and 0 >= int(p.get('fast_motion_minimum_px', 1))
            and 0.0 >= float(p.get('fast_motion_minimum_source_loss_coverage', 0.0))
            and 0.0 >= float(p.get('fast_motion_minimum_destination_gain_coverage', 0.0)))
        if fast_empty: return {}
        if (bool(p.get('reversal_aware_motion', False)) and steps == 1
                and 0.0 >= float(p.get('reversal_aware_minimum_motion_support', .7))
                and 0.0 >= float(p.get('reversal_aware_minimum_source_loss_coverage', .7))
                and 0.0 >= float(p.get('reversal_aware_minimum_destination_gain_coverage', .7))
                and 0 >= int(p.get('reversal_aware_minimum_supportive_px', 8))):
            empty &= ~((reversal > 0) & (displacement <= float(p.get('reversal_aware_max_link_px', p['max_link_px']))))
        constraints = p.get('_retained_source_constraints', {})
        eligible = [i for i, o in enumerate(batch.observations) if empty[i]
            and constraints.get((state.identity, int(o['local_id']))) is None]
        if not eligible: return {}
        shared = np.asarray([len(state.green_track_ids & o['green_track_ids']) for o in batch.observations])
        extend = (shared != 0) & (not bool(p.get('shared_motion_requires_support', False))
            or 0.0 >= float(p.get('shared_motion_minimum_support', 0.0)))
        permitted = maximum * np.where(extend, float(p.get('shared_motion_distance_multiplier', 1.5)), 1.0)
        basic = (displacement <= recovery) & (float(p['minimum_area_ratio']) <= area_ratio) & (area_ratio <= float(p['maximum_area_ratio'])) & (mean_change <= float(p['maximum_mean_log2_change']))
        allowed = (distance <= permitted) & basic
        same = batch.preferred == state.identity
        reserved_pairs = np.asarray([state.identity in reserved or int(v) in reserved for v in batch.preferred])
        if np.any(reserved_pairs):
            limit = max(maximum, recovery * np.sqrt(max(steps, 1)))
            allowed[reserved_pairs] = (same & (distance <= limit) & basic)[reserved_pairs]
        high = p.get('_high_confidence_identity_set')
        if high is None: high = {int(v) for v in p.get('high_confidence_identity_ids', [])}
        reward = float(p.get('preferred_identity_reward', 0.0))
        if high and state.identity not in high: reward = float(p.get('low_confidence_preferred_identity_reward', reward))
        cost = (distance / max(float(p['max_link_px']), 1e-6)
            + float(p.get('area_weight', .45)) * area_change
            + float(p.get('mean_weight', .25)) * mean_change
            + float(p.get('lineage_area_weight', 0.0)) * lineage_area
            + float(p.get('lineage_mean_weight', 0.0)) * lineage_mean
            + float(p.get('lineage_shape_weight', 0.0)) * lineage_shape
            + float(p.get('direction_reversal_weight', 0.0)) * reversal
            - float(p['motion_weight']) * 0.0
            - np.where(shared != 0, float(p.get('shared_motion_reward', 1.5)), 0.0)
            - np.where(same, reward, 0.0) + .15 * max(steps - 1, 0))
        output = {}
        for i in eligible:
            output[i] = {'allowed': bool(allowed[i]), 'cost': float(cost[i]), 'distance_px': float(distance[i]),
                'observed_displacement_px': float(displacement[i]),
                'predicted_y': float(rows[i][1][0]), 'predicted_x': float(rows[i][1][1]),
                'area_ratio': float(area_ratio[i]), 'mean_log2_change': float(mean_change[i]),
                'lineage_area_log2_change': float(lineage_area[i]), 'lineage_mean_log2_change': float(lineage_mean[i]),
                'lineage_shape_log2_change': float(lineage_shape[i]), 'direction_reversal_penalty': float(reversal[i]),
                'raw_direction_reversal_penalty': float(reversal[i]), 'shared_green_tracks': int(shared[i]),
                'fast_motion_handoff': False, 'reversal_aware_handoff': False,
                'motion_prediction_model': 'constant_velocity', 'gap_steps': int(steps),
                'motion_support': 0.0, 'supportive_px': 0, 'contradictory_px': 0, 'union_px': 0,
                'source_loss_coverage': 0.0, 'destination_gain_coverage': 0.0}
        return output


def matrix(states, batch):
    """A Matrix for these states, or None so every state keeps Batch.build."""
    if not states or not batch.supported or (batch.cache_wrapped and getattr(batch, 'gap', None) is None):
        return None
    try:
        return Matrix(states, batch)
    except Exception:
        return None
