"""Skip link-feature work for pairs proven too far to link, exactly.

Under the feature batch, link_features answers a pair beyond the recovery
distance with a DeferredRejectedPair holding only its arguments. The
assignment routines built one per state-observation pair, then read
'allowed' (always False) or its recorded gap and displacement. Here those
pairs are recorded per row and formed only on lookup; every other pair is
evaluated by link_features in the original order. The subset solver skips
them (a rejected pair leaves its cost at BIG), subset tables index the full
table instead of copying it, and the red-led reservation skips a far pair
only when its recorded age, gap or zero motion evidence already ends in
`continue`, before any evidence is computed.

Under a frozen gap-memory wrapper, a row with a retained measured source sends
each pair either to the ordinary features (the version's own early returns) or
to the features of the measured source. Where the caller discards the memory's
events, a measured-source pair beyond the recovery distance from that source is
recorded the same way: 'allowed' False, and its full frozen result formed only
on lookup of a field not fixed by the source's displacement.
"""
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import inspect

import numpy as np

HASHES = {
    '_assign_legacy': 'f92f10cf95a46cf78967fcd3ff2573f1ccc8dca3342a057a2f875be847f72999',
    '_assign_confidence_layers': '67d8f2fe960dcbaecdb63d119681ef4b2dab96f421a92969386eae6daaa797aa',
    '_solve_subset': '52fb5fe2f249df0a5100ac563f3c3a075dc02580336b6dc679e3c4e352c57f21',
    '_subset_reservations': 'ee9c9275d00f298bac6578cfc6b9546846d093dc72b574e15d39510c09731552',
    '_red_led_motion_reservations': '613bf0023b5b2dcb8c2914e657bc602a3ed23e029eaf93325e53c594e8a5081e',
}

TABLE = '    features = _sparse_pair_table(states, observations, target_t, lag, direction, params)\n'
REPLACEMENTS = {
    '_assign_legacy': [(
        '''    features: dict[tuple[int, int], dict] = {}
    for row, state in enumerate(states):
        for col, observation in enumerate(observations):
            value = link_features(state, observation, target_t, lag, direction, params)
            features[row, col] = value
''', TABLE)],
    '_assign_confidence_layers': [(
        '''    features = {
        (row, col): link_features(
            state, observation, target_t, lag, direction, params)
        for row, state in enumerate(states)
        for col, observation in enumerate(observations)
    }
''', TABLE)],
    '_solve_subset': [(
        '''    for local_row, row in enumerate(state_indices):
        for local_col, col in enumerate(observation_indices):
            value = features[row, col]
''', '''    far = getattr(features, "far_keys", None)
    for local_row, row in enumerate(state_indices):
        for local_col, col in enumerate(observation_indices):
            if far is not None and (row, col) in far:
                continue
            value = features[row, col]
''')],
    '_subset_reservations': [(
        '''    subset_features = {
        (local_row, local_col): features[row, col]
        for local_row, row in enumerate(state_indices)
        for local_col, col in enumerate(observation_indices)
    }
''', '    subset_features = _sparse_subset(features, state_indices, observation_indices)\n')],
    '_red_led_motion_reservations': [(
        '''        pairs = (((row, destination_index) for row in range(len(states)))
                 if direction > 0 else
                 ((destination_index, col) for col in range(len(observations))))
''', '''        pairs = _sparse_red_led_pairs(
            features, states, observations, destination_index, direction, params)
''')],
}


# Frozen measured_soma_gap_memory.features early returns, by file hash: r213
# returns the ordinary features for a state at most one frame from the target;
# r215 instead for an ordinary fast-motion or reversal-aware handoff. Both
# return them for a retained-source certificate or an observation without a
# measured core.
GAP_RULES = {
    '2535e5785260bbc345fb88fd092ffc0d4205699650b2e8f31c1df24edca92bb5': dict(adjacent=True, handoff=False),
    'e265acb11ed35dc5facc2e52798db0b90e9760ef5f4c21315e0c8e2b028c3f01': dict(adjacent=False, handoff=True),
}
_gap_hashes = {}
# Set only around a gap-memory tracking prefix whose memory events the caller
# discards; a deferred measured-source pair appends no event.
_events_discarded = ContextVar('gap_events_discarded', default=False)


@contextmanager
def gap_events_discarded():
    token = _events_discarded.set(True)
    try:
        yield
    finally:
        _events_discarded.reset(token)


def gap_rule(link, memory):
    path = link.__globals__.get('__file__')
    if not path or memory is None:
        return None
    if path not in _gap_hashes:
        try:
            with open(path, 'rb') as handle:
                _gap_hashes[path] = hashlib.sha256(handle.read()).hexdigest()
        except OSError:
            _gap_hashes[path] = None
    rule = GAP_RULES.get(_gap_hashes[path])
    cls = type(memory)
    if rule is None or getattr(cls.features, '__code__', None) is None or cls.features.__code__.co_filename != path:
        return None
    code = link.__code__
    return dict(rule, features=cls.features,
                original=link.__closure__[code.co_freevars.index('original_features')].cell_contents)


class RecordedSource:
    """The memory, answering source() with the answer recorded at the row's start."""
    __slots__ = ('memory', 'reference')

    def __init__(self, memory, reference):
        self.memory = memory
        self.reference = reference

    def source(self, state):
        return self.reference

    def __getattr__(self, name):
        return getattr(self.memory, name)


class DeferredGapPair(Mapping):
    """A measured-source pair beyond the recovery distance from that source."""
    __slots__ = ('call', 'known', 'result', 'stats')

    def __init__(self, call, known, stats):
        self.call = call
        self.known = known
        self.result = None
        self.stats = stats

    def complete(self):
        if self.result is None:
            self.result = self.call()
            self.stats['deferred_materialized'] += 1
        return self.result

    def __getitem__(self, key):
        if key == 'allowed':
            return False
        known = self.known
        if key in known:
            return known[key]
        # The measured-source route is taken only without a certificate.
        if key == 'independent_source_soma_retained':
            raise KeyError(key)
        return self.complete()[key]

    def __iter__(self):
        return iter(self.complete())

    def __len__(self):
        return len(self.complete())


class FarKeys:
    __slots__ = ('rows',)

    def __init__(self, rows):
        self.rows = rows

    def __contains__(self, key):
        entry = self.rows.get(key[0])
        return entry is not None and key[1] in entry[1]


class PairTable:
    """Evaluated pairs, plus far rejected pairs formed on first lookup."""
    __slots__ = ('values', 'rows', 'far_keys', 'make')

    def __init__(self, make):
        self.values = {}
        self.rows = {}
        self.far_keys = FarKeys(self.rows)
        self.make = make

    def __getitem__(self, key):
        value = self.values.get(key)
        if value is None:
            entry = self.rows.get(key[0])
            if entry is None or key[1] not in entry[1]:
                raise KeyError(key)
            value = self.values[key] = self.make(entry, key[1])
        return value

    def far_info(self, row, col):
        return self.rows[row][2](col)


class SubsetFar:
    __slots__ = ('view',)

    def __init__(self, view):
        self.view = view

    def __contains__(self, key):
        view = self.view
        return (view.states[key[0]], view.observations[key[1]]) in view.parent.far_keys


class SubsetView:
    """The original subset dictionary's lookups, read from the full table."""
    __slots__ = ('parent', 'states', 'observations', 'far_keys')

    def __init__(self, parent, states, observations):
        self.parent = parent
        self.states = list(states)
        self.observations = list(observations)
        self.far_keys = SubsetFar(self)

    def __getitem__(self, key):
        return self.parent[self.states[key[0]], self.observations[key[1]]]

    def far_info(self, row, col):
        return self.parent.far_info(self.states[row], self.observations[col])


def subset(features, state_indices, observation_indices):
    if isinstance(features, (PairTable, SubsetView)):
        return SubsetView(features, state_indices, observation_indices)
    return {
        (local_row, local_col): features[row, col]
        for local_row, row in enumerate(state_indices)
        for local_col, col in enumerate(observation_indices)
    }


def red_led_pairs(features, states, observations, destination_index, direction, params):
    pairs = (((row, destination_index) for row in range(len(states)))
             if direction > 0 else
             ((destination_index, col) for col in range(len(observations))))
    far = getattr(features, 'far_keys', None)
    if far is None:
        return pairs
    # The same settings _red_led_motion_reservations reads before its loop.
    minimum_source = float(params.get("red_led_minimum_source_loss_coverage", 0.7))
    minimum_destination = float(params.get("red_led_minimum_destination_gain_coverage", 0.7))
    minimum_px = int(params.get("red_led_minimum_supportive_px", 12))
    minimum_age = int(params.get("red_led_minimum_state_age", 1))
    high_source = float(params.get("red_led_high_minimum_source_loss_coverage", minimum_source))
    high_destination = float(params.get(
        "red_led_high_minimum_destination_gain_coverage", minimum_destination))
    # Zero recorded coverage or support fails its threshold whichever of the
    # ordinary or cross-high thresholds applies.
    zero_rejected = ((0.0 < minimum_source and 0.0 < high_source)
                     or (0.0 < minimum_destination and 0.0 < high_destination)
                     or 0 < minimum_px)
    kept = []
    for row, col in pairs:
        if (row, col) in far:
            if states[row].age < minimum_age:
                continue
            info = features.far_info(row, col)
            if int(info[0]) != 1:
                continue
            # DeferredRejectedPair records zero motion evidence exactly when
            # the gap is one frame and the displacement exceeds this bound.
            if zero_rejected and info[5] > float(params.get('fast_motion_max_link_px', info[7])):
                continue
        kept.append((row, col))
    return kept


def install(module, context, features, batch_type, original, stats, deferred_type):
    """Patch the assignment routines; return a function that restores them."""
    sources = {}
    for name, digest in HASHES.items():
        function = getattr(module, name, None)
        if function is None:
            return lambda: None
        source = inspect.getsource(function)
        if hashlib.sha256(source.encode()).hexdigest() != digest:
            return lambda: None
        sources[name] = source
    stats.setdefault('sparse_far_pairs', 0)
    stats.setdefault('gap_rows', 0)
    stats.setdefault('gap_source_rows', 0)
    stats.setdefault('gap_far_pairs', 0)
    stats.setdefault('gap_source_far_pairs', 0)
    stats.setdefault('sparse_tables', 0)

    def table(states, observations, target_t, lag, direction, params):
        link = module.link_features
        batch = context.get()
        sparse = (isinstance(batch, batch_type) and batch.supported
                  and observations is batch.observations and not (target_t != batch.target)
                  and params is batch.params and getattr(batch, 'defer_far', False)
                  and link is features)
        # Under a frozen gap-memory wrapper a row whose memory source is None
        # receives the ordinary features unchanged (deferred far pairs are
        # completed by the unwrapped original). source() runs once at the row's
        # start, where the wrapper first called it; its later calls for the
        # same state return the cached answer without changing the memory.
        gap = getattr(batch, 'gap', None) if isinstance(batch, batch_type) else None
        gap_sparse = (gap is not None and not sparse and batch.supported and observations is batch.observations
                      and not (target_t != batch.target) and params is batch.params and link is gap[0])

        def make(entry, col):
            stats['pairs'] += 1
            stats['deferred_pairs'] += 1
            if len(entry) > 3 and col in entry[3]:
                return entry[4](col)
            return deferred_type(entry[0], observations[col], target_t, lag, direction, params,
                                 row=entry[2](col), original=original, stats=stats)
        result = PairTable(make)
        values = result.values

        def gap_far(state, reference, row):
            """Record this source row's rejected pairs; return their columns."""
            p = params
            steps = abs(int(target_t) - int(state.t))
            if rule['handoff'] and steps == 1 and (
                    (not bool(p.get('red_led_motion', False))
                     and 0.0 >= float(p.get('fast_motion_minimum_support', np.inf))
                     and 0 >= int(p.get('fast_motion_minimum_px', 1))
                     and 0.0 >= float(p.get('fast_motion_minimum_source_loss_coverage', 0.0))
                     and 0.0 >= float(p.get('fast_motion_minimum_destination_gain_coverage', 0.0)))
                    or (bool(p.get('reversal_aware_motion', False))
                        and 0.0 >= float(p.get('reversal_aware_minimum_motion_support', .7))
                        and 0.0 >= float(p.get('reversal_aware_minimum_source_loss_coverage', .7))
                        and 0.0 >= float(p.get('reversal_aware_minimum_destination_gain_coverage', .7))
                        and 0 >= int(p.get('reversal_aware_minimum_supportive_px', 8)))):
                # Zero motion evidence could itself start a handoff here.
                return ()
            # The state's row, as the wrapper's first ordinary call builds it.
            if batch.state is not state:
                batch.build(state)
            if batch.matrix_row is not None:
                ordinary_far = batch.far_row
                displacement = batch.displacement_row
                recovery = batch.matrix.maximum_recovery

                def spec(col, steps=batch.steps_row, displacement=displacement, recovery=recovery):
                    return (steps, None, None, None, None, displacement[col], None, recovery)
            else:
                rows = batch.rows
                ordinary_far = [value[5] > value[7] for value in rows]
                displacement = [value[5] for value in rows]
                spec = rows.__getitem__
            recovery = float(p.get('maximum_recovery_distance_px', np.inf))
            fast = float(p.get('fast_motion_max_link_px', recovery))
            measured = reference[0]
            measured_steps = abs(int(target_t) - int(measured.t))
            # Batch.build's displacement for the measured snapshot, whose
            # position is the measured state's own array.
            source_displacement = np.linalg.norm(measured.position - batch.positions, axis=1)
            constraints = p.get('_retained_source_constraints', {})
            memory = gap[1]
            areas = memory.observation_areas
            recorded = RecordedSource(memory, reference)
            far = set()
            # A deferred pair keeps only its observation, source distance and
            # whether zero motion evidence is recorded; its known values, call
            # and spec are made when the table is read, from the same values.
            deferred = {}
            for col, observation in enumerate(observations):
                local_id = int(observation['local_id'])
                if constraints.get((state.identity, local_id)) is not None:
                    ordinary = True
                elif rule['handoff'] and steps == 1 and not displacement[col] > fast:
                    # Its motion evidence, so its handoff flags, are measured.
                    continue
                else:
                    # The wrapper's own memo, filled as the wrapper fills it.
                    key = (int(target_t), local_id)
                    if key not in areas:
                        areas[key] = memory.core_area(observation['mask'], target_t)
                    ordinary = areas[key] is None
                if ordinary:
                    if ordinary_far[col]:
                        far.add(col)
                    continue
                distance = float(source_displacement[col])
                if not distance > recovery:
                    continue
                deferred[col] = (observation, distance, measured_steps != 1 or distance > fast)
                far.add(col)
            if far:
                def information(col):
                    held = deferred.get(col)
                    if held is None:
                        return spec(col)
                    # red_led_pairs reads zero motion evidence as a displacement beyond its bound.
                    return (steps, None, None, None, None,
                            held[1] if measured_steps == 1 else np.inf, None, recovery)

                def materialize(col):
                    observation, distance, zero = deferred[col]
                    known = {'allowed': False, 'gap_steps': int(steps), 'observed_displacement_px': distance}
                    if zero:
                        known.update(motion_support=0.0, supportive_px=0, contradictory_px=0, union_px=0,
                                     source_loss_coverage=0.0, destination_gain_coverage=0.0)

                    def call(observation=observation):
                        return rule['features'](recorded, rule['original'], state, observation,
                                                target_t, lag, direction, params)
                    return DeferredGapPair(call, known, stats)
                result.rows[row] = (state, far, information, deferred, materialize)
                stats['sparse_far_pairs'] += len(far)
                stats['gap_far_pairs'] += len(far) - len(deferred)
                stats['gap_source_far_pairs'] += len(deferred)
            return far

        rule = None
        if gap_sparse and observations and _events_discarded.get():
            rule = gap_rule(link, gap[1])
            if rule is not None:
                # observation_areas is keyed by local id; filling it ahead of the
                # wrapper keeps its contents only when each id names one observation.
                local_ids = [int(observation['local_id']) for observation in observations]
                if len(set(local_ids)) != len(local_ids):
                    rule = None
        if (rule is not None and getattr(batch, 'snapshot_states', False) is None
                and getattr(batch, 'snapshot_matrix', False) is None
                and len({state.identity for state in states}) == len(states)):
            # The wrapper costs a gap-source row a second time on its measured
            # snapshot under the row's identity; replace() returns one copy per
            # (snapshot, identity), so making them here, in row order, gives the
            # wrapper the same objects and their rows one matrix. source() keeps
            # each identity's own answer, and no two rows share an identity.
            make = rule['features'].__globals__.get('replace')
            snapshots = {}
            for state in states:
                reference = gap[1].source(state)
                if reference is None or make is None:
                    continue
                if rule['adjacent'] and abs(int(target_t) - int(state.t)) <= 1:
                    continue
                effective = make(reference[0], identity=state.identity)
                snapshots.setdefault(id(effective), effective)
            if snapshots:
                batch.snapshot_states = list(snapshots.values())
        for row, state in enumerate(states):
            far = ()
            row_sparse = sparse
            reference = None
            if gap_sparse and observations:
                stats['gap_rows'] += 1
                reference = None if gap[1] is None else gap[1].source(state)
                if reference is None:
                    row_sparse = True
                else:
                    stats['gap_source_rows'] += 1
                    if rule is not None and rule['adjacent'] and abs(int(target_t) - int(state.t)) <= 1:
                        # r213 returns the ordinary features for every pair of this row.
                        row_sparse = True
                    elif rule is not None:
                        far = gap_far(state, reference, row)
            if row_sparse:
                # features() builds the state's row at its first pair.
                if batch.state is not state:
                    batch.build(state)
                if batch.matrix_row is not None:
                    far = set(np.flatnonzero(batch.matrix.far[batch.matrix_row]).tolist())
                    steps, displacement = batch.steps_row, batch.displacement_row
                    recovery = batch.matrix.maximum_recovery

                    def spec(col, steps=steps, displacement=displacement, recovery=recovery):
                        return (steps, None, None, None, None, displacement[col], None, recovery)
                else:
                    rows = batch.rows
                    far = {i for i, value in enumerate(rows) if value[5] > value[7]}
                    spec = rows.__getitem__
                if far:
                    result.rows[row] = (state, far, spec)
                    stats['sparse_far_pairs'] += len(far)
                    if not sparse:
                        stats['gap_far_pairs'] += len(far)
            for col, observation in enumerate(observations):
                if col in far:
                    continue
                values[row, col] = link(state, observation, target_t, lag, direction, params)
        stats['sparse_tables'] += 1
        return result

    globals_ = getattr(module, '_assign_legacy').__globals__
    helpers = dict(_sparse_pair_table=table, _sparse_subset=subset, _sparse_red_led_pairs=red_led_pairs)
    sentinel = object()
    previous_helpers = {name: globals_.get(name, sentinel) for name in helpers}
    globals_.update(helpers)
    previous = {}
    for name, source in sources.items():
        for old, new in REPLACEMENTS[name]:
            assert source.count(old) == 1, name
            source = source.replace(old, new)
        namespace = {}
        exec(compile(source, f'<sparse-{name.strip("_")}>', 'exec'), globals_, namespace)
        previous[name] = getattr(module, name)
        setattr(module, name, namespace[name])
    stats['sparse_supported'] = True

    def restore():
        for name, function in previous.items():
            setattr(module, name, function)
        for name, value in previous_helpers.items():
            if value is sentinel:
                globals_.pop(name, None)
            else:
                globals_[name] = value
    return restore
