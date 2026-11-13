"""Optional native assignment constraint using coexisting measured somas.

The context applies uniformly to every assignment evaluated by the tracker.
It does not supply cell names, regions, feedback or recording selectors.
"""
from contextlib import contextmanager
from coexisting_soma_gap_evidence import CoexistingSomaGapEvidence


@contextmanager
def constrain_coexisting_gap_assignments(tracking, raw, points):
    evidence = CoexistingSomaGapEvidence(raw, points)
    original_features, original_assign = tracking.link_features, tracking._assign
    original_endpoint = evidence._endpoint
    endpoints = {}
    audit = []
    statistics = dict(assignments=0, eligible_links=0, evidence_queries=0,
                      endpoint_cache_hits=0, vetoed_links=0)

    def endpoint(mask, frame):
        # Cache only within one _assign call. Its input masks remain fixed.
        key = (id(mask), frame)
        if key in endpoints:
            statistics['endpoint_cache_hits'] += 1
        else:
            endpoints[key] = original_endpoint(mask, frame)
        return endpoints[key]

    def features(state, observation, target_t, lag, direction, params):
        result = original_features(state, observation, target_t, lag, direction, params)
        if not result['allowed'] or abs(int(state.t)-int(target_t)) <= 1:
            return result
        statistics['eligible_links'] += 1
        statistics['evidence_queries'] += 1
        certificate = evidence.measure(state.mask, state.t, observation['mask'], target_t)
        if certificate['coexisting_raw_somas']:
            statistics['vetoed_links'] += 1
            audit.append(dict(identity=int(state.identity), from_t=int(state.t),
                              to_t=int(target_t), direction=int(direction),
                              observation_local_id=int(observation['local_id']),
                              original_cost=float(result['cost']), **certificate))
            result = dict(result, allowed=False, coexisting_gap_endpoints=True,
                          independent_source_soma_retained=True)
        return result

    def assign(*args, **kwargs):
        endpoints.clear()
        statistics['assignments'] += 1
        try:
            result = original_assign(*args, **kwargs)
            if any(value.get('coexisting_gap_endpoints') for _, _, value in result[0]):
                raise AssertionError('Assignment override bypassed endpoint coexistence constraint')
            return result
        finally:
            endpoints.clear()

    evidence._endpoint = endpoint
    tracking.link_features, tracking._assign = features, assign
    try:
        yield dict(audit=audit, statistics=statistics)
    finally:
        tracking.link_features, tracking._assign = original_features, original_assign

