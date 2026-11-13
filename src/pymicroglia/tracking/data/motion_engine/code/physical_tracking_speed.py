"""Exact-output acceleration of physical track comparisons. No scientific thresholds are changed."""
import ast
from contextlib import contextmanager
import inspect
import numpy as np

def candidate_indices(detections, px, py, maximum_distance, cache):
    if not detections:
        return []
    identity = id(detections)
    if cache.get('identity') != identity or cache.get('object') is not detections:
        cache.update(identity=identity, object=detections, xy=np.array([(float(row['x']), float(row['y'])) for row in detections]))
    xy = cache['xy']
    if not np.isfinite([px, py, maximum_distance]).all() or maximum_distance < 0 or (not np.isfinite(xy).all()):
        return enumerate(detections)
    bound = np.nextafter(float(maximum_distance), np.inf)
    eligible = (np.abs(xy[:, 0] - px) <= bound) & (np.abs(xy[:, 1] - py) <= bound)
    return ((int(i), detections[int(i)]) for i in np.flatnonzero(eligible))

@contextmanager
def bounded_link_search(module):
    original = module.link_observations
    tree = ast.parse(inspect.getsource(original))
    matches = []
    for node in ast.walk(tree):
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and (node.iter.func.id == 'enumerate') and (len(node.iter.args) == 1) and isinstance(node.iter.args[0], ast.Name) and (node.iter.args[0].id == 'detections') and isinstance(node.target, ast.Tuple) and ([v.id for v in node.target.elts] == ['di', 'detection']):
            if any((isinstance(v, ast.Name) and v.id == 'endpoint_distance' for v in ast.walk(node))):
                matches.append(node)
    assert len(matches) == 1, 'Original linker structure changed; review required'
    matches[0].iter = ast.parse('_nearby(detections, px, py, maximum_distance)', mode='eval').body
    cache = {}
    namespace = dict(vars(module))
    namespace['_nearby'] = lambda detections, px, py, distance: candidate_indices(detections, px, py, distance, cache)
    exec(compile(ast.fix_missing_locations(tree), original.__code__.co_filename + '::bounded', 'exec'), namespace)
    module.link_observations = namespace['link_observations']
    try:
        yield
    finally:
        module.link_observations = original

@contextmanager
def reuse_start_rows(module):
    original = module.reconnection_hypotheses
    tree = ast.parse(inspect.getsource(original))
    matches = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and (node.targets[0].id == 'startpoint') and (ast.unparse(node.value) == 'after.iloc[0]'):
            matches.append(node)
    assert len(matches) == 1, 'Reconnection implementation changed; review required'
    matches[0].value = ast.parse('_start_row(after)', mode='eval').body
    cache = {}

    def start_row(frame):
        key = id(frame)
        if key not in cache:
            cache[key] = (frame, frame.iloc[0])
        assert cache[key][0] is frame
        return cache[key][1]
    namespace = dict(vars(module))
    namespace['_start_row'] = start_row
    exec(compile(ast.fix_missing_locations(tree), original.__code__.co_filename + '::start_row_cache', 'exec'), namespace)
    module.reconnection_hypotheses = namespace['reconnection_hypotheses']
    try:
        yield
    finally:
        module.reconnection_hypotheses = original

@contextmanager
def physical_tracking_speed(module, *, relative_contrast=False):
    from encounter_speed import bounded_encounters
    from valley_sampling_speed import vectorized_valleys
    from reconnection_speed_spatial import spatial_reconnections
    from motion_sampling_speed import cached_motion_samples
    from relative_valley_ratio import relative_valleys
    from adaptive_link_search import adaptive_link_search
    valleys = relative_valleys if relative_contrast else vectorized_valleys
    with cached_motion_samples(module), adaptive_link_search(module), spatial_reconnections(module), valleys(module), bounded_encounters(module):
        yield
