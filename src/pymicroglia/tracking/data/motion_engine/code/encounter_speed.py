"""Skip provably distant encounter pairs, retaining original pair order and maths."""
import ast
from contextlib import contextmanager
import inspect
import numpy as np


def nearby_rows(rows, index, cache, minimum_encounter_radius_px=10.0):
    if cache.get('rows') is not rows:
        cache.clear()
        xy = np.array([(r.x, r.y) for r in rows], dtype=float)
        radius = np.array([r.radius_px for r in rows], dtype=float)
        cache.update(rows=rows, xy=xy, radius=radius,
                     finite=bool(np.isfinite(xy).all() and np.isfinite(radius).all()))
    if not cache['finite']:
        return rows[index + 1:]
    if 'neighbors' not in cache:
        xy, radius = cache['xy'], cache['radius']
        count = len(rows)
        neighbors = []
        # Memory budget only: every pair uses the same original bounds.
        block_size = max(1, 262144 // max(count, 1))
        columns = np.arange(count)
        for start in range(0, count, block_size):
            stop = min(start + block_size, count)
            limit = np.nextafter(np.maximum(minimum_encounter_radius_px,
                1.5 * (radius[start:stop, None] + radius[None, :])), np.inf)
            difference = np.abs(xy[None, :, :] - xy[start:stop, None, :])
            eligible = ((difference[:, :, 0] <= limit) & (difference[:, :, 1] <= limit)
                        & (columns[None, :] > np.arange(start, stop)[:, None]))
            neighbors.extend(np.flatnonzero(row) for row in eligible)
        cache['neighbors'] = neighbors
    return (rows[int(i)] for i in cache['neighbors'][index])



@contextmanager
def bounded_encounters(module):
    original = module.encounter_analysis
    tree = ast.parse(inspect.getsource(original))
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.For)
               and isinstance(node.target, ast.Name) and node.target.id == 'right'
               and ast.unparse(node.iter) == 'rows[index + 1:]']
    assert len(matches) == 1, 'Encounter source changed; equivalence review required'
    matches[0].iter = ast.parse('_nearby_rows(rows, index, minimum_encounter_radius_px)', mode='eval').body
    cache = {}
    namespace = dict(vars(module))
    namespace['_nearby_rows'] = lambda rows, index, minimum: nearby_rows(rows, index, cache, minimum)
    exec(compile(ast.fix_missing_locations(tree), original.__code__.co_filename + '::bounded_encounters', 'exec'), namespace)
    module.encounter_analysis = namespace['encounter_analysis']
    try:
        yield
    finally:
        module.encounter_analysis = original
