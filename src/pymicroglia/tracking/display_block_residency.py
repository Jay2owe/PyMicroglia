"""Keep bounded display blocks within the existing temporary RAM budget."""
from functools import wraps


def render_with_budget(renderer,pool):
    if pool is None or not hasattr(pool,'resident_dimensions'):
        return renderer
    @wraps(renderer)
    def render(*args,**kwargs):
        # The existing renderer bounds its blocks and stores the complete RGB
        # movie on disk. This scope changes only temporary block residency.
        with pool.resident_dimensions(4):
            return renderer(*args,**kwargs)
    return render
