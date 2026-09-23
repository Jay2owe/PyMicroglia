"""Draw a prepared block of saved evidence on its caller's figure."""
from ._contract import Drawn


def draw(figure, data, *, style=None, **options):
    rendered, axes = data.drawing.render(figure)
    if rendered is not figure:
        raise ValueError("Saved renderer did not use the supplied canvas")
    return Drawn(data.figure_data, axes)
