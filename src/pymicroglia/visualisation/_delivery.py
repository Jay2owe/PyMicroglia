"""Finish the embedded figure locally before delivering it to a synced folder."""
from contextlib import contextmanager
from pathlib import Path
import os
import shutil
import tempfile


@contextmanager
def native_vector_images(figure, destination):
    """Keep nearest-neighbour image pixels native in SVG exports.

    Matplotlib otherwise expands small heatmaps/crops to the output DPI, once
    for tight layout and again for delivery. SVG can store the original pixels
    with its existing pixelated rendering hint. Raster exports and explicitly
    smoothed images retain their original renderer. Large images retain the
    downsampling path to avoid increasing the exported payload.
    """
    changed = []
    if Path(destination).suffix.lower() == '.svg':
        # Figure.axes includes axes inside subfigures. Walking every artist
        # with findobj also asks axes to create their ticks, which can cost
        # more than rendering a small image-free figure.
        for axis in figure.axes:
            # Keep Matplotlib's existing compositing for layered images.
            if len(axis.images) != 1:
                continue
            artist = axis.images[0]
            if artist.get_interpolation() != 'nearest':
                continue
            data = artist.get_array()
            if data is None or data.shape[0] * data.shape[1] > 1_000_000:
                continue
            if not artist.get_transform().is_affine:
                continue
            artist.set_interpolation('none')
            changed.append(artist)
    try:
        yield
    finally:
        for artist in changed:
            artist.set_interpolation('nearest')


@contextmanager
def render_destination(destination):
    """Keep the preceding file intact if rendering or promotion is interrupted."""
    from auto_organotypic.io import replace_with_retry
    destination = Path(destination)
    with tempfile.TemporaryDirectory(prefix='pymicroglia-render-') as temporary:
        rendered = Path(temporary)/destination.name
        yield rendered
        descriptor,name = tempfile.mkstemp(prefix='.figure-',suffix=destination.suffix,
                                           dir=destination.parent)
        os.close(descriptor)
        candidate = Path(name)
        try:
            shutil.copyfile(rendered,candidate)
            replace_with_retry(candidate,destination)
        finally:
            candidate.unlink(missing_ok=True)
