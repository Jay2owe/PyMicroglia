"""Finish the embedded figure locally before delivering it to a synced folder."""
from contextlib import contextmanager
from pathlib import Path
import os
import shutil
import tempfile


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
