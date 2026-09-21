"""The three bit flags of a tracker's provenance sidecar.

``<stem>_provenance.tif`` is one ``uint8`` frame per label frame, in label
frame space, and each pixel carries these three bits. They are moved verbatim
from ``Motion/analysis/io.py``; ``Motion/code/pipeline.py`` keeps its own copy
of the same three names and values, and the two halves never import each
other. When the tracker itself is ported, its copy is deleted and this module
is the one place the values live. Until then a test here pins all three, so a
drift on either side is a failing test rather than a silently reshuffled flag.

``ADDED`` is the narrow one: an outline pixel the segmentation never saw, as
opposed to ``INFERRED``, which is an outline pixel that was seen but whose
owner the tracker worked out. Bit 0 alone is a statement about the *name*;
bit 2 is a statement about the *picture*. On the accepted reference movies
bit 2 has never appeared without bit 0.
"""

from __future__ import annotations

__all__ = ["PROVENANCE_INFERRED", "PROVENANCE_UNRESOLVED", "PROVENANCE_ADDED",
           "FLAGS", "unpack"]

#: The owning identity was decided by reconstruction, not by observing that
#: cell there.
PROVENANCE_INFERRED = 0b001
#: Foreground the tracker never resolved to anyone.
PROVENANCE_UNRESOLVED = 0b010
#: An outline pixel the segmentation never saw: the tracker supplied the
#: pixel, not just the name.
PROVENANCE_ADDED = 0b100

#: Name -> bit, in bit order, for anything that wants to iterate the flags.
FLAGS: dict[str, int] = {
    "inferred": PROVENANCE_INFERRED,
    "unresolved": PROVENANCE_UNRESOLVED,
    "added": PROVENANCE_ADDED,
}


def unpack(packed):
    """Split a packed provenance array into its three boolean planes.

    Takes any array-like the bit operators accept -- a numpy array in practice
    -- and hands back ``{"inferred": ..., "unresolved": ..., "added": ...}``.
    Numpy is not imported here so the contract stays importable without it.
    """
    return {name: (packed & bit).astype(bool) for name, bit in FLAGS.items()}
