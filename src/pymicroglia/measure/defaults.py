"""The default metric lists that name microglia rather than arithmetic.

The survey behind the Motion port found that the microglial framing enters
the derived modules through their ``defaults`` blocks, not their code:
``rhythms``, ``trend`` and ``recurrence`` fit whatever ``metrics`` names, and
what they name by default is the set of readouts a microglia study asks
about. The arithmetic is generic; the lists are not. They live here, in one
file, so that a user measuring another cell type changes one line and every
module that fits a level, a slope or a state follows.

``morphology``, ``sholl`` and ``surveillance`` are microglia-specific by
design (a ramification index is a microglial idea) and keep their own
docstrings; nothing of theirs is here.
"""

from __future__ import annotations

__all__ = ["MICROGLIA_METRICS", "MICROGLIA_STATE_METRICS"]

#: The measurements a microglial cell can be said to have a *level* of, and
#: therefore the ones ``rhythms`` fits a period to and ``trend`` fits a slope
#: to by default. A configuration that changes one list almost always wants
#: to change the other, and is expected to say so in ``module_options``
#: rather than have one module reach into another's settings.
MICROGLIA_METRICS: tuple[str, ...] = (
    "area_px",
    "corrected_mean",
    "solidity",
    "ramification_index",
    "turnover_index",
    "step_px",
)

#: The broader nine-measurement state vector ``recurrence`` asks its question
#: over: whether a cell returns to a fuller morphological state it has been
#: in before. Deliberately wider than :data:`MICROGLIA_METRICS`; the
#: ``regimes`` classifier asks the narrower size-and-movement question.
MICROGLIA_STATE_METRICS: tuple[str, ...] = (
    "area_px",
    "circularity",
    "solidity",
    "ramification_index",
    "aspect_ratio",
    "skeleton_branches",
    "turnover_index",
    "step_px_gapless",
    "punctateness",
)
