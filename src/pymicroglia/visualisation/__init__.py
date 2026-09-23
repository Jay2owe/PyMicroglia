"""Figures, and the four rules that stop this folder becoming plotting.py.

``PyFLASH/plotting.py`` is 31,497 lines — half that codebase — holding 620
functions, 557 of them private, to serve about forty public plots. Roughly 790
lines per plot, because each one carried its own private stack for layout,
saving, labelling and statistics routing. The same had already started here:
figures were 1,042 of ``dluc_pipeline.py``'s 3,643 lines before this package
existed.

The four rules are structural facts a test checks, not guidelines:

1. **No file here exceeds 600 lines.** When the trace panel outgrew it, the
   computing half moved to ``pymicroglia.trace_tables`` — which is the cap
   working, not the cap failing.
2. **No figure computes.** Nothing under this package imports scipy,
   scikit-image, or any module of PyMicroglia that *produces* an artefact. A
   figure receives a table or a stored artefact and draws it. This is the
   load-bearing one: a figure that cannot compute cannot grow a private helper
   stack, because there is nothing for the helpers to do.
3. **One save path.** ReproFig rendering appears exactly once in the package, in
   :func:`panels.save`, which is what makes every figure arrive with its exact
   plotted table and a provenance record beside it.
4. **No colour is spelled out.** Every colour comes from ``analysis_kit.style``
   by name; no six-digit hex literal and no ``rcParams`` assignment appears
   here.

The families, and where each lives:

``panels``
    The grammar. Layout, labelling, colour resolution, and the save path.
``traces``
    The stacked trace panel, ported from ``trace_panel_figure.py``.
``qc``
    Registration, cosmic-ray, channel-identity and first/mid/last-frame checks.
``overlays``
    Masks, labels and regions drawn over the frame they were found in.
``bundle``
    The ``plot-that`` provenance layout, written rather than remembered.

Importing this package does not import Matplotlib. Every figure module reaches
it inside a function, so ``pymicroglia describe`` still answers on a machine
with no plotting stack, and the figure actions stay resolvable rather than
pending.
"""

from __future__ import annotations

__all__ = ["FAMILIES", "panels", "traces", "qc", "overlays", "bundle"]

#: One entry per figure family, with the question each family answers. Kept as
#: data so ``describe`` and a future coverage test can both read it, in the
#: spirit of PyFLASH's describe-coverage sets: a new family that nobody
#: classified is a test failure rather than an omission somebody notices later.
FAMILIES: dict[str, str] = {
    "morphology": "how saved cell shapes change across distance and time",
    "coupling": "how saved measurements relate within individual cells",
    "surveillance": "how much tracked footprint is gained, lost and retained over time",
    "spatial": "where saved cell measurements and supported timing occur in the field",
    "motility": "how recorded cell positions change between observations",
    "review": "what the saved observations and diagnostics show before interpreting them",
    "audit": "which period methods are supported by independent saved checks",
    "rhythms": "what periods and evidence were saved for each measured trace",
    "relationships": "how saved measurements relate within and across cells",
    "behaviour": "which shared cell states the saved evidence supports",
    "coordination": "which cells share supported changes at their measured positions",
    "intervention": "how measured responses change around a declared intervention",
    "traces": "time traces stacked one panel per cell or region",
    "qc": "did this processing step do what it claims",
    "overlays": "which pixels became an object, over the tissue they came from",
}

from . import bundle, overlays, panels, qc, traces  # noqa: E402

from .figures import available_views
