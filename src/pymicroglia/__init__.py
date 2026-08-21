"""PyMicroglia — microglial imaging analysis.

Registration, filtering, segmentation, tracing and figures for microglial
bioluminescence time-lapses, over a keyed artefact store that makes re-analysis
cheap: the expensive steps write what they *derived* — transforms, masks,
labels, traces — and a changed parameter misses the cache on its own.

Self-contained by rule. This package depends on nothing in
``Protocols/Analysis``: those scripts and macros are sources it was copied from,
never things it imports, calls or edits. It also imports ``analysis_kit``
softly, so a missing audit layer costs a run record and never a result.

Import stays cheap on purpose — reading a parameter block must not pull in a
scientific stack — so the science modules are imported on use, not here.
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import _optional, catalogue, config, imagej, io, metadata, params
from . import recording
from . import results, series, store
from .metadata import ChannelMap, Metadata, Window, assign_channels, usable_window
from .recording import capture
from .series import Series, open_series
from .knowledge import describe, discover, doctor, validate
from .params import ParamBlock, ParamDoc, harvest, harvest_many, parameter_name
from .registry import REGISTRY, ClaimRequired, pending
from .results import Batch, Recording
from .run import ActionInvalid, ActionPending, run_action, run_recorded

__all__ = [
    "__version__",
    # parameters
    "ParamBlock",
    "ParamDoc",
    "harvest",
    "harvest_many",
    "parameter_name",
    # what an agent may run
    "REGISTRY",
    "describe",
    "discover",
    "doctor",
    "validate",
    "pending",
    "run_action",
    "run_recorded",
    "ActionPending",
    "ActionInvalid",
    "ClaimRequired",
    "capture",
    # the artefact store
    "store",
    # the results of a run, without its pixels
    "Recording",
    "Batch",
    "results",
    # the pixels
    "Series",
    "open_series",
    "Metadata",
    "ChannelMap",
    "Window",
    "assign_channels",
    "usable_window",
    "io",
    "metadata",
    "series",
    # submodules
    "params",
    "recording",
    "imagej",
    "catalogue",
    "config",
    "_optional",
]
