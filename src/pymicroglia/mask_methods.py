"""Replaceable cell-mask predictors, shared by detection and outline refinement.

``unet`` uses the existing learned mask unchanged. Other methods are importable
``package.module:function`` references, so a saved request is replayable without
a process-local registry. They receive images, never Motion identities.
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .tracking.contract import sha256_of


def model_reference(weights, *, default=False) -> dict:
    """Fingerprint a chosen model without loading the network."""
    chosen = weights if weights is not None else (
        os.environ.get("PYMICROGLIA_MASK_WEIGHTS") if default else None)
    result = {"weights": str(chosen) if chosen is not None else None}
    if chosen is not None and Path(chosen).is_file():
        result["weights_sha256"] = sha256_of(chosen)
    return result


def resolve(method: str):
    """Resolve an explicitly selected implementation; never guess a fallback."""
    if not isinstance(method, str) or method.count(":") != 1:
        raise ValueError("custom mask methods need 'package.module:function'")
    module, name = method.split(":")
    function = getattr(importlib.import_module(module), name)
    if not callable(function):
        raise ValueError(f"{method} is not callable")
    return function


def describe(method: str, options: Mapping[str, Any] | None = None, *,
             builtin: str = "unet") -> dict:
    """Validate serialisable settings and identify the selected source code."""
    settings = json.loads(json.dumps(dict(options or {}), allow_nan=False))
    if method == builtin:
        if builtin == "unet" and settings:
            raise ValueError("unet uses the named cut/grow_to/area settings; "
                             "method_options are for custom predictors")
        return {"method": method, "options": settings}
    function = resolve(method)
    source = inspect.getsourcefile(function)
    if not source or not Path(source).is_file():
        raise ValueError(f"{method} must have a source file for provenance")
    return {"method": method, "options": settings,
            "implementation_sha256": sha256_of(source)}


def configure(method: str, weights, cut, options=None) -> tuple:
    """Resolve a model only for methods that use one."""
    description = describe(method, options)
    if method == "unet":
        from .learned_mask import apply as masking

        weights = masking.weights() if weights is None else weights
        cut = masking.ACCEPTED_CUT if cut is None else cut
    return (Path(weights) if weights is not None else None), cut, description


def validate_labels(cells, shape, *, name="mask") -> np.ndarray:
    cells = np.asarray(cells)
    if cells.shape != tuple(shape) or cells.dtype.kind not in "ui" \
            or np.any(cells < 0) or np.any(cells > 65535):
        raise ValueError(f"{name} must contain integer labels 0..65535 "
                         "with exactly the input image shape")
    return cells


def predict(pictures, *, method="unet", weights=None, threads=0,
            cut=None, grow_to=None, min_area=None, max_area=0,
            um_per_px=None, method_options=None) -> dict:
    """Predict labels from a T,Y,X array using an independently chosen method.

    A custom function receives ``pictures, weights=, threads=, um_per_px=,
    options=`` and returns a mapping containing ``cells``, ``probability``
    and optionally ``settings`` and ``warning``. All pixels stay aligned.
    """
    if method == "unet":
        from .learned_mask import apply as masking

        model = masking.hurry(masking.load_model(weights), threads=threads)
        out = dict(masking.mask_pictures(
            model, pictures, um_per_px=um_per_px, cut=cut, grow_to=grow_to,
            min_area=min_area, max_area=max_area,
            source="the recording's own metadata"))
        out["pixel_range"] = list(getattr(model, "pixel_range", ()))
    else:
        options = {"cut": cut, "grow_to": grow_to, "min_area": min_area,
                   "max_area": max_area, **dict(method_options or {})}
        images = np.array(pictures, copy=True)
        images.flags.writeable = False
        out = dict(resolve(method)(images, weights=weights, threads=threads,
                                   um_per_px=um_per_px, options=options))
        out.setdefault("settings", {})
        out["settings"] = {"cut": cut, "um_per_px": um_per_px, **out["settings"]}
        out.setdefault("warning", None)
        out.setdefault("pixel_range", [])
    cells = validate_labels(out["cells"], np.shape(pictures))
    probability = np.asarray(out["probability"])
    if probability.shape != cells.shape or not np.isfinite(probability).all() \
            or np.any(probability < 0) or np.any(probability > 1):
        raise ValueError("mask probability must be finite, in 0..1, "
                         "and have the input image shape")
    out["mask"] = cells > 0
    return out
