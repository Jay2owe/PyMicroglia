"""Outline proposals without access to cell identity tracking."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .. import mask_methods

SINGLE_FRAME_OPTIONS = frozenset({
    "cut", "grow_to", "min_area", "max_area", "min_gain",
})
COMMON_OPTIONS = {"policy", "policy_options"}


def describe(method="single_frame", options=None) -> dict:
    """Validate a refiner independently of the base mask settings."""
    builtins = {"single_frame", "cd68", "saved"}
    description = mask_methods.describe(method, options,
                                         builtin=method if method in builtins else "single_frame")
    if method in builtins:
        from ..learned_mask.cd68 import OPTIONS as CD68_OPTIONS

        allowed = {"single_frame": SINGLE_FRAME_OPTIONS, "cd68": CD68_OPTIONS,
                   "saved": {"path", "kind", "threshold"}}[method]
        unknown = set(options or {}) - allowed - COMMON_OPTIONS
        if unknown:
            name = "single-frame" if method == "single_frame" else method
            raise ValueError(f"unknown {name} refinement options: {sorted(unknown)}")
    policy = (options or {}).get("policy", "photon_support" if method == "single_frame" else "overlap")
    from .mask_proposals import validate_policy

    validate_policy(policy, (options or {}).get("policy_options"))
    description["external_inputs"] = {
        name: {"path": str(path.resolve()), "sha256": mask_methods.sha256_of(path)}
        for name, path in input_files(method, options).items()}
    return description


def input_files(method, options=None):
    """External geometry inputs are hashed just like model checkpoints."""
    if method != "saved":
        return {}
    if not (options or {}).get("path"):
        raise ValueError("saved refinement requires a proposal path")
    path = Path(options["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"proposal": path}


def propose(raw, baseline, *, method="single_frame", weights=None,
            threads=4, um_per_px=None, options=None, capture=None) -> tuple:
    """Run a provider and the common shape-to-cell correspondence.

    Custom ``package.module:function`` refiners receive read-only copies of
    ``raw, baseline`` plus ``weights=, threads=, um_per_px=, options=``. Return
    a ``kind/data`` mapping as documented in :mod:`.mask_proposals`. Existing
    array-returning providers retain their baseline-labelled contract, except
    when intersection explicitly uses only their foreground.
    """
    description = describe(method, options)
    options = dict(options or {})
    from . import mask_proposals

    policy = options.pop("policy", "photon_support" if method == "single_frame" else "overlap")
    policy_options = dict(options.pop("policy_options", {}) or {})
    if method == "single_frame":
        weights, cut, _ = mask_methods.configure("unet", weights, options.get("cut"))
        single = mask_methods.predict(
            raw, weights=weights, threads=threads, um_per_px=um_per_px,
            cut=cut, grow_to=options.get("grow_to"),
            min_area=options.get("min_area"), max_area=options.get("max_area", 0))["cells"]
        proposal = {"kind": "labels", "data": single}
        if policy == "photon_support":
            policy_options.setdefault("min_gain", options.get("min_gain", 0.03))
        description["resolved_cut"] = cut
    elif method == "cd68":
        from ..learned_mask.cd68 import predict

        proposal = predict(raw, weights=weights, threads=threads, options=options)
    elif method == "saved":
        import tifffile

        path = Path(options["path"])
        kind = options.get("kind", "labels")
        data = json.loads(path.read_text(encoding="utf-8")) if kind == "polygons" else tifffile.imread(path)
        if isinstance(data, np.ndarray) and data.ndim == 2 and len(raw) == 1:
            data = data[None]
        proposal = {"kind": kind, "data": data, "threshold": options.get("threshold", 0.5)}
    else:
        images, detections = np.array(raw, copy=True), np.array(baseline, copy=True)
        images.flags.writeable = detections.flags.writeable = False
        proposal = mask_methods.resolve(method)(
            images, detections, weights=weights, threads=threads,
            um_per_px=um_per_px, options=options)
    gated, replacements, conversion = mask_proposals.attach(
        raw, baseline, proposal, policy=policy, options=policy_options, capture=capture)
    description["proposal"] = conversion
    return gated.astype(baseline.dtype), replacements, description


def validate_proposals(baseline, gated):
    """Only a boundary can change; each detection keeps its existing pixels."""
    mask_methods.validate_labels(gated, baseline.shape, name="refined detections")
    for old, new in zip(baseline, gated):
        if set(np.unique(old)) - {0} != set(np.unique(new)) - {0}:
            raise ValueError("refinement must retain every detection label in each frame")
        if np.any((old > 0) & (new > 0) & (old != new)):
            raise ValueError("refinement cannot transfer pixels between detections")
