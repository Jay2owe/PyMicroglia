"""Validate stage contracts and resolve conventions for the microglia chain."""
from __future__ import annotations

from typing import Any, Mapping, Sequence


def _registry(folder, options: Mapping[str, Any]):
    """The conventions the chain resolved for this folder, or ``None``.

    Read back from the chain's own output root when it had one; otherwise
    resolved again from the same manifest under the same options, which is
    the same answer. A folder the chain could not index has none, and this
    run says so rather than failing after the chain has finished.
    """
    from auto_organotypic import conventions as _conventions

    root = options.get("output_root")
    found = _conventions.load(root) if root else None
    if found is not None:
        return found
    try:
        from auto_organotypic.sources import read_manifest
        recordings = read_manifest(folder, instrument=options.get("manifest_kind"))
        return _conventions.resolve(recordings, options=options)
    except Exception:
        return None


def _the_stages_we_turn_off_still_exist(chain: Any, skipped: Mapping[str, str]) -> None:
    """Refuse a run whose ``skip`` names a stage the chain no longer has.

    ``skip`` is filtered against the stage list and an unknown name in it is
    simply ignored — sensible there, and the one place this pipeline is exposed
    to an upstream rename. If ``outline`` were ever called something else, the
    three names below would quietly stop skipping anything and every recording
    would be outlined, oriented and cropped around a two-lobe structure these
    recordings do not contain. It would not raise; it would produce a full run
    of confident, wrong results.

    Everything else about the chain updates itself here, because nothing else
    about it is named in this file. This is the exception, so it is checked
    rather than trusted.
    """
    known = set(chain.stage_names())
    missing = sorted(set(skipped) - known)
    if missing:
        raise ValueError(
            f"auto_microglia turns off {', '.join(missing)}, which "
            f"auto_organotypic {getattr(chain, '__version__', '')} no longer "
            f"has. Until NOT_OURS is brought back into step, a run would "
            f"silently perform the stages this pipeline exists to leave out. "
            f"The chain's stages are now: {', '.join(chain.stage_names())}.")


def _the_mask_feeds_the_measurement(names: Sequence[str]) -> None:
    """Here the mask is a step, so it comes *before* the cells, not after.

    The opposite of what ``check_stage_order`` enforces on
    :mod:`.dluc_single_cell`, and not a contradiction of it: there the mask is a
    branch off a finished measurement, here it is what finds the cells that get
    measured. Both are checked, because either one backwards measures something
    other than what the run claims.
    """
    from . import StageOrderError

    order = list(names)
    if "cells" in order and "cell_masks" in order:
        if order.index("cell_masks") > order.index("cells"):
            raise StageOrderError(
                "the learned cell mask ran after the cell measurement. In this "
                "pipeline the mask is what finds the cells, so a measurement "
                "made before it is a measurement of the previous run's labels "
                "or of nothing at all.")
