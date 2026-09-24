"""Per-recording helpers for the Auto-Organotypic microglia pipeline."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..review import Review


def _validate_cell_grid_options(image_options: Mapping[str, Any] | None,
                                video_options: Mapping[str, Any] | None) -> None:
    """Reject misspelled visual settings before running the upstream chain."""
    import inspect
    from auto_organotypic import grid, video_grid
    from ..visualisation.cell_image_grid import cell_image_grid
    from ..visualisation.cell_video_grid import cell_video_grid

    for label, supplied, wrapper, upstream in (
            ("tracked_cell_grid_options", image_options, cell_image_grid,
             grid.stack_to_grid),
            ("tracked_cell_video_grid_options", video_options, cell_video_grid,
             video_grid.stack_to_video_grid)):
        if supplied is None:
            continue
        if not isinstance(supplied, Mapping):
            raise TypeError(f"{label} must be a settings mapping")
        reserved = {"raw", "labels", "sources", "output_dir",
                    "period_decisions",
                    "source_frame_offset", "frame_interval_h"}
        forbidden = sorted(reserved.intersection(supplied))
        if forbidden:
            raise ValueError(f"{label} cannot replace pipeline-owned inputs: "
                             + ", ".join(forbidden))
        allowed = (set(inspect.signature(wrapper).parameters) |
                   set(inspect.signature(upstream).parameters))
        unknown = sorted(set(supplied) - allowed)
        if unknown:
            raise ValueError(f"{label} has unknown setting(s): "
                             + ", ".join(map(str, unknown)))
        nested = supplied.get("display_options", {})
        if not isinstance(nested, Mapping):
            raise TypeError(f"{label}.display_options must be a mapping")
        bad_nested = sorted(set(nested) - set(inspect.signature(upstream).parameters))
        if bad_nested:
            raise ValueError(f"{label}.display_options has unknown setting(s): "
                             + ", ".join(map(str, bad_nested)))


def _cell_grids(kind: str, tracking: Mapping[str, Any],
                handoff: Mapping[str, Any], eligibility: Mapping[str, Any],
                folder: Path, *, options: Mapping[str, Any] | None = None,
                period_evidence: Mapping[str, Any] | None = None
                ) -> dict[str, Any]:
    """Render one recording per call with its own eligibility label view."""
    import json
    from ..visualisation.cell_image_grid import cell_image_grid
    from ..visualisation.cell_video_grid import cell_video_grid

    if kind not in ("images", "videos"):
        raise ValueError("cell grid destination must be images or videos")
    document = json.loads(Path(handoff["outputs"]["motion_inputs"])
                          .read_text(encoding="utf-8"))
    settings = dict(options or {})
    name = settings.pop("output_name", None)
    action = cell_image_grid if kind == "images" else cell_video_grid
    out: dict[str, Any] = {}
    for stem, result in tracking.items():
        prepared = document["prepared"][stem]
        labels = eligibility[stem]["views"][kind]["labels"]
        destination = folder / "visual" / kind / stem
        shared = ((period_evidence or {}).get(stem)
                  if settings.get("significant_period_only") else None)
        if settings.get("significant_period_only") and shared is None:
            raise ValueError(f"{stem} has no shared period decisions")
        call_options = {**settings, **({"period_decisions": shared["decisions"]}
                                     if shared is not None else {})}
        if "um_per_px" not in call_options and prepared.get("um_per_px") is not None:
            call_options["um_per_px"] = prepared["um_per_px"]
        try:
            report = action(
                prepared["measurement_raw"], labels,
                output_dir=destination,
                output_name=name or f"{stem}_cell_{'grid' if kind == 'images' else 'video_grid'}",
                source_frame_offset=int(result["source_frame_offset"]),
                frame_interval_h=float(prepared["frame_interval_min"]) / 60.0,
                **call_options)
        except ValueError as error:
            if ("contains no tracked cells" not in str(error) and
                    "no cells meet the cell-grid selection" not in str(error)):
                raise
            out[stem] = {"status": str(error),
                         "eligibility_labels": str(labels),
                         "display_only": True, "output": None}
            continue
        out[stem] = {
            "output": report["output"], "display_only": True,
            "eligibility_labels": str(labels),
            "cell_identities": report["cell_grid"]["cell_identities"],
            "selection": report["cell_grid"].get("selection"),
            "period_report": shared["report"] if shared is not None else None,
            "source_frame_offset": int(result["source_frame_offset"]),
            "crop": [one["provenance"]["crop_size_px"]
                     for one in report.get("tile_sources", [])],
            **({"cycle_selection": report["cell_grid"]["cycle_selection"]}
               if kind == "images" else {}),
        }
    return out


def _shared_cell_periods(tracking: Mapping[str, Any],
                         handoff: Mapping[str, Any], folder: Path, *,
                         recipe: Mapping[str, Any], trace_channel: int = 1
                         ) -> dict[str, Any]:
    """Save one complete-population period test for both grid destinations."""
    import json
    from ..visualisation.cell_selection import period_evidence

    document = json.loads(Path(handoff["outputs"]["motion_inputs"])
                          .read_text(encoding="utf-8"))
    destination = folder / "tracked_cell_period_selection"
    destination.mkdir(parents=True, exist_ok=True)
    results = {}
    for stem, result in tracking.items():
        prepared = document["prepared"][stem]
        try:
            evidence = period_evidence(
                prepared["measurement_raw"], result["labels"],
                source_frame_offset=int(result["source_frame_offset"]),
                frame_interval_h=float(prepared["frame_interval_min"]) / 60.0,
                trace_channel=trace_channel, period_recipe=recipe)
        except ValueError as error:
            if "contains no tracked cells" not in str(error):
                raise
            evidence = {"period_recipe": dict(recipe), "decisions": {},
                        "status": "no tracked cells"}
        path = destination / f"{stem}_period_decisions.json"
        path.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n",
                        encoding="utf-8")
        results[stem] = {"report": str(path),
                         "decisions": evidence["decisions"]}
    return results


def _measure_tracked(tracking: Mapping[str, Any], handoff: Mapping[str, Any],
                     folder: Path, *, modules: Sequence[str],
                     eligibility: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Measure accepted labels against original photons, never Motion counts."""
    import json

    from .. import measure as _measurement

    document = json.loads(Path(handoff["outputs"]["motion_inputs"])
                          .read_text(encoding="utf-8"))
    out = {}
    for stem, result in tracking.items():
        prepared = document["prepared"][stem]
        decisions = result.get("decisions")
        raw_path = prepared["measurement_raw"]
        selected = ((eligibility or {}).get(stem, {}).get("views", {})
                    .get("analysis", {}))
        labels_path = selected.get("labels", result["labels"])
        label_sha = selected.get("sha256", result.get("sha256", {}).get("labels"))
        movie = {
            "stem": stem, "labels": labels_path, "raw": raw_path,
            "unclaimed": result.get("unclaimed"),
            "provenance": result.get("provenance"),
            "evidence": result.get("evidence"),
            "history": decisions.get("root") if isinstance(decisions, Mapping)
                       else None,
            "source_frame_offset": result["source_frame_offset"],
            "sha256": {**{role: digest for role, digest in
                          result.get("sha256", {}).items()
                          if role not in ("raw", "labels")},
                       "labels": label_sha,
                       "raw": prepared["measurement_raw_sha256"]},
        }
        measured = _measurement.measure(
            [movie], output_dir=folder / "tracked_measurement" / stem,
            frame_interval_min=prepared["frame_interval_min"],
            enabled_modules=tuple(modules), if_exists="error")
        run_record = measured["run"]
        out[stem] = {"folder": run_record["folder"],
                     "run_label": run_record["run_label"]}
    return out


def _eligible_tracks(tracking: Mapping[str, Any], handoff: Mapping[str, Any],
                     folder: Path, *, enabled: bool, max_gap_h: float | None,
                     max_gap_frames: int | None,
                     max_missing_frames: int | None,
                     max_missing_fraction: float | None,
                     exclude_from: str | Sequence[str]) -> dict[str, Any]:
    """Destination views over final identities; Motion files remain untouched."""
    import json
    from ..eligibility import DESTINATIONS, evaluate

    document = json.loads(Path(handoff["outputs"]["motion_inputs"])
                          .read_text(encoding="utf-8"))
    out = {}
    for stem, result in tracking.items():
        prepared = document["prepared"][stem]
        if enabled:
            out[stem] = evaluate(
                result["labels"], output_dir=folder / "eligibility" / stem,
                frame_interval_h=float(prepared["frame_interval_min"]) / 60.0,
                max_gap_hours=max_gap_h,
                max_gap_frames=max_gap_frames,
                max_missing_frames=max_missing_frames,
                max_missing_fraction=max_missing_fraction,
                exclude_from=exclude_from)
        else:
            digest = result.get("sha256", {}).get("labels")
            out[stem] = {
                "eligible_identities": [], "excluded_identities": [],
                "exclude_from": [],
                "views": {destination: {
                    "labels": result["labels"], "filtered": False,
                    "sha256": digest,
                } for destination in DESTINATIONS},
                "audit": None, "report": None,
            }
    return out


def _tracked_videos(tracking: Mapping[str, Any], handoff: Mapping[str, Any],
                    eligibility: Mapping[str, Any], folder: Path, *,
                    options: Mapping[str, Any] | None) -> dict[str, Any]:
    import json
    from ..video import tracked_cell_video

    document = json.loads(Path(handoff["outputs"]["motion_inputs"])
                          .read_text(encoding="utf-8"))
    settings = dict(options or {})
    reserved = {"labels", "output_dir", "source_frame_offset",
                "frame_interval_h"}
    conflicts = reserved.intersection(settings)
    if conflicts:
        raise ValueError("tracked_video_options cannot replace pipeline-owned "
                         f"inputs: {', '.join(sorted(conflicts))}")
    out = {}
    for stem, result in tracking.items():
        prepared = document["prepared"][stem]
        labels = eligibility[stem]["views"]["videos"]["labels"]
        out[stem] = tracked_cell_video(
            prepared["measurement_raw"], labels=labels,
            output_dir=folder / "tracked_video" / stem,
            source_frame_offset=result["source_frame_offset"],
            frame_interval_h=float(prepared["frame_interval_min"]) / 60.0,
            **settings)
    return out


def _tracked_images(tracking: Mapping[str, Any], handoff: Mapping[str, Any],
                    eligibility: Mapping[str, Any], folder: Path, *,
                    options: Mapping[str, Any] | None) -> dict[str, Any]:
    import json
    from ..visualisation.overlays import tracked_cell_image

    document = json.loads(Path(handoff["outputs"]["motion_inputs"])
                          .read_text(encoding="utf-8"))
    settings = dict(options or {})
    reserved = {"labels", "output_dir", "source_frame_offset",
                "frame_interval_h"}
    conflicts = reserved.intersection(settings)
    if conflicts:
        raise ValueError("tracked_image_options cannot replace pipeline-owned "
                         f"inputs: {', '.join(sorted(conflicts))}")
    out = {}
    for stem, result in tracking.items():
        prepared = document["prepared"][stem]
        labels = eligibility[stem]["views"]["images"]["labels"]
        out[stem] = tracked_cell_image(
            prepared["measurement_raw"], labels=labels,
            output_dir=folder / "tracked_image" / stem,
            source_frame_offset=result["source_frame_offset"],
            frame_interval_h=float(prepared["frame_interval_min"]) / 60.0,
            **settings)
    return out


def _mask_every(recordings, folder: Path, notes: Review, *, say=None,
                **settings) -> dict[str, Any]:
    """The learned mask over every recording the chain registered.

    ``say`` is the chain's own ``on_progress``, handed the same shape it gets
    from Auto-Organotypic's stages. This is the slowest thing in the run by a
    wide margin -- minutes per recording against seconds for everything around
    it -- so a watcher that is told about every other stage and not this one
    reports a run that has stopped.
    """
    from . import cell_masks as _masking

    # Left out rather than passed as ``None``: the window's default is the
    # duration the weights were trained on, and it lives in one place next to
    # the weights. Forwarding a ``None`` here would ask that function to treat
    # "not specified" as a number.
    if settings.get("window_hours") is None:
        settings.pop("window_hours", None)

    out: dict[str, Any] = {}
    for index, row in enumerate(recordings, 1):
        path = Path(str(_said(row, "path")))
        one = _Registered(path, row)
        where = folder / "cell_masks"
        started = time.monotonic()
        # Two masks from one model, answering two different questions. The
        # per-frame stack is what identity tracking links; the still is one
        # region per cell for the whole recording, which is what a trace needs.
        made = _masking.mask_run(one, where, notes, **settings)
        made["still"] = _masking.mask_still(
            one, where, notes,
            **{k: v for k, v in settings.items() if k != "window_hours"})
        out[str(path)] = made
        if say is not None:
            say({"stage": "cell_masks",
                 "status": "reused" if made.get("reused") else "done",
                 "seconds": round(time.monotonic() - started, 3),
                 "recording": path.name, "of": len(recordings), "n": index})
    return out


def _measure_every(recordings, masks, folder: Path, notes: Review, *,
                   options: Mapping[str, Any] | None,
                   count: int, alpha: float, seed: int) -> dict[str, Any]:
    """Traces, control and rhythm from Auto-Organotypic; admissibility from here.

    Measuring a labelled region over time is that package's, and is shared with
    every pipeline that does it, so pointing its own engine at these labels is
    what keeps "imports the chain" true. The one question it does not ask is
    whether an object is real at all, and that stays here.
    """
    from auto_organotypic import region_trace as _trace

    out: dict[str, Any] = {}
    for row in recordings:
        path = Path(str(row["path"]))
        mask = masks.get(str(path))
        if mask is None:
            continue
        labels = _labels_from(mask)
        settings = dict(options or {})
        settings.setdefault("output_dir", str(folder / "cells" / path.stem))
        traced = _trace.run(str(path), labels=labels, **settings)
        admissibility = _admissible(path, row, labels, notes, count=count,
                                    alpha=alpha, seed=seed)
        out[str(path)] = {
            "outputs": {"traces": settings["output_dir"],
                        **mask["still"], **mask["outputs"]},
            "summary": {"regions": int(labels.max()) if labels.size else 0,
                        **admissibility, "trace": _small(traced)},
            "admissible": admissibility.get("admissible", 0),
        }
    return out


def _labels_from(mask: Mapping[str, Any]):
    """The still mask's labels, read back off disk.

    They are not carried in the run record and must not be: a record names a
    file, it never contains one, and a label image per recording inside a JSON
    is how a run record becomes the largest thing in the output tree.
    """
    import numpy as np
    import tifffile

    return np.asarray(tifffile.imread(mask["still"]["mask_cells_still"]),
                      np.uint16)


def _admissible(path: Path, row, labels, notes: Review, *, count: int,
                alpha: float, seed: int) -> dict[str, Any]:
    """Area-matched decoys on tissue: could a patch that size look this rhythmic?

    A mask says where a cell is, and not whether it is one.

    *On tissue* carries the whole test. A decoy placed off tissue has a
    mask-minus-ring baseline of about zero, so its dF/F divides by nothing and
    every real object is then scored against a distribution of near-singular
    ratios -- which is why ``controls.decoy_test`` refuses that configuration
    outright rather than warning about it. So the tissue mask is cut here the
    way every other stretch of this package cuts it: the channel the
    bioluminescence is actually brightest inside, chosen by measuring it.
    """
    import numpy as np

    from .. import controls as _controls
    from .. import segmentation as _segmentation
    from .. import series as _series

    try:
        with _series.open_series(str(path)) as opened:
            assigned = _metadata_channels(opened)
            frames = opened.shape[0]
            planes = [np.asarray(opened.frame(index, assigned.dluc), np.float32)
                      for index in range(frames)]
            times = _hours(opened, frames)
            chosen, _table, _why = _segmentation.pick_tissue_mask(
                opened, {"dluc": assigned.dluc, "bf": assigned.bf,
                         "struct": assigned.struct,
                         "other": list(assigned.other)},
                np.mean(planes, axis=0))
        decoys = _controls.decoy_test(
            None, 0, np.asarray(labels), np.asarray(chosen["_tissue"], bool),
            times, count=count, alpha=alpha, planes=planes,
            rng=np.random.default_rng(seed))
    except Exception as exc:                     # noqa: BLE001 - reported
        notes.note("admissibility", chosen="not tested", confidence="low",
                   changes_result=True,
                   why=f"the decoy test could not run on {path.name}: {exc}",
                   remedy="Every object below is unscreened: nothing here says "
                          "whether a patch of tissue that size could look this "
                          "rhythmic by chance.")
        return {"admissible": 0, "tested": 0, "decoys": "failed"}
    if chosen["dluc_on_minus_off_counts"] <= 0:
        # The mask is the wrong way round: the light is outside it. Decoys are
        # then drawn from pixels a cell could not be in, and every object below
        # is being compared against the wrong background.
        notes.note("admissibility", chosen="tested against a doubtful tissue "
                                           "mask", confidence="low",
                   changes_result=True,
                   why=" ".join(_why) or "no channel gives a tissue mask with "
                                         "the bioluminescence inside it.",
                   remedy="channels='dluc=<n>,bf=<n>,struct=<n>'")
    kept = [one for one in decoys.records if one.get("admissible")]
    return {"admissible": len(kept), "tested": len(decoys.records),
            "tissue_channel": int(chosen["channel"]),
            "decoys": "on tissue, absolute counts"}


# ------------------------------------------------------------- small parts
def _said(row: Any, field: str) -> Any:
    """One field of a recording, however this caller happens to hold it.

    The chain hands its stages ``Recording`` objects; this package's own run
    record holds the same recordings as the dictionaries they serialise to. The
    mask reads three fields and does not care which it was given, so this is
    where the two shapes meet rather than two copies of the stage body.
    """
    if isinstance(row, Mapping):
        return row.get(field)
    return getattr(row, field, None)


def _value(derived: Any) -> float | None:
    """The number out of a ``Derived``, in either shape, or ``None``."""
    if isinstance(derived, Mapping):
        derived = derived.get("value")
    else:
        derived = getattr(derived, "value", derived)
    try:
        return float(derived) if derived is not None else None
    except (TypeError, ValueError):
        return None


class _Registered:
    """One registered recording, wearing the face ``cell_masks`` reads.

    Deliberately not a :class:`~.registered.Prepared`: that carries a window,
    shifts and a cosmic-ray history from *this* package's registration, and this
    recording went through Auto-Organotypic's. Reusing the type would make a
    manifest say a run did something it did not.
    """

    def __init__(self, path: Path, row: Any):
        self.path = Path(path)
        # The path, not ``self``: the store identifies a recording by its file,
        # and ``cell_masks`` names its outputs from whatever is here.
        self.source = self.path
        self._row = row
        self._pixels = None

    @property
    def dluc(self):
        import numpy as np

        from .. import series as _series

        if self._pixels is None:
            with _series.open_series(str(self.path)) as opened:
                channels = _metadata_channels(opened)
                self._pixels = np.stack([
                    np.asarray(opened.frame(index, channels.dluc), "float32")
                    for index in range(opened.shape[0])])
        return self._pixels

    @property
    def frame_interval_h(self) -> float:
        seconds = _value(_said(self._row, "interval_s"))
        return seconds / 3600.0 if seconds else 0.0

    @property
    def um_per_px(self):
        return _value(_said(self._row, "pixel_size_um"))


def _metadata_channels(opened):
    from .. import metadata as _metadata

    return _metadata.assign_channels(opened)


def _hours(opened, frames: int):
    import numpy as np

    times = getattr(opened.meta, "times_h", None)
    if times is not None and len(times) >= frames:
        return np.asarray(times[:frames], float)
    interval = float(getattr(opened.meta, "interval_s", 0) or 0) / 3600.0
    return np.arange(frames, dtype=float) * (interval or 0.5)


def _small(value: Any) -> Any:
    """A stage result with anything document-sized left out of the manifest."""
    if isinstance(value, Mapping):
        return {key: _small(item) for key, item in value.items()
                if not hasattr(item, "shape")}
    if isinstance(value, (list, tuple)):
        return [_small(item) for item in value][:20]
    return value
