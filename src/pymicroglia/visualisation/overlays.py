"""Masks, labels and regions drawn over the frame they were found in.

Stage 07 left this note: "Drawing anything. The QC overlay showing which pixels
became a cell is a figure, and belongs to stage 09." This is that figure, and
the reason it was worth waiting for is that a label image is unreadable on its
own — twelve integers on a black background tell nobody whether the segmenter
found cells or found noise. Over the tissue it took them from, they do.

Three panels, following ``mask_context_figure`` in the dLuc pipeline, because
each answers a different objection:

* the structural channel with the outlines on it — are these on tissue?
* the accumulated bioluminescence with the same outlines — are these where the
  light is?
* the same, plus each object's local background ring — is the ring measuring
  background, or is it measuring the neighbour?

Solid outlines are prominence-segmented cells; dashed are permissive sweep
candidates. The distinction is in the artefact and is drawn, never decided here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from . import panels as _panels
from . import qc as _qc

__all__ = ["OUTLINE_CYCLE", "cell_overlay", "outline", "roi_overlay"]

#: Object outlines cycle through the house reporter colours. Positional, not
#: semantic: object 3's colour means "the third one", nothing more, which is
#: why it comes from a cycle rather than from a name.
OUTLINE_CYCLE = "semantic"


def outline(axis, mask, *, colour: str, width: float = 1.5,
            dashed: bool = False, alpha: float = 1.0) -> None:
    """One boolean mask as a contour at its own edge."""
    import numpy as np

    axis.contour(np.asarray(mask, float), levels=[0.5], colors=[colour],
                 linewidths=width, alpha=alpha,
                 linestyles="--" if dashed else "-")


def _tag(axis, position, text: str, colour: str) -> None:
    """The object's number, boxed so it reads over a bright frame."""
    y, x = float(position[0]), float(position[1])
    axis.text(x + 7, y - 7, str(text), color="white", fontsize=10,
              weight="bold",
              bbox={"fc": "black", "ec": colour, "alpha": 0.8, "pad": 1.0})


def cell_overlay(source, *, output_dir=None, output_name=None,
                 overwrite: bool = False, labels=None, background=None,
                 structural=None, rings=None, candidates=(),
                 theme: str = _panels.HOUSE_THEME, dpi: int = 130,
                 fig_width_in: float = 19.0, fig_height_in: float = 6.5,
                 output_formats: Sequence[str] = ("png",),
                 figure_profile: str = "master",
                 figure_safe_columns: Sequence[str] = (),
                 public_sources: Mapping[str, str] | None = None,
                 dpi_preset: str | None = None,
                 render_preset: str | None = None,
                 render_width_in: float | None = None,
                 render_height_in: float | None = None,
                 format_options: Mapping[str, Any] | None = None,
                 allow_reencode: bool = False,
                 proof: bool = False,
                 required_grades: Sequence[str] = (),
                 signing_key_path: str | None = None,
                 signing_password_env: str | None = None,
                 trust_policy_path: str | None = None,
                 encrypted_sections: Sequence[str] = (),
                 encryption_password_env: str | None = None,
                 recipient_file: str | None = None,
                 broker_policy_path: str | None = None,
                 claim: str = "") -> dict[str, Any]:
    """Which pixels became a cell, over the tissue they came from.

    ``labels`` defaults to the stored segmentation for this source, so the
    usual call is ``cell_overlay(path)`` after ``segment(path)``. ``rings`` is
    the local background each trace was measured against; without it the third
    panel is dropped rather than drawn empty, because an absent ring and a
    ring of zero area are different facts.
    """
    import numpy as np

    stored = None
    if labels is None:
        stored = _stored_segmentation(source)
        labels = stored.load()
    labels = np.asarray(labels)

    profile, background_note = _background(source, background)
    tissue = np.asarray(structural, float) if structural is not None else None

    backgrounds = [(profile, f"bioluminescence ({background_note}) + masks",
                    _qc.DLUC_CMAP, True)]
    if tissue is not None:
        backgrounds.insert(0, (tissue, "structural tissue + masks",
                               _qc.STRUCTURAL_CMAP, False))
    if rings is not None:
        backgrounds.append((profile, "masks + local background rings",
                            _qc.DLUC_CMAP, True))

    count = int(labels.max())
    colours = _panels.colours(OUTLINE_CYCLE)
    grid = _panels.grid(1, len(backgrounds), width=fig_width_in,
                        height=fig_height_in, theme=theme)

    for position, (axis, (array, title, cmap, log)) in enumerate(
            zip(grid, backgrounds)):
        _panels.image(axis, array, cmap=cmap, log=log, title=title)
        for index in range(1, count + 1):
            mask = labels == index
            if not mask.any():
                continue
            colour = colours[(index - 1) % len(colours)]
            outline(axis, mask, colour=colour,
                    dashed=index in set(candidates))
            if rings is not None and title.startswith("masks +"):
                outline(axis, np.asarray(rings) == index, colour=colour,
                        width=0.8, alpha=0.65)
            _tag(axis, _centre(mask), index, colour)

    grid.figure.tight_layout()
    grid.title(f"{count} objects over the tissue they were found in",
               subtitle="solid = prominence-segmented cell; "
                        "dashed = permissive sweep candidate")

    table = _object_table(labels, profile, count, set(candidates))
    return _panels.save_for(
        grid.figure, source, table, stage="cell_overlay",
        output_dir=output_dir,
        output_name=output_name or "cell_masks_overlay", overwrite=overwrite,
        dpi=dpi, artefacts=[stored],
        claim=claim or f"{count} objects sit on tissue and on the "
                       f"bioluminescence",
        output_formats=output_formats, figure_profile=figure_profile,
        figure_safe_columns=(list(figure_safe_columns) or None),
        public_sources=public_sources,
        dpi_preset=dpi_preset, render_preset=render_preset,
        render_width_in=render_width_in, render_height_in=render_height_in,
        format_options=format_options, allow_reencode=allow_reencode,
        proof=proof, required_grades=required_grades,
        signing_key_path=signing_key_path,
        signing_password_env=signing_password_env,
        trust_policy_path=trust_policy_path,
        encrypted_sections=encrypted_sections,
        encryption_password_env=encryption_password_env,
        recipient_file=recipient_file,
        broker_policy_path=broker_policy_path,
        settings={"theme": theme, "candidates": sorted(set(candidates)),
                  "background": background_note})


def roi_overlay(source, *, output_dir=None, output_name=None,
                overwrite: bool = False, polygons: Sequence[Any] = (),
                background=None, structural=None,
                theme: str = _panels.HOUSE_THEME, dpi: int = 130,
                fig_width_in: float = 13.0, fig_height_in: float = 6.5,
                output_formats: Sequence[str] = ("png",),
                figure_profile: str = "master",
                figure_safe_columns: Sequence[str] = (),
                public_sources: Mapping[str, str] | None = None,
                dpi_preset: str | None = None,
                render_preset: str | None = None,
                render_width_in: float | None = None,
                render_height_in: float | None = None,
                format_options: Mapping[str, Any] | None = None,
                allow_reencode: bool = False,
                proof: bool = False,
                required_grades: Sequence[str] = (),
                signing_key_path: str | None = None,
                signing_password_env: str | None = None,
                trust_policy_path: str | None = None,
                encrypted_sections: Sequence[str] = (),
                encryption_password_env: str | None = None,
                recipient_file: str | None = None,
                broker_policy_path: str | None = None,
                claim: str = "") -> dict[str, Any]:
    """Hand-drawn or automatic regions over the frame they were drawn on.

    A region is a decision — it is keyed on the source alone and survives a
    parameter change — so what this figure is for is letting somebody check the
    decision once rather than re-making it every run.
    """
    import numpy as np

    profile, background_note = _background(source, background)
    tissue = np.asarray(structural, float) if structural is not None else None
    frames = [(profile, f"bioluminescence ({background_note}) + regions",
               _qc.DLUC_CMAP, True)]
    if tissue is not None:
        frames.insert(0, (tissue, "structural tissue + regions",
                          _qc.STRUCTURAL_CMAP, False))

    colours = _panels.colours(OUTLINE_CYCLE)
    grid = _panels.grid(1, len(frames), width=fig_width_in,
                        height=fig_height_in, theme=theme)
    table: dict[str, list[float]] = {}
    for axis, (array, title, cmap, log) in zip(grid, frames):
        _panels.image(axis, array, cmap=cmap, log=log, title=title)
        for index, polygon in enumerate(polygons):
            colour = colours[index % len(colours)]
            xs, ys = _xy(polygon)
            axis.plot(list(xs) + xs[:1], list(ys) + ys[:1], lw=1.8,
                      color=colour,
                      label=getattr(polygon, "name", f"region {index + 1}"))
            table[f"region_{index + 1}_x"] = [float(v) for v in xs]
            table[f"region_{index + 1}_y"] = [float(v) for v in ys]
        if polygons:
            _panels.legend(axis, columns=1)

    grid.figure.tight_layout()
    grid.title(f"{len(polygons)} regions",
               subtitle="a region is a decision, kept against the source alone")
    return _panels.save_for(
        grid.figure, source, table or {"region": []}, stage="roi_overlay",
        output_dir=output_dir, output_name=output_name or "roi_overlay",
        overwrite=overwrite, dpi=dpi,
        output_formats=output_formats, figure_profile=figure_profile,
        figure_safe_columns=(list(figure_safe_columns) or None),
        public_sources=public_sources,
        dpi_preset=dpi_preset, render_preset=render_preset,
        render_width_in=render_width_in, render_height_in=render_height_in,
        format_options=format_options, allow_reencode=allow_reencode,
        proof=proof, required_grades=required_grades,
        signing_key_path=signing_key_path,
        signing_password_env=signing_password_env,
        trust_policy_path=trust_policy_path,
        encrypted_sections=encrypted_sections,
        encryption_password_env=encryption_password_env,
        recipient_file=recipient_file,
        broker_policy_path=broker_policy_path,
        claim=claim or f"{len(polygons)} regions sit where they were drawn",
        settings={"theme": theme, "background": background_note})


# ------------------------------------------------------------------ helpers
def _stored_segmentation(source):
    from .. import store

    found = store.resolve("segmentation", source, required=False)
    if found is None:
        raise FileNotFoundError(
            "no stored segmentation for this source. Run "
            "segmentation.segment() first, or pass labels= directly. This "
            "module draws what a stage found; it does not find anything.")
    return found


def _background(source, given) -> tuple[Any, str]:
    """The image the outlines are drawn over, and what it is.

    The right background is the accumulated profile the segmenter thresholded,
    because then the picture shows the image the decision was actually made on.
    That profile is a sum over every frame, which this module may not compute
    and the segmenter does not currently store, so it has to be handed in.

    Without it the fall-back is the first frame — and the returned note says so,
    on the panel and in the provenance. A single frame is a different image, and
    a reader silently shown one would be answering a different question.
    """
    import numpy as np

    if given is not None:
        return np.asarray(given, float), "accumulated profile, supplied"
    with _series(source) as opened:
        return np.asarray(opened.frame(0, 0), float), "frame 0 only"


def _series(source):
    from .. import series

    return series.open_series(source)


def _centre(mask) -> tuple[float, float]:
    import numpy as np

    ys, xs = np.nonzero(mask)
    return (float(ys.mean()), float(xs.mean()))


def _xy(polygon) -> tuple[list[float], list[float]]:
    """A ``roi.Polygon``'s vertices, or any sequence of ``(x, y)`` pairs.

    Both are accepted because a region reaches this figure either from the
    stored decision, which is a ``Polygon``, or from somebody drawing one by
    hand to check a threshold before committing to it.
    """
    if hasattr(polygon, "x") and hasattr(polygon, "y"):
        return ([float(v) for v in polygon.x], [float(v) for v in polygon.y])
    points = list(polygon)
    return ([float(point[0]) for point in points],
            [float(point[1]) for point in points])


def _object_table(labels, profile, count: int, candidates: set
                  ) -> dict[str, list[float]]:
    """One row per object: where it is, how big, and how bright.

    Read off the label image and the profile the figure draws, so the CSV
    beside the figure holds exactly what the picture shows and nothing it does
    not.
    """
    import numpy as np

    rows: dict[str, list[float]] = {"label": [], "area_px": [], "y": [],
                                    "x": [], "mean_profile": [],
                                    "sweep_candidate": []}
    for index in range(1, count + 1):
        mask = labels == index
        if not mask.any():
            continue
        y, x = _centre(mask)
        rows["label"].append(float(index))
        rows["area_px"].append(float(mask.sum()))
        rows["y"].append(y)
        rows["x"].append(x)
        rows["mean_profile"].append(float(np.asarray(profile, float)[mask].mean()))
        rows["sweep_candidate"].append(1.0 if index in candidates else 0.0)
    return rows


