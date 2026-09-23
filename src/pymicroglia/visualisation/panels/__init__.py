"""The figure grammar, and the only place this package writes a figure.

Everything a plot would otherwise reinvent lives here once: how a stack of
panels is laid out, how an axis is labelled, where a colour comes from, and how
a finished figure reaches disk. ``PyFLASH/plotting.py`` reached 31,497 lines
because none of that was shared — each of its forty public plots grew its own
private layout, saving and labelling helpers, roughly 790 lines apiece. This
module is the shared half, so a new figure here is the drawing and nothing else.

Three rules hold it in place, each checked by a test rather than remembered:

**One save path.** ReproFig rendering appears exactly once in this package, in
:func:`save`. That is what makes the provenance bundle automatic rather than
something each figure has to remember, and it is why no figure can quietly
write a PNG with no table beside it.

**No colour is spelled out.** Every colour is asked for by name from
``analysis_kit.style``. There is no local table to fall back on, and a missing
kit raises :class:`FigureStyleMissing` rather than substituting anything —
four projects have already drawn their own version of the house red.

**No figure computes.** Nothing under ``visualisation/`` imports scipy,
scikit-image, or any module of this package that *produces* an artefact. A
figure receives a table or a stored artefact and draws it, which is also why
every figure's exact plotted data already exists as a table by the time
:func:`save` needs one.

On themes: the house look is the kit's ``pyflash`` theme, and a figure that has
no engine to match uses it. ``theme="engine"`` is the deliberate exception — it
leaves Matplotlib's own defaults alone so a ported figure comes out pixel-for-
pixel like the script it replaces. Changing a trace panel's type size is not a
palette change, and a saved run should not move under somebody.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ... import _optional

__all__ = [
    "Panels",
    "FigureStyleMissing",
    "ENGINE_THEME",
    "HOUSE_THEME",
    "stack",
    "grid",
    "label",
    "ticks",
    "vlines",
    "zero_line",
    "legend",
    "colour",
    "colours",
    "resolve_colour",
    "grey",
    "overflow_colours",
    "overflow_cmap_name",
    "image",
    "save",
    "save_for",
    "default_output_dir",
]

#: Leave Matplotlib's defaults alone. What a ported figure asks for when it has
#: an engine-drawn reference to match.
ENGINE_THEME = "engine"

#: The kit's house look, used by every figure with nothing to reproduce.
HOUSE_THEME = "pyflash"


class FigureStyleMissing(RuntimeError):
    """The house palette is not installed, and there is no local substitute."""


# ------------------------------------------------------------- dependencies
def _plt():
    """Matplotlib's pyplot, on a non-interactive backend.

    Imported here rather than at module scope so that importing PyMicroglia to
    read a parameter block does not pull in a plotting stack, and so a machine
    without Matplotlib can still resolve and describe the figure actions.
    """
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Figures need Matplotlib: pip install 'pymicroglia[figure]'."
        ) from exc
    # force=False leaves an already-chosen backend alone; on a headless run it
    # picks Agg, which is what every engine this was ported from does.
    matplotlib.use("Agg", force=False)
    from matplotlib import pyplot

    return pyplot


def _style():
    """``analysis_kit.style``, or a refusal naming what to install.

    Reached through :mod:`pymicroglia._optional` because that is the only route
    to the kit this package allows itself. Unlike the run record, a missing
    style here is fatal: drawing in substitute colours is the failure this is
    meant to prevent, not a degraded mode of it.
    """
    kit = _optional.kit()
    style = getattr(kit, "style", None) if kit is not None else None
    if style is None:
        raise FigureStyleMissing(
            "Figures need analysis-kit for the house palette: "
            "pip install analysis-kit. This one is hard, unlike the audit "
            "layer — a figure drawn in substitute colours is exactly the "
            "drift the shared palette exists to stop, so there is no local "
            "colour table to fall back on.")
    return style


# ------------------------------------------------------------------ colours
def colour(name: str, group: str | None = None) -> str:
    """One house colour, by name. Unknown names raise."""
    return _style().colour(name, group)


def colours(group: str = "semantic", n: int | None = None) -> list[str]:
    """A series cycle, as values. ``semantic`` is the lab's reporter order."""
    return _style().cycle(group, n)


def resolve_colour(value: Any) -> Any:
    """A house name resolved; a user's own ``#rrggbb`` or CSS name passed on.

    The panel grammar lets somebody write ``#a340d1`` in a request, and that
    must keep working. What it must not do is let a *default* be spelled that
    way, which is why the defaults in the catalogue are names.
    """
    return _style().resolve(value)


def grey(name: str) -> str:
    """A Matplotlib grey *level*, as the string it has to stay.

    ``"0.72"`` and ``0.72`` mean different things to Matplotlib, and converting
    the level to a hex triple changes nothing visually while losing the fact
    that it was a level. The kit stores them as strings for the same reason.
    """
    return _style().GREY_LEVEL[name]


def overflow_cmap_name() -> str:
    """The colormap sampled once a panel holds more traces than the cycle."""
    return _style().SEMANTIC_OVERFLOW_CMAP


def overflow_colours(n: int, *, cmap: str | None = None) -> list[str]:
    """``n`` distinct colours sampled across a colormap.

    Used instead of repeating a cycle. Repeating draws the seventh trace in the
    first one's colour, and on an eight-recording overlay that makes two lines
    impossible to tell apart.
    """
    import matplotlib as mpl

    table = _plt().get_cmap(cmap or overflow_cmap_name())
    return [mpl.colors.to_hex(table(0.05 + 0.9 * k / max(n - 1, 1)))
            for k in range(n)]


# ------------------------------------------------------------------- layout
@dataclass
class Panels:
    """A figure and its axes, with the theme that produced them."""

    figure: Any
    axes: list[Any]
    theme: str = ENGINE_THEME
    head_in: float = 0.0
    size_in: tuple[float, float] = (0.0, 0.0)
    notes: list[str] = field(default_factory=list)

    def __iter__(self):
        return iter(self.axes)

    def __len__(self) -> int:
        return len(self.axes)

    def __getitem__(self, index):
        return self.axes[index]

    @property
    def last(self):
        return self.axes[-1]

    def title(self, text: str, *, subtitle: str = "", size: float = 13.0) -> None:
        """The figure's own title, placed above the reserved head space."""
        whole = f"{text}\n{subtitle}" if subtitle.strip() else text
        height = self.size_in[1] or 1.0
        self.figure.suptitle(whole, fontsize=size,
                             y=1 - 0.12 / height, va="top")

    def tighten(self) -> None:
        height = self.size_in[1] or 1.0
        self.figure.tight_layout(rect=(0, 0, 1, 1 - self.head_in / height))


def _applied(theme: str):
    """Set the house rcParams for one theme, or put Matplotlib's own back.

    ``ENGINE_THEME`` is not "do nothing" — it *restores* Matplotlib's defaults,
    which is what the engine it reproduces started from in a fresh process.
    Doing nothing looked equivalent and was not: a house theme applied earlier
    in the same process is global, so the trace panel came out 724 px wide
    instead of 828 depending on which figure had been drawn before it. A ported
    figure that changes with test order is not a ported figure.

    Both branches mutate global state, because that is what a Matplotlib theme
    is. Which one a figure was drawn under is recorded in its provenance.
    """
    import matplotlib

    if theme == ENGINE_THEME:
        matplotlib.rcdefaults()
        return
    _style().apply(theme)


def stack(n: int, *, height_per_panel: float = 1.4, width: float = 11.5,
          head_in: float = 0.0, share_x: bool = True, share_y: bool = False,
          theme: str = ENGINE_THEME) -> Panels:
    """``n`` panels in one column on a shared time axis.

    The trace panel, the control panel and the detrend-transfer panel are all
    this shape. Height grows with the panel count and the head space is
    reserved rather than borrowed from the top panel, so a long title never
    eats into the first trace.
    """
    plt = _plt()
    _applied(theme)
    height = height_per_panel * max(int(n), 1) + head_in
    figure, axes = plt.subplots(max(int(n), 1), 1, figsize=(width, height),
                                sharex=share_x, sharey=share_y, squeeze=False)
    figure._pymicroglia_theme = theme
    return Panels(figure=figure, axes=list(axes[:, 0]), theme=theme,
                  head_in=head_in, size_in=(width, height))


def grid(rows: int, cols: int, *, width: float = 15.0, height: float = 7.0,
         theme: str = HOUSE_THEME, constrained: bool = False) -> Panels:
    """A rows-by-cols grid, read left to right. Every image figure is one."""
    plt = _plt()
    _applied(theme)
    figure, axes = plt.subplots(int(rows), int(cols), figsize=(width, height),
                                squeeze=False,
                                constrained_layout=bool(constrained))
    figure._pymicroglia_theme = theme
    return Panels(figure=figure, axes=[ax for row in axes for ax in row],
                  theme=theme, size_in=(width, height))


# ---------------------------------------------------------------- furniture
def label(ax, *, x: str | None = None, y: str | None = None,
          title: str | None = None, bold: bool = False,
          size: float = 10.0, title_size: float = 9.0,
          title_where: str = "right") -> None:
    """Axis labels and a corner title, in one call rather than four."""
    if x is not None:
        ax.set_xlabel(x, fontsize=size + 1)
    if y is not None:
        ax.set_ylabel(y, fontsize=size,
                      weight="bold" if bold else "normal")
    if title is not None:
        ax.set_title(title, fontsize=title_size, loc=title_where, pad=2)


def ticks(ax, hours) -> None:
    """Explicit x ticks, or Matplotlib's own when ``hours`` is ``None``."""
    if hours is not None:
        ax.set_xticks(list(hours))


def vlines(ax, hours, *, colour_name: str = "tab:green", alpha: float = 0.45,
           width: float = 0.7) -> None:
    """Periodic vertical lines, drawn behind everything.

    ``tab:green`` is Matplotlib's own named colour, not a house one and not a
    hex literal. The house palette has no role for "furniture that marks a
    day boundary", the kit is another agent's to extend, and recolouring it
    would move every 24 h line in every trace panel already drawn.
    """
    if hours is None:
        return
    for hour in hours:
        ax.axvline(float(hour), color=resolve_colour(colour_name),
                   lw=width, alpha=alpha, zorder=0)


def zero_line(ax, *, colour_name: str = "black", width: float = 0.8) -> None:
    ax.axhline(0, color=resolve_colour(colour_name), lw=width, zorder=1.5)


def legend(ax, *, columns: int = 1, size: float = 8.0,
           where: str = "upper right") -> None:
    ax.legend(loc=where, ncol=int(columns), fontsize=size, frameon=False)


def image(ax, array, *, cmap: str, percentiles: Sequence[float] = (0.5, 99.5),
          log: bool = False, vmin: float | None = None,
          vmax: float | None = None, title: str = ""):
    """One frame, scaled for display and stripped of ticks.

    ``log`` and the percentile range are display transforms and nothing else:
    they change what a reader sees, never a number that leaves this module.
    The engines apply exactly these two to every image panel they draw.
    """
    import numpy as np

    values = np.asarray(array, float)
    if log:
        values = np.log1p(np.clip(values, 0, None))
    low = vmin if vmin is not None else float(np.percentile(values, percentiles[0]))
    high = vmax if vmax is not None else float(np.percentile(values, percentiles[1]))
    drawn = ax.imshow(values, cmap=cmap, vmin=low, vmax=high)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=12)
    return drawn


# ------------------------------------------------------------------- saving
def save(figure, path, *, table: Mapping[str, Sequence[Any]],
         sources: Iterable[Any] = (), claim: str = "",
         artefacts: Iterable[Any] = (), settings: Mapping[str, Any] | None = None,
         statistics: Sequence[Mapping[str, Any]] = (),
         statistics_status: str = "incomplete",
         figure_profile: str = "master",
         figure_safe_columns: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
         public_sources: Mapping[str, str] | None = None,
         formats: Sequence[str] = ("png",), dpi: int | None = None,
         dpi_preset: str | None = None, render_preset: str | None = None,
         width: float | None = None, height: float | None = None,
         format_options: Mapping[str, Any] | None = None,
         allow_reencode: bool = False,
         overwrite: bool = True, bundle: bool = False, ledger: bool = True,
         slug: str | None = None, notes: Iterable[str] = (),
         drawn: Any = None, proof: bool = False,
         statistical_specs: Sequence[Mapping[str, Any]] = (),
         required_grades: Sequence[str] = (),
         signing_key_path: str | None = None,
         signing_password_env: str | None = None,
         trust_policy_path: str | None = None,
         encrypted_sections: Sequence[str] = (),
         encryption_password_env: str | None = None,
         recipient_file: str | None = None,
         broker_policy_path: str | None = None,
         proof_policy: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Write a figure, the exact table behind it, and where that table came from.

    The only ReproFig render path in PyMicroglia. Everything a figure needs to be
    trusted later is written here, in one place, so no figure can be produced
    without it:

    * the figure itself, in every requested format;
    * ``<stem>.csv`` — the values actually drawn, column per series;
    * the folder artefact ledger — every source with its SHA256, the settings
      in force, and the stored artefacts the figure was drawn from;
    * ``<stem>_bundle/`` — the same thing in the ``plot-that`` layout, so a
      figure a registered action produced is already an audited bundle.

    ``table`` is not a convenience. A figure whose plotted values exist only
    inside the PNG cannot be checked, re-drawn at another size, or compared
    with the run that produced it.
    """
    from .. import bundle as bundles
    from reprofig import (
        SourceReference,
        approved_public_tables,
        build_record,
        derive_profile,
        save_figure,
        table_from_data,
        validate_artifact,
    )
    from ... import __version__
    from reprofig.schema import json_safe

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    stem = target.with_suffix("").as_posix()
    extension = target.suffix.lstrip(".")
    order = list(dict.fromkeys(
        ([extension] if extension else [])
        + [str(f).lstrip(".") for f in formats if f]))

    # Every render this call makes, gathered before any of them happens, so
    # there is one ReproFig render call in the package rather than one per destination.
    root = Path(f"{stem}{bundles.BUNDLE_SUFFIX}")
    name = slug or target.stem
    inside = bundles.figure_targets(root, name) if bundle else {}
    renders: list[tuple[Path, int | None]] = [
        (Path(f"{stem}.{suffix}"), dpi) for suffix in order]
    if bundle:
        renders.append((inside["svg"], None))
        renders.append((inside["preview"], 110))

    settings = dict(settings or {})
    settings.setdefault("theme", getattr(figure, "_pymicroglia_theme", ENGINE_THEME))
    described = bundles.describe_sources(sources)
    table_target = Path(f"{stem}.csv").resolve()
    if any(row.get("exists") and Path(row["path"]).resolve() == table_target for row in described):
        raise ValueError("Figure table would overwrite a source; choose a different output directory or name")
    exact_table = table_from_data(
        bundles.table_bytes(table),
        name="plotted_data",
        purpose="plot_and_statistics",
    )
    source_records = [
        SourceReference(
            role=str(item.get("role", "source")),
            relative_path=str(item.get("file_name") or "") or None,
            sha256=item.get("sha256"),
            size_bytes=item.get("bytes"),
            modified_at=item.get("modified"),
            source_id=item.get("name"),
        )
        for item in described
    ]
    master_record = build_record(
        title=name,
        original_stem=target.stem,
        producer={
            "package": "PyMicroglia",
            "package_version": __version__,
            "function": "pymicroglia.visualisation.panels.save",
        },
        analysis={
            "claim": claim,
            "settings": settings,
            "drawn": drawn,
        },
        data_tables=[exact_table],
        statistics=json_safe(list(statistics)),
        statistics_status=statistics_status,
        sources=source_records,
    )
    from ..proof_output import prepare_proof

    proof, policy = prepare_proof(
        figure, master_record, exact_table, claim=claim,
        statistical_specs=statistical_specs, proof=proof,
        proof_policy=proof_policy, required_grades=required_grades,
        signing_key_path=signing_key_path,
        signing_password_env=signing_password_env,
        trust_policy_path=trust_policy_path,
        encrypted_sections=encrypted_sections,
        encryption_password_env=encryption_password_env,
        recipient_file=recipient_file, broker_policy_path=broker_policy_path,
    )
    figure_record = derive_profile(
        master_record,
        figure_profile,
        safe_columns=figure_safe_columns,
        public_sources=public_sources,
    )
    companion_table = (
        exact_table
        if figure_profile == "master"
        else approved_public_tables(
            master_record, safe_columns=figure_safe_columns
        )[0]
    )

    written: list[Path] = []
    final_record = figure_record
    variant_record = master_record
    for out, render_dpi in renders:
        if out.exists() and not overwrite:
            raise FileExistsError(
                f"refusing to overwrite {out}. Pass overwrite=True to replace "
                f"it — a re-run silently replacing the figure you were "
                f"comparing against is a mistake that only has to happen once.")
        selected_options = format_options
        if format_options:
            extension = out.suffix.lower().lstrip(".")
            aliases = {"jpg": "jpeg", "jpe": "jpeg", "tif": "tiff", "heic": "heif"}
            requested = aliases.get(extension, extension)
            keyed = {
                aliases.get(str(key).lower().lstrip("."), str(key).lower().lstrip(".")): value
                for key, value in format_options.items()
            }
            if set(keyed) & {"svg", "pdf", "png", "jpeg", "tiff", "webp", "avif", "heif"}:
                selected_options = keyed.get(requested)
        from .._delivery import render_destination
        with render_destination(out) as staged:
            final_record = save_figure(
                figure,
                staged,
                record=variant_record,
                figure_profile=figure_profile,
                dpi=render_dpi,
                dpi_preset=dpi_preset,
                render_preset=render_preset,
                width=width,
                height=height,
                format_options=selected_options,
                savefig_kwargs={"bbox_inches": "tight"},
                allow_reencode=allow_reencode,
                safe_columns=figure_safe_columns,
                public_sources=public_sources,
                proof=proof,
                proof_policy=policy if policy else None,
            )
        if proof:
            # Protected sections are reused across carrier variants, while
            # each signature is rebound to that carrier's visual reference.
            variant_record = final_record
        report = validate_artifact(
            out,
            expected_profile=figure_profile,
            public_safety=figure_profile != "master",
        )
        if not report.valid:
            raise ValueError(
                "; ".join(
                    issue.message
                    for issue in report.issues
                    if issue.severity == "error"
                )
            )
        if out.parent != root / "fig":
            written.append(out)

    table_path = Path(f"{stem}.csv")
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_bytes((companion_table.contents or "").encode("utf-8"))

    public_output = figure_profile != "master"
    sidecar_sources = [] if public_output else described
    sidecar_settings = {} if public_output else settings
    sidecar_artefacts = () if public_output else artefacts
    sidecar_claim = "Publication-safe figure derivative" if public_output else claim
    sidecar_notes = [] if public_output else list(notes)
    sidecar_drawn = None if public_output else drawn
    provenance_path = None
    if ledger:
        from .._ledger import record
        provenance_path = record(written, table_path, sources=sidecar_sources,
            settings=sidecar_settings, artefacts=sidecar_artefacts, claim=sidecar_claim,
            notes=sidecar_notes, drawn=sidecar_drawn)

    result: dict[str, Any] = {
        "figures": written,
        "table": table_path,
        "provenance": provenance_path,
        "sources": described,
        "bundle": None,
    }
    if proof:
        from ..proof_output import proof_summary

        result["proof"] = proof_summary(
            [out for out, _render_dpi in renders], policy
        )
    if bundle:
        result["bundle"] = bundles.write_bundle(
            root, slug=name, table=table, sources=sidecar_sources,
            claim=sidecar_claim, artefacts=sidecar_artefacts,
            settings=sidecar_settings, notes=sidecar_notes,
            table_csv=(companion_table.contents or "").encode("utf-8"))
    if broker_policy_path:
        from ..proof_output import promote_outputs

        result["broker"] = promote_outputs(
            written, policy_path=broker_policy_path,
            workspace_parent=target.parent, stem=target.stem,
        )
    return result


from ..save_actions import default_output_dir, save_for
