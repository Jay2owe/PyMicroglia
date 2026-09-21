"""What a measurement module declares about itself.

Every kind of readout -- morphology, intensity, motility, rhythms -- is a
self-contained module that registers itself here. Adding a new readout means
writing one file and importing it from :mod:`pymicroglia.measure.modules`;
nothing else in the package changes, and the run, the manifest and the
catalogue pick it up automatically.

A module declares what image data it needs. If that data is not available for
a movie the module is skipped and the skip is recorded, rather than failing
the run.

Ported from Motion's ``analysis/registry.py`` on 2026-09-21 and renamed,
because PyMicroglia already has a ``registry`` -- the one that binds catalogue
actions to modules. The contract is unchanged: :class:`Column`, :class:`Output`,
the two decorators (``measurement`` and ``derived``, which Motion called
``register`` and ``register_derived`` and which are kept as aliases so a
module ports without edits), the two dictionaries and the cross-validation in
:func:`declared_columns` and :func:`declared_tables`.

No pandas at import time: this module is read by ``describe`` before anything
is measured, and the type hints that mention a table are strings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterable

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from .context import MeasurementContext

__all__ = [
    "Column",
    "Output",
    "ORIGINS",
    "SHARED_COLUMNS",
    "AnalysisModule",
    "DerivedModule",
    "measurement",
    "derived",
    "register",
    "register_derived",
    "list_modules",
    "get_module",
    "list_derived",
    "get_derived",
    "forget",
    "declared_columns",
    "declared_tables",
]


@dataclass(frozen=True)
class Column:
    """One column a module writes, as the rest of the package needs to talk about it.

    Declared here rather than beside the figures because the module is where
    the number is made, and a label kept somewhere else is a label that goes
    stale silently: nothing fails when they disagree, the axis just says
    something slightly wrong on every figure that draws it.

    ``unit`` may name the frame interval as ``{interval}``: "pixels replaced per
    30 min" has to say 30 min because the number is per frame, and hard-coding
    it makes the label wrong the first time somebody images every 15 minutes.

    ``role`` is a colour family the theme can resolve, so one measurement keeps
    one colour across every figure without any builder repeating the mapping.
    """

    name: str
    label: str
    unit: str = ""
    role: str = "morphology"


@dataclass(frozen=True)
class Output:
    """One CSV a module writes, as the run needs to talk about it.

    ``Column`` says what a number means; this says what a *row* is. Between
    them a module describes its own output completely, and the run writer stops
    having to know anything about particular modules.

    ``grain`` answers "one row per what": the columns whose combination is
    unique in this table. It is what lets the writer decide, without a
    hard-coded list, whether this table is a file of its own or columns of a
    shared table at the same grain.

    ``fold`` is deliberately separate from ``grain``. ``presence`` has the same
    grain as ``cell_frame`` and must NOT be folded into it: it carries a row
    for every cell in every frame including the frames where that cell is
    absent, and folding it would either lose the absences or turn cell_frame
    into a grid of mostly blank rows. Grain says what a row is; ``fold`` says
    whether these are the same rows.

    ``origin`` is ``"measured"`` for a number this package computed and
    ``"tracker"`` for a table copied verbatim out of a tracking run. It picks
    the folder the table is written into, so a reader can tell at a glance
    which half of a run folder this package is answerable for.

    ``optional`` marks a table that is not written for every movie, because the
    inputs it needs are not part of every tracking chain. An absent optional
    table is a skip, not a failure.
    """

    name: str
    grain: tuple[str, ...]
    fold: bool = False
    origin: str = "measured"
    optional: bool = False


#: The two origins an ``Output`` may claim. ``measured`` is this package's own
#: arithmetic; ``tracker`` is somebody else's CSV reproduced without comment.
ORIGINS = frozenset({"measured", "tracker"})


@dataclass
class AnalysisModule:
    """A module that reads the pixels of one movie and returns tables."""

    name: str
    description: str
    requires: tuple[str, ...]
    measure: Callable[["MeasurementContext"], dict[str, "pd.DataFrame"]]
    summarise: Callable[[dict[str, "pd.DataFrame"], "MeasurementContext"], dict] | None = None
    defaults: dict = field(default_factory=dict)
    produces: tuple[Column, ...] = ()
    writes: tuple[Output, ...] = ()

    def available(self, context: "MeasurementContext") -> tuple[bool, str]:
        """Whether this movie carries what the module reads.

        An empty ``channels`` counts as absent, not as present-and-empty. The
        field is a dictionary so that a movie with no extra channels is the
        default rather than a special case, which means the missing-input test
        cannot be ``is None`` for it alone.
        """
        for need in self.requires:
            value = getattr(context, need, None)
            if value is None or (isinstance(value, dict) and not value):
                return False, f"{self.name} needs {need}, which this movie does not have"
        return True, ""


@dataclass
class DerivedModule:
    """A module that reads the joined measurement tables, not the pixels.

    Rhythm fitting and anything else that is arithmetic on already measured
    numbers lives here, so it runs in seconds and can be re-run without
    touching the image stacks again.
    """

    name: str
    description: str
    needs_columns: tuple[str, ...]
    derive: Callable[["pd.DataFrame", "MeasurementContext"], dict[str, "pd.DataFrame"]]
    defaults: dict = field(default_factory=dict)
    produces: tuple[Column, ...] = ()
    writes: tuple[Output, ...] = ()
    resolve_params: Callable[["MeasurementContext"], dict] | None = None

    def parameters(self, context: "MeasurementContext") -> dict:
        """The effective settings used by this module for one recording."""
        if self.resolve_params is not None:
            return dict(self.resolve_params(context))
        return {**self.defaults, **context.module_params(self.name)}


#: The two dictionaries every module registers into. Module-level so that
#: importing a module is what registers it, exactly as in Motion.
MEASUREMENTS: dict[str, AnalysisModule] = {}
DERIVATIONS: dict[str, DerivedModule] = {}


def derived(
    name: str,
    description: str,
    needs_columns: Iterable[str] = (),
    defaults: dict | None = None,
    produces: Iterable[Column] = (),
    writes: Iterable[Output] = (),
    resolve_params: Callable[["MeasurementContext"], dict] | None = None,
) -> Callable:
    """Decorate a function that turns the joined cell-frame table into more tables."""

    def decorator(function: Callable) -> Callable:
        DERIVATIONS[name] = DerivedModule(
            name=name,
            description=description,
            needs_columns=tuple(needs_columns),
            derive=function,
            defaults=dict(defaults or {}),
            produces=tuple(produces),
            writes=tuple(writes),
            resolve_params=resolve_params,
        )
        return function

    return decorator


def measurement(
    name: str,
    description: str,
    requires: Iterable[str] = ("labels",),
    defaults: dict | None = None,
    summarise: Callable | None = None,
    produces: Iterable[Column] = (),
    writes: Iterable[Output] = (),
) -> Callable:
    """Decorate a ``measure`` function to add a module to the package."""

    def decorator(function: Callable) -> Callable:
        MEASUREMENTS[name] = AnalysisModule(
            name=name,
            description=description,
            requires=tuple(requires),
            measure=function,
            summarise=summarise,
            defaults=dict(defaults or {}),
            produces=tuple(produces),
            writes=tuple(writes),
        )
        return function

    return decorator


#: Motion's names for the two decorators, so a module ports without edits.
register = measurement
register_derived = derived


def list_modules() -> list[AnalysisModule]:
    return [MEASUREMENTS[key] for key in sorted(MEASUREMENTS)]


def get_module(name: str) -> AnalysisModule:
    if name not in MEASUREMENTS:
        known = ", ".join(sorted(MEASUREMENTS)) or "none registered"
        raise KeyError(f"unknown analysis module {name!r}; known modules: {known}")
    return MEASUREMENTS[name]


def list_derived() -> list[DerivedModule]:
    return [DERIVATIONS[key] for key in sorted(DERIVATIONS)]


def get_derived(name: str) -> DerivedModule:
    if name not in DERIVATIONS:
        known = ", ".join(sorted(DERIVATIONS)) or "none registered"
        raise KeyError(f"unknown derived module {name!r}; known modules: {known}")
    return DERIVATIONS[name]


def forget(name: str) -> None:
    """Unregister one module by name. For tests that register a stand-in."""
    MEASUREMENTS.pop(name, None)
    DERIVATIONS.pop(name, None)


#: The columns that say which row this is rather than what was measured, so no
#: module declares them: ``identity`` and ``frame_index`` are the join keys
#: (``run._join_cell_frame``), the next three come from
#: ``MeasurementContext.frame_table``, and the last three are stamped onto every
#: table by ``run._stamp`` so a pooled analysis is a concatenation.
SHARED_COLUMNS = frozenset({
    "identity", "frame_index", "imagej_frame", "source_imagej_frame", "hours",
    "stem", "condition", "subject",
})


def declared_columns() -> dict[str, Column]:
    """Every column the registered modules say they write, keyed by name.

    Two modules may declare the same column -- ``morphology`` and ``motility``
    both write a centroid -- and that is not a mistake: the join in
    ``run._join_cell_frame`` keeps one copy and drops the rest. What would be a
    mistake is the two disagreeing about what it means, because then the label
    on the axis depends on which module happened to sort first.
    """
    found: dict[str, Column] = {}
    owner: dict[str, str] = {}
    for module in (*list_modules(), *list_derived()):
        for column in module.produces:
            seen = found.get(column.name)
            if seen is not None and seen != column:
                raise ValueError(
                    f"{module.name} and {owner[column.name]} both write "
                    f"{column.name!r} but describe it differently:\n"
                    f"  {owner[column.name]}: {seen}\n"
                    f"  {module.name}: {column}\n"
                    "One column has one meaning, or the axis label depends on "
                    "which module sorted first."
                )
            found[column.name] = column
            owner.setdefault(column.name, module.name)
    return found


def declared_tables() -> dict[str, Output]:
    """Every table the registered modules say they write, keyed by name.

    Unlike a column, a table has exactly one author. Two modules writing one
    name is refused and both are named, because the second one silently
    overwrites the first in the dictionary the run collects results into, and
    the file that lands is whichever module happened to run last.

    A grain naming a column that no module produces and that is not in
    ``SHARED_COLUMNS`` is refused too: a grain is a promise that those columns
    identify a row, and a promise about a column that does not exist is a table
    nobody can join. An empty grain is allowed only for a copied table, where
    saying what one row is would be this package asserting the meaning of
    somebody else's CSV.
    """
    found: dict[str, Output] = {}
    owner: dict[str, str] = {}
    known = set(declared_columns()) | SHARED_COLUMNS
    for module in (*list_modules(), *list_derived()):
        for output in module.writes:
            if output.name in found:
                raise ValueError(
                    f"{module.name} and {owner[output.name]} both write the table "
                    f"{output.name!r}. One table has one author, or the file that "
                    "lands is whichever module ran last."
                )
            if output.origin not in ORIGINS:
                raise ValueError(
                    f"{module.name} declares {output.name!r} with origin "
                    f"{output.origin!r}; expected one of {sorted(ORIGINS)}"
                )
            if not output.grain and output.origin != "tracker":
                raise ValueError(
                    f"{module.name} declares {output.name!r} without a grain. "
                    "Only a table copied from the tracking pass may leave it "
                    "empty; for a table this package computes, not knowing what "
                    "one row is means not knowing what was written."
                )
            unknown = sorted(set(output.grain) - known)
            if unknown:
                raise ValueError(
                    f"{module.name} declares {output.name!r} at a grain of "
                    f"{', '.join(unknown)}, which no module produces. Declare "
                    "the column in a module's PRODUCES, or the table promises a "
                    "key nothing can join on."
                )
            found[output.name] = output
            owner[output.name] = module.name
    return found
