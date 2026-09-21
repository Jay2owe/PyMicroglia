"""Turn tracked labels into tables: one row per cell per frame, rolled up.

Everything downstream -- rhythms, states, figures, the pipelines -- reads the
tables this package writes: one row per cell per frame (``cell_frame``),
rolled up per cell (``cell_summary``) and per frame (``frame_summary``),
pooled across movies, and re-rolled inside declared windows. Each kind of
readout is a module that registers itself in :mod:`.declare`; the chassis
here loads a movie (:mod:`.inputs`), runs every module that has what it
needs (:mod:`.run`), joins and rolls up (:mod:`.summarise`), and stores every
table through the artefact store so each results folder carries one ledger.

Ported from Motion's ``analysis/`` package on 2026-09-21 as stage 03 of the
consolidation plan. ``measure`` below is the catalogue action; ``pool``,
``window`` and ``contrasts`` are three further actions over an existing run
folder, in their own modules.

Deliberately light at import: ``describe`` reads this package before
anything is measured, so pandas is imported inside the functions that need
it and the science modules are imported when a run starts.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .context import ChannelStack, MeasurementContext, ObjectStack, Scale
from .declare import (Column, Output, declared_columns, declared_tables,
                      derived, measurement)
from .spec import (ChannelSpec, ContrastSpec, MeasureConfig, MovieSpec,
                   ObjectSetSpec, SideTableSpec, WindowSpec, load_config)

__all__ = [
    "measure", "parameter_choices", "check_parameters",
    "Scale", "ChannelStack", "ObjectStack", "MeasurementContext",
    "Column", "Output", "measurement", "derived",
    "declared_columns", "declared_tables",
    "MovieSpec", "ChannelSpec", "ObjectSetSpec", "SideTableSpec",
    "WindowSpec", "ContrastSpec", "MeasureConfig", "load_config",
]


def measure(movies: Sequence[Any], *, output_dir, run_label: str | None = None,
            if_exists: str = "version",
            enabled_modules: Sequence[str] | None = None,
            module_options: Mapping[str, Mapping[str, Any]] | None = None,
            frame_interval_min: float | None = None,
            microns_per_pixel: float | None = None,
            conditions: Any = None,
            windows: Sequence[Mapping[str, Any]] = (),
            contrasts: Sequence[Mapping[str, Any]] = (),
            metric_groups: Mapping[str, Any] | None = None,
            verify_hashes: bool = True, claim: str = "") -> dict[str, Any]:
    """Measure every configured module over each movie into ``<output_dir>/<run>/``.

    ``movies`` is a sequence of :class:`MovieSpec` or of mappings in the shape
    ``analysis_config.json`` uses for one movie (relative paths resolve
    against the working directory; ``load_config`` resolves them against the
    file). ``module_options`` is ``{module_name: {option: value}}`` and is
    checked against each module's declared settings before anything runs.
    ``conditions``, ``windows``, ``contrasts`` and ``metric_groups`` take the
    blocks of that file unchanged.

    Returns the run manifest. Every table lands through the artefact store,
    so each results folder carries exactly one ``artefacts.json``.
    """
    from .run import run

    config = MeasureConfig.from_parts(
        movies, frame_interval_min=frame_interval_min,
        microns_per_pixel=microns_per_pixel,
        enabled_modules=enabled_modules, module_params=module_options,
        verify_hashes=verify_hashes, conditions=conditions, windows=windows,
        contrasts=contrasts, metric_groups=metric_groups)
    return run(config, output_dir, run_label=run_label, if_exists=if_exists,
               claim=claim)


def parameter_choices() -> dict[str, list[str]]:
    """What ``describe measure`` offers as the choices for a parameter.

    The module names are data in :mod:`.modules`, so listing them imports
    none of the science.
    """
    from .modules import MODULE_NAMES

    return {"enabled_modules": list(MODULE_NAMES)}


def check_parameters(params: Mapping[str, Any]) -> list[str]:
    """Problems ``validate measure`` should report before anything runs.

    ``module_options`` is checked against each module's declared settings,
    which means loading the modules; an unknown option is reported naming
    the module, exactly as the run would refuse it. A module name nothing
    answers to under ``enabled_modules`` is reported too.
    """
    problems: list[str] = []
    options = params.get("module_options")
    if options:
        from . import modules as _modules
        from .run import check_module_options

        _modules.load()
        try:
            check_module_options(options)
        except (TypeError, ValueError) as exc:
            problems.append(f"module_options: {exc}")
    enabled = params.get("enabled_modules")
    if enabled:
        from .modules import MODULE_NAMES

        unknown = [str(name) for name in enabled if str(name) not in MODULE_NAMES]
        if unknown:
            problems.append(f"enabled_modules names {unknown}, which no module "
                            f"answers to; the modules are {', '.join(MODULE_NAMES)}")
    return problems
