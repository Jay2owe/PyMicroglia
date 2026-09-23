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


def measure(movies: Sequence[Any] | None = None, *, output_dir,
            analysis_config=None, run_label: str | None = None,
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

    ``analysis_config`` loads a complete existing JSON configuration, including
    calibration, saved plot plans and pipeline declarations. Explicit arguments
    override corresponding nonempty settings.

    Returns the run manifest. Every table lands through the artefact store,
    so each results folder carries exactly one ``artefacts.json``.
    """
    from .run import run

    if analysis_config is not None:
        config = load_config(analysis_config)
        if movies is not None:
            config.movies = [MovieSpec.from_dict(movie) for movie in movies]
        if enabled_modules is not None: config.enabled_modules = list(enabled_modules)
        if module_options is not None: config.module_params = dict(module_options)
        # Configuration input retains calibration, display plans and workflow
        # declarations. Scalar settings may be overridden explicitly.
        if frame_interval_min is not None: config.frame_interval_min = frame_interval_min
        if microns_per_pixel is not None: config.microns_per_pixel = microns_per_pixel
        if conditions is not None:
            from .conditions import ConditionSet
            config.conditions = ConditionSet.from_config(conditions)
        if windows: config.windows = [WindowSpec.from_dict(value) for value in windows]
        if metric_groups is not None:
            from .spec import parse_metric_groups
            config.metric_groups = parse_metric_groups(metric_groups)
        if contrasts:
            from .contrasts import parse_contrasts
            config.contrasts = parse_contrasts(contrasts, config.metric_groups)
        if not verify_hashes: config.verify_hashes = False
    else:
        if movies is None:
            raise ValueError('Provide movies or analysis_config')
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



def _rhythm_option_contract():
    """Nested rhythm settings as declared by measurement and its authority."""
    from copy import deepcopy
    from .modules.rhythms import DEFAULTS
    from .. import workbench
    scientific = workbench.scientific_options()
    mapping = {"period_estimation_method": "fit_method",
               "primary_rhythm_test": "significance_method",
               "min_cycles_for_confident_period": "min_cycles"}
    methods = workbench.call("period_methods").data["methods"]
    result = {}
    for name, default in DEFAULTS.items():
        row = deepcopy(scientific.get(mapping.get(name, name), {}))
        row["default"] = deepcopy(default)
        row.setdefault("description", name.replace("_", " ").capitalize())
        result[name] = row
    result["period_estimation_method"]["choices"] = [m["key"] for m in methods]
    result["primary_rhythm_test"]["choices"] = [m["key"] for m in methods if m["gives_significance"]]
    result["period_search_hours"]["units"] = "hours"
    result["period_search_hours"]["description"] = "Lower and upper period search bounds; a search setting, not an assumed biological period."
    return result
