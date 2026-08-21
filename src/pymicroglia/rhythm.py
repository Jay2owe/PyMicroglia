"""Periods and cosinor fits, borrowed rather than reimplemented.

Every statistic here comes from ``circadian_workbench``. Nothing in this file
computes a periodogram, a period or a cosinor fit, and a test asserts that by
searching the module for the names. A third Lomb-Scargle in this lab is the
exact drift ``analysis-kit`` was built to end.

**``circadian_workbench`` is a hard optional dependency.** Contrast with
``_optional.kit()``: the audit layer is skipped silently when absent, because
losing a run record must never break the science. Rhythm analysis *is* the
science, so its absence is an error with a named message rather than a quiet
``None``.

**No control, no result.** :func:`test_rhythm` refuses a source that has no
instrumental-control artefact. In the reference dataset the structural channel
showed a 22.8 h sinusoid at Lomb-Scargle power 0.966 that was also present off
tissue, in a second channel, and in image sharpness — a daily focus cycle, not
biology. Running the control is therefore a precondition of making a claim, not
a step somebody remembers.

The verdict travels **attached to the result**, not beside it, so a figure or a
run record cannot show a period without also carrying whether the control
passed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

__all__ = [
    "ControlMissing",
    "RhythmResult",
    "periodogram",
    "cosinor",
    "test_rhythm",
]

#: Search band, hours. From ``dluc_pipeline.py``'s ``LS_PMIN``/``LS_PMAX``.
DEFAULT_PERIOD_RANGE = (15.0, 40.0)
#: Lomb-Scargle power above which a trace counts as rhythmic, from ``LS_RHYTHMIC``.
DEFAULT_RHYTHMIC_POWER = 0.5
#: Resampling bin the workbench works in. Traces here are sampled every half
#: hour or so, and a bin shorter than the sampling interval only interpolates.
DEFAULT_BIN_MINUTES = 30

_CONTROL_NOTE = (
    "In the reference dataset the structural channel showed a 22.8 h sinusoid "
    "at Lomb-Scargle power 0.966 — and the same rhythm was present off tissue "
    "where there is no sample, in a second channel, and in image sharpness. It "
    "was a daily focus cycle, not biology."
)


class ControlMissing(RuntimeError):
    """A rhythm was asked for on a source with no instrumental control."""


def _cw():
    """``circadian_workbench``, or a message saying why this cannot proceed."""
    try:
        import circadian_workbench as workbench
    except ImportError as exc:                      # pragma: no cover - env
        raise ImportError(
            "Rhythm analysis needs circadian-workbench: "
            "pip install circadian-workbench. It is a hard optional "
            "dependency, not a soft one — silently skipping a periodogram "
            "would be worse than failing."
        ) from exc
    return workbench


def _analysis():
    _cw()
    from circadian_workbench import analysis

    return analysis


@dataclass
class RhythmResult:
    """A period, a fit, and the control that says whether to believe them."""

    labels: list[str]
    periods: list[dict[str, Any]] = field(default_factory=list)
    fits: list[dict[str, Any]] = field(default_factory=list)
    #: Never optional. Constructing one of these without a control is what
    #: ``ControlMissing`` prevents.
    control: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def control_passes(self) -> bool:
        return bool(self.control.get("passes"))

    def as_dict(self) -> dict[str, Any]:
        """Serialised for a record — and the control goes with it, always."""
        return {"labels": list(self.labels),
                "periods": list(self.periods),
                "cosinor": list(self.fits),
                "instrumental_control": dict(self.control),
                "control_passes": self.control_passes}


# ------------------------------------------------------------- the adapter
def _frame(times_h, values):
    """A trace as the workbench's selected frame: timestamps and one column.

    The workbench works in wall-clock time because it was built for activity
    records; a trace here is hours from the start of a window. The conversion
    is a fixed epoch plus the hours, which keeps the spacing exact and makes
    the absolute date meaningless — which it is.
    """
    import numpy as np
    import pandas as pd

    hours = np.asarray(times_h, float)
    values = np.asarray(values, float)
    if len(hours) != len(values):
        raise ValueError(f"{len(hours)} times against {len(values)} values")
    epoch = pd.Timestamp("2000-01-01")
    frame = pd.DataFrame({
        "timestamp": epoch + pd.to_timedelta(hours, unit="h"),
        "activity": values,
    })
    frame["excluded"] = False
    frame["analysis_activity"] = frame["activity"]
    return frame


def _config(period_range, bin_minutes: int, extra: Mapping[str, Any] | None = None):
    analysis = _analysis()
    settings = {"period_min_hours": float(period_range[0]),
                "period_max_hours": float(period_range[1]),
                "bin_minutes": int(bin_minutes)}
    if extra:
        settings.update(extra)
    return analysis.normalized_config(settings)


def periodogram(times_h, values, *, period_range=DEFAULT_PERIOD_RANGE,
                bin_minutes: int = DEFAULT_BIN_MINUTES,
                config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Lomb-Scargle, chi-square and F periodograms, from the workbench.

    Returns the peak and the significance verdict alongside the full curves, so
    a caller that only wants "what period" does not have to know the shape of
    the workbench's result.
    """
    import numpy as np

    analysis = _analysis()
    settings = _config(period_range, bin_minutes, config)
    found = analysis.periodograms(_frame(times_h, values), settings)
    verdict = analysis.rhythm_significance(found, settings)

    power = np.asarray(found["lomb_power"], float)
    periods = np.asarray(found["periods_hours"], float)
    peak = int(np.nanargmax(power)) if np.isfinite(power).any() else None
    return {
        "peak_period_hours": float(periods[peak]) if peak is not None else float("nan"),
        "peak_power": float(power[peak]) if peak is not None else float("nan"),
        "significance": verdict,
        "periods_hours": periods,
        "lomb_power": power,
        "source": "circadian_workbench.analysis.periodograms",
    }


def cosinor(times_h, values, period_hours: float) -> dict[str, Any]:
    """Mesor, amplitude, acrophase and the fit's p-value, from the workbench."""
    analysis = _analysis()
    return dict(analysis.fit_cosinor(_frame(times_h, values), float(period_hours)))


# --------------------------------------------------------------- the gate
def _stored_control(source) -> dict[str, Any] | None:
    from . import controls, store

    found = store.resolve(controls.CONTROL_STAGE, source, required=False)
    return None if found is None else found.load()


def test_rhythm(source, *, traces=None, times_h=None, labels=None,
                output_dir=None, output_name=None, overwrite: bool = False,
                ls_pmin: float = DEFAULT_PERIOD_RANGE[0],
                ls_pmax: float = DEFAULT_PERIOD_RANGE[1],
                ls_n: int = 800,
                ls_rhythmic: float = DEFAULT_RHYTHMIC_POWER,
                bin_minutes: int = DEFAULT_BIN_MINUTES,
                fit_cosinor: bool = True,
                reuse: bool = True) -> RhythmResult:
    """Period-test every trace — but only for a source that has a control.

    There is no ``skip_control``. If a dataset genuinely cannot support a
    control, that is a conversation, not a keyword: a period reported without
    one is indistinguishable from a period that is the microscope's.

    ``ls_n`` is accepted and not used: the workbench walks its own candidate
    grid at a fixed 0.05 h step across ``ls_pmin``..``ls_pmax``, which is
    finer than the engine's 800 points over the same band. Kept in the
    vocabulary because the engine declares it and one name means one thing.
    """
    period_range = (float(ls_pmin), float(ls_pmax))
    import numpy as np

    from . import controls, tracing

    control = _stored_control(source)
    if control is None:
        raise ControlMissing(
            f"No instrumental control for {source}. {_CONTROL_NOTE} "
            "Run controls.run_controls(source) first — its result is stored, "
            "so this costs once per source.")

    if traces is None:
        found = tracing.extract_traces(source, output_dir=output_dir,
                                       reuse=reuse)
        traces, times_h, labels = found.processed, found.times_h, found.labels
    values = np.atleast_2d(np.asarray(traces, float))
    if times_h is None:
        raise ValueError("times_h is required when traces are passed directly")
    names = list(labels) if labels is not None else [
        f"trace_{index + 1}" for index in range(len(values))]

    periods: list[dict[str, Any]] = []
    fits: list[dict[str, Any]] = []
    for name, row in zip(names, values):
        found = periodogram(times_h, row, period_range=period_range,
                            bin_minutes=bin_minutes)
        periods.append({
            "label": name,
            "period_hours": found["peak_period_hours"],
            "power": found["peak_power"],
            "significant": found["significance"].get("significant"),
            "status": found["significance"].get("status"),
            # the engine's simpler test, kept alongside the workbench's so a
            # number quoted from either can be traced to which one said it
            "rhythmic_by_power": bool(found["peak_power"] >= float(ls_rhythmic)),
        })
        if fit_cosinor and np.isfinite(found["peak_period_hours"]):
            fits.append({"label": name,
                         **cosinor(times_h, row, found["peak_period_hours"])})

    result = RhythmResult(labels=names, periods=periods, fits=fits,
                          control=control)

    if output_dir is not None:
        from . import store

        with _open(source) as opened:
            store.put("rhythm", opened.source,
                      {"period_range": [float(v) for v in period_range],
                       "bin_minutes": int(bin_minutes)},
                      kind="scalars", value=result.as_dict(),
                      name=str(output_name or "rhythm"),
                      output_dir=output_dir,
                      method_version=controls.METHOD_VERSION)
    return result


def _open(source):
    from . import series as _series

    return _series.open_series(source)
