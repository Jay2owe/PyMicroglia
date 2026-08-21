"""The quality-control record every science stage produces.

One shape, used by registration here and by every later stage: a **summary**
row saying whether the step is trustworthy, a **per-frame table** saying where
it is not, and a handful of **scalars** that go into the run record so a
thousand runs stay skimmable.

Two rules about the numbers, both learned from the engines this was copied from:

**Fixed-precision strings, not floats.** The engines write
``f"{value:.12f}"``. That is not decoration — it is what makes two runs of the
same analysis produce identical files, and what lets a stored artefact be
compared with ``diff`` rather than with a tolerance. Values are formatted once,
here, and stored as text.

**The threshold is recorded next to the verdict.** ``qc_pass`` alone is a claim;
``qc_pass`` beside ``max_residual_threshold_px`` is a claim somebody can check,
and raising a threshold to make a warning disappear becomes visible in the
artefact instead of invisible in a command line.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

__all__ = ["QCReport", "fixed", "table_from_rows", "percentiles"]


def fixed(value: Any, places: int = 12) -> str:
    """A number as the engines write it: fixed precision, no exponent.

    Ports must produce the same characters, not merely the same value. A stored
    artefact that differs only in formatting looks like a changed result to
    every tool that compares files.
    """
    return f"{float(value):.{places}f}"


def table_from_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Any]]:
    """Rows of a CSV as columns, keeping the first row's column order."""
    if not rows:
        return {}
    columns = list(rows[0])
    return {name: [row.get(name) for row in rows] for name in columns}


def percentiles(values, *, places: int = 12) -> dict[str, str]:
    """Median, 95th percentile and maximum, formatted once."""
    import numpy as np

    array = np.asarray(values, dtype=float)
    return {
        "median": fixed(np.median(array), places),
        "p95": fixed(np.percentile(array, 95), places),
        "max": fixed(np.max(array), places),
    }


@dataclass
class QCReport:
    """What one processing step is prepared to say about its own output.

    ``summary`` and ``frames`` become tier-A artefacts under the names the
    engines already use, because those names appear in a methods paragraph that
    ends up in a manuscript. ``scalars`` is the short form that goes into the
    run record.
    """

    stage: str
    summary: dict[str, Any] = field(default_factory=dict)
    frames: dict[str, list[Any]] = field(default_factory=dict)
    scalars: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    passed: bool = True

    #: Artefact names, frozen. ``phase_green_red_video_export.py`` writes a
    #: methods paragraph naming these files, so they are prose in a paper as
    #: much as they are filenames.
    frames_name: str = "registration_shifts_and_qc"
    summary_name: str = "registration_summary"

    def __len__(self) -> int:
        first = next(iter(self.frames.values()), [])
        return len(first)

    def store(self, source, params: Mapping[str, Any], *, output_dir,
              method_version: str = "", upstream: Iterable[str] = (),
              extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Write both tables as keyed tier-A artefacts."""
        from . import store as artefacts

        written = {}
        if self.frames:
            written["frames"] = artefacts.put(
                self.stage, source, params, kind="table", value=self.frames,
                name=self.frames_name, output_dir=output_dir,
                method_version=method_version, upstream=upstream, extra=extra)
        if self.summary:
            written["summary"] = artefacts.put(
                f"{self.stage}_summary", source, params, kind="table",
                value=table_from_rows([self.summary]), name=self.summary_name,
                output_dir=output_dir, method_version=method_version,
                upstream=upstream, extra=extra)
        return written

    def as_dict(self) -> dict[str, Any]:
        return {"stage": self.stage, "frames": len(self), "passed": self.passed,
                "scalars": dict(self.scalars), "notes": list(self.notes)}
