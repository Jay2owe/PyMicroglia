"""Declared comparisons over pooled tables; statistics belong to Workbench."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = ["TESTS", "PAIRED_TESTS", "SINGLE_GROUP_TESTS", "CORRECTIONS",
           "STATISTICS_FILE", "COLUMNS", "CONTRAST_UNITS", "CONTRAST_AGGREGATES",
           "ContrastSpec", "parse_contrasts", "status", "contrasts"]

#: Where the statistics land inside a run. The statistics are the one table
#: that is about the run as a whole, so it sits beside the pooled tables.
STATISTICS_FILE = "statistics"

#: The tests a contrast may name. ``groups`` is how many groups the test
#: compares (``None`` for any number), ``paired`` whether each unit is matched
#: with itself across the two groups, ``effect_kind`` what the effect column
#: reports. The arithmetic is Circadian Workbench's and is not here.
TESTS: dict[str, dict[str, Any]] = {
    "mannwhitney": {"groups": 2, "paired": False,
                    "effect_kind": "median_difference", "label": "Mann-Whitney U"},
    "kruskal": {"groups": None, "paired": False,
                "effect_kind": "epsilon_squared", "label": "Kruskal-Wallis H"},
    "welch_t": {"groups": 2, "paired": False,
                "effect_kind": "hedges_g", "label": "Welch's t"},
    "anova": {"groups": None, "paired": False,
              "effect_kind": "eta_squared", "label": "one-way ANOVA"},
    "wilcoxon": {"groups": 2, "paired": True,
                 "effect_kind": "median_paired_difference",
                 "label": "Wilcoxon signed-rank"},
    "paired_t": {"groups": 2, "paired": True,
                 "effect_kind": "hedges_g_paired", "label": "paired t"},
    "signed_rank": {"groups": 1, "paired": False,
                    "effect_kind": "median_against_zero",
                    "label": "Wilcoxon signed-rank against zero"},
}

PAIRED_TESTS = tuple(name for name, spec in TESTS.items() if spec["paired"])

#: Tests that take one group and compare it against zero rather than against
#: another group.
SINGLE_GROUP_TESTS = tuple(name for name, spec in TESTS.items()
                           if spec["groups"] == 1)

#: The multiple-testing corrections a family may name. Applied within a
#: declared family and never across the run.
CORRECTIONS: tuple[str, ...] = ("benjamini_hochberg", "bonferroni", "sidak", "none")

#: One row per contrast per metric, tested or refused, in the order Motion
#: writes ``statistics.csv``.
COLUMNS: tuple[str, ...] = (
    "contrast", "family", "table", "window", "metric", "group_by", "group_a",
    "group_b", "unit", "aggregate", "test", "test_label", "n_a", "n_b",
    "statistic", "p_value", "effect_kind", "effect", "effect_ci_low",
    "effect_ci_high", "correction", "alpha", "p_corrected", "significant", "note",
)

#: The dotted target the tests and corrections will come from: the one module
#: allowed to import Circadian Workbench, which stage 05 of the port writes.
#: Read at call time so a later stage changes it in one place.
STATISTICS_TARGET = "pymicroglia.workbench:group_contrasts"



#: A contrast name is a plain lower-case identifier, like a window name.
_NAME = re.compile(r"[a-z][a-z0-9_]*")


#: The units of replication this package will test at. No default anywhere.
CONTRAST_UNITS = ("cell", "movie", "subject")
#: How several rows are reduced to one value for a unit. Also no default.
CONTRAST_AGGREGATES = ("median", "mean")


@dataclass(frozen=True)
class ContrastSpec:
    """One declared comparison.

    ``unit`` has no default on purpose. Eighty-three cells from one movie are
    not eighty-three independent samples. There is deliberately no way to say
    "test every metric".
    """

    name: str
    table: str
    metrics: tuple[str, ...]
    group_by: str
    groups: tuple[str, ...]
    unit: str
    test: str
    aggregate: str | None = None
    family: str = "default"
    window: str | None = None
    correction: str = "benjamini_hochberg"
    alpha: float = 0.05
    description: str = ""

    def as_dict(self) -> dict[str, Any]:
        out = dict(self.__dict__)
        out["metrics"] = list(self.metrics)
        out["groups"] = list(self.groups)
        return out

    @classmethod
    def from_dict(cls, raw_dict: Mapping[str, Any], tests: tuple[str, ...],
                  paired_tests: tuple[str, ...],
                  corrections: tuple[str, ...],
                  single_group_tests: tuple[str, ...],
                  metric_groups: dict) -> "ContrastSpec":
        name = str(raw_dict.get("name", "")).strip()
        if not _NAME.fullmatch(name):
            raise ValueError(
                f"contrast name {name!r} is not a plain lower-case identifier; "
                "it names a family of results in statistics.csv, so it must "
                "look like `reporter_by_genotype`"
            )

        def required(key: str) -> object:
            if key not in raw_dict or raw_dict[key] in (None, ""):
                raise ValueError(
                    f"contrast {name!r} does not say {key!r}, and this package "
                    "does not guess it"
                )
            return raw_dict[key]

        metrics = required("metrics")
        if not isinstance(metrics, (list, tuple)) or not metrics:
            raise ValueError(
                f"contrast {name!r}: metrics must be a non-empty list of column "
                "names. There is no way to test every metric at once - a "
                "correction across a thousand tests nobody meant to run "
                "destroys the power to find the twenty that were the point."
            )
        from .metric_groups import resolve_metrics

        metrics = list(resolve_metrics(
            metrics, metric_groups, where=f"contrast {name!r} metrics"))
        one_sample = str(raw_dict.get("test", "")).strip() in single_group_tests
        groups = required("groups")
        if not isinstance(groups, (list, tuple)) or not groups:
            raise ValueError(
                f"contrast {name!r}: groups must be a list of "
                f"{raw_dict.get('group_by', 'the grouping column')} values"
            )
        if one_sample and len(groups) != 1:
            raise ValueError(
                f"contrast {name!r}: {raw_dict['test']} compares one group "
                f"against zero, but {len(groups)} are named. Naming a second "
                "would read as a comparison this test does not make."
            )
        if not one_sample and len(groups) < 2:
            raise ValueError(
                f"contrast {name!r}: groups must be a list of at least two "
                f"values of {raw_dict.get('group_by', 'the grouping column')} "
                "to compare"
            )
        groups = tuple(str(group) for group in groups)
        if len(set(groups)) != len(groups):
            raise ValueError(f"contrast {name!r}: the same group is named twice")

        unit = str(required("unit"))
        if unit not in CONTRAST_UNITS:
            raise ValueError(
                f"contrast {name!r}: unit={unit!r} is not a unit of replication "
                f"this package knows; it accepts {' or '.join(CONTRAST_UNITS)}"
            )
        test = str(required("test"))
        if test not in tests:
            raise ValueError(
                f"contrast {name!r}: test={test!r} is not one this package "
                f"offers; it accepts {', '.join(tests)}"
            )
        if test in paired_tests and len(groups) != 2:
            raise ValueError(
                f"contrast {name!r}: {test} is a paired test and compares two "
                f"groups, but {len(groups)} are named. A paired test matches "
                "each unit with itself in the other group, which only means "
                "something for two."
            )

        aggregate = raw_dict.get("aggregate")
        aggregate = None if aggregate in (None, "") else str(aggregate)
        if unit == "cell" and aggregate is not None:
            raise ValueError(
                f"contrast {name!r} is at unit={unit!r}, where each row is "
                "already one unit and there is nothing to aggregate; remove "
                "aggregate, which would otherwise imply there was."
            )
        if unit != "cell":
            if aggregate is None:
                raise ValueError(
                    f"contrast {name!r} is at unit={unit!r}, so several cells "
                    "are reduced to one value before testing, and this package "
                    "does not choose the statistic for you; set aggregate to "
                    f"{' or '.join(CONTRAST_AGGREGATES)}"
                )
            if aggregate not in CONTRAST_AGGREGATES:
                raise ValueError(
                    f"contrast {name!r}: aggregate={aggregate!r} is not one "
                    f"this package knows; it accepts "
                    f"{' or '.join(CONTRAST_AGGREGATES)}"
                )

        correction = str(raw_dict.get("correction", "benjamini_hochberg"))
        if correction not in corrections:
            raise ValueError(
                f"contrast {name!r}: correction={correction!r} is not one this "
                f"package offers; it accepts {', '.join(corrections)}"
            )
        alpha = float(raw_dict.get("alpha", 0.05))
        if not 0.0 < alpha < 1.0:
            raise ValueError(
                f"contrast {name!r}: alpha={alpha} is not between 0 and 1")

        window = raw_dict.get("window")
        return cls(
            name=name,
            table=str(required("table")),
            metrics=tuple(str(metric) for metric in metrics),
            group_by=str(required("group_by")),
            groups=groups,
            unit=unit,
            test=test,
            aggregate=aggregate,
            family=str(raw_dict.get("family", "default")),
            window=None if window in (None, "") else str(window),
            correction=correction,
            alpha=alpha,
            description=str(raw_dict.get("description", "")),
        )



def parse_contrasts(entries: object, groups: dict) -> list[ContrastSpec]:
    """Parse the ``contrasts`` block, refusing a repeat and a split family.

    A family is the set of results one correction is applied across, so two
    contrasts in one family that disagree about the correction or about alpha
    are refused.
    """
    if not entries:
        return []
    if isinstance(entries, (str, bytes)) or not isinstance(entries, Sequence):
        raise TypeError(
            f"contrasts must be a list of contrast blocks, not "
            f"{type(entries).__name__}"
        )
    parsed = [entry if isinstance(entry, ContrastSpec) else
              ContrastSpec.from_dict(entry, tuple(TESTS), tuple(PAIRED_TESTS),
                                     tuple(CORRECTIONS), tuple(SINGLE_GROUP_TESTS),
                                     groups)
              for entry in entries]
    seen: set[str] = set()
    for contrast in parsed:
        if contrast.name in seen:
            raise ValueError(f"contrast {contrast.name!r} is declared twice")
        seen.add(contrast.name)

    families: dict[str, ContrastSpec] = {}
    for contrast in parsed:
        first = families.setdefault(contrast.family, contrast)
        for setting in ("correction", "alpha"):
            if getattr(first, setting) != getattr(contrast, setting):
                raise ValueError(
                    f"contrasts {first.name!r} and {contrast.name!r} are both "
                    f"in family {contrast.family!r} but declare "
                    f"{setting}={getattr(first, setting)!r} and "
                    f"{setting}={getattr(contrast, setting)!r}. A family is the "
                    "set of results one correction is applied across, so it has "
                    "one correction and one alpha."
                )
    return parsed



def _importable(module: str) -> bool:
    import importlib.util
    import sys

    if module in sys.modules:
        return True
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def status() -> tuple[str, str]:
    """``("ready", "")`` once the Workbench importer resolves, else pending."""
    module, _, attribute = str(STATISTICS_TARGET).partition(":")
    if not module or not attribute:
        return "pending", f"{STATISTICS_TARGET!r} is not a 'module:attribute' target."
    if not _importable(module):
        return "pending", (
            f"{STATISTICS_TARGET} does not resolve: the tests and corrections "
            "come from Circadian Workbench through the one importer module the "
            "rhythm stage of the port adds, and it has not landed.")
    return "ready", ""


def contrasts(run_dir, *, contrasts: Sequence[Mapping[str, Any]] = (),
              metric_groups: Mapping[str, Any] | None = None,
              claim: str = "") -> dict[str, Any]:
    """Test declared comparisons through Workbench and record the pooled table."""
    import pandas as pd
    from .. import store, workbench
    from .run import read_manifest, write_manifest, write_table, manifest_path

    root = Path(run_dir)
    manifest = read_manifest(root)
    settings = manifest.get("settings", {})
    groups = metric_groups if metric_groups is not None else settings.get("metric_groups", {})
    specs = parse_contrasts(contrasts or settings.get("contrasts", []), groups)
    tables, missing, sources = {}, {}, []
    for spec in specs:
        path = root / "pooled" / (spec.table + ".csv")
        tables[spec.table] = pd.read_csv(path) if path.is_file() else None
        if path.is_file():
            sources.append(path)
        missing[spec.table] = (manifest.get("pooled") or {}).get("tables", {}).get(
            spec.table, {}).get("columns_missing", {})
    result = workbench.group_contrasts.group_contrasts(
        tables, specs, columns_missing=missing)
    source = store.collection(sources or [manifest_path(root)],
                              path=str(root / "pooled" / "statistics.csv"))
    record = write_table(result, name="statistics", folder=root / "pooled",
                         source=source, params={"contrasts": [c.as_dict() for c in specs],
                         "claim": claim, "workbench_version": workbench.WORKBENCH_VERSION})
    manifest["statistics"] = record
    write_manifest(root, manifest)
    return record
