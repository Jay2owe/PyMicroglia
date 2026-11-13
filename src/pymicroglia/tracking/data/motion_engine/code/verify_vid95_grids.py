"""Check that all VID95 grid bundles use the requested reference defaults."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "analysis outputs/full_vid95_fresh_20260921"
STEMS = [f"95_{row}{col}" for row, cols in
         (("A", range(1, 7)), ("B", range(1, 7)), ("C", range(1, 3)))
         for col in cols]


def main() -> None:
    for stem in STEMS:
        bundle = RUN / "figures/all-cell-trace-grid" / stem.lower().replace("_", "-")
        settings_path = bundle / "data/der/settings.csv"
        if not settings_path.exists():
            settings_path = bundle / "data/der/der_settings.csv"
        settings = json.loads(pd.read_csv(settings_path).iloc[0].settings_json)
        data = pd.read_csv(bundle / "data/der/figure_data.csv")
        masters = list((bundle / "fig").glob("*.svg"))
        if len(masters) != 1 or not masters[0].is_file():
            raise ValueError(f"{stem}: expected one audited grid SVG")
        expected = {"metrics": ["signal_mean"], "trace_view": "detrended",
                    "trace_normalization": "minmax", "multiple_testing": "none",
                    "fit_method": "lomb", "significance_method": "lomb",
                    "detrend": "robust_linear", "hour_ticks": 12.0,
                    "period_testing": True}
        for key, value in expected.items():
            if settings.get(key) != value:
                raise ValueError(f"{stem}: {key}={settings.get(key)!r}, expected {value!r}")
        if not settings.get("period_display_style") or set(data.metric) != {"signal_mean"}:
            raise ValueError(f"{stem}: reference display or metric missing")
        if data.frame_index.max() > 98 or data.frame_index.min() < 0:
            raise ValueError(f"{stem}: grid extends beyond 99 retained frames")
        if not (bundle / "data/der/statistics.csv").is_file():
            raise FileNotFoundError(f"{stem}: period-test table missing")
        print(f"{stem}: {data.identity.nunique()} cells, reference defaults, {masters[0]}", flush=True)


if __name__ == "__main__":
    main()
