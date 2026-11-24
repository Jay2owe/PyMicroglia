"""Tracking tower: approved defaults, configurable layout, measured paths and the declared figure."""
import json
from dataclasses import asdict

import numpy as np
import pytest
from test_cell_review_data import recording  # noqa: F401  (fixture)

from pymicroglia.figure_tables import tracking_tower as tt
from pymicroglia.figure_tables.saved import Tables
from pymicroglia.measure.cell_signal import analyse as signal


def tower_inputs(recording):
    _, _, raw, labels = recording
    hours = np.arange(len(raw)) * .25
    table = signal(raw, labels, source_frame_offset=1, frame_interval_h=.25)
    return raw, labels, hours, table


def test_defaults_are_the_approved_layout_and_overrides_are_named():
    s = tt.TowerSettings()
    assert (s.snapshot_interval_h, s.plane_gap_ratio, s.shear, s.depth) == (6., .30, -.45, .24)
    assert (s.outline_width_pt, s.surface_opacity, s.plane_background_opacity) == (.4, .65, .75)
    assert s.outer_boundaries_only and s.include_final_frame and s.uniform_plane_spacing
    changed = tt.settings({"plane_gap_ratio": .5}, outline_width_pt=1.)
    assert changed.plane_gap_ratio == .5 and changed.outline_width_pt == 1. and changed.shear == -.45
    with pytest.raises(ValueError, match="unknown tower settings"):
        tt.settings(plane_gap=.5)
    assert tt.DISPLAY_PRESETS["cd68"]["gamma"] == .3 and tt.DISPLAY_PRESETS["rbm3"]["percentiles"] == (8., 99.5)


def test_line_tower_keeps_measurements_and_breaks_paths_at_missing_masks(recording):
    raw, labels, hours, table = tower_inputs(recording)
    before = labels.copy()
    data = tt.analyse(labels, raw.astype(float), display_range=(0., 192.), frame_hours=hours, measurements=table,
                      source_frame_offset=1, tower=tt.settings(snapshot_interval_h=.5))
    checks = data["checks"]
    assert checks["source_frames"][0] == 1 and checks["source_frames"][-1] == 5
    assert checks["identities"] == [19, 26]
    # Cell 19 is absent from labelled frame 3: two links, one internal gap; cell 26 has one observation.
    assert checks["consecutive_links"] == 2 and checks["internal_gap_breaks"] == 1
    assert np.array_equal(labels, before)
    centroids = data["table"].query("kind == 'centroid' and identity == 19 and mask_present")
    assert list(centroids.area_px) == [6., 6., 7., 6.]
    widths = [w for group in data["tracks"] for w in group["widths"]]
    assert all(1. <= w <= 7.2 for w in widths)
    assert [p["label"] for p in data["planes"]] == ["0 h", "0 h", "1 h"]   # 0, .5, 1 h rounded to whole hours
    precise = tt.analyse(labels, raw.astype(float), display_range=(0., 192.), frame_hours=hours, measurements=table,
                         source_frame_offset=1, tower=tt.settings(snapshot_interval_h=.5, time_label_decimals=1))
    assert [p["label"] for p in precise["planes"]] == ["0 h", "0.5 h", "1 h"]


def test_final_frame_is_optional(recording):
    raw, labels, hours, table = tower_inputs(recording)
    data = tt.analyse(labels, raw.astype(float), display_range=(0., 192.), frame_hours=hours, measurements=table,
                      source_frame_offset=1, tower=tt.settings(snapshot_interval_h=.4, include_final_frame=False))
    assert data["checks"]["source_frames"] == [1, 3, 4]   # 0, .4 and .8 h targets; no 1 h final frame


def test_perimeter_surfaces_never_exceed_their_opacity(recording):
    raw, labels, hours, table = tower_inputs(recording)
    data = tt.analyse(labels, raw.astype(float), display_range=(0., 192.), frame_hours=hours, measurements=table,
                      style="perimeter", source_frame_offset=1, tower=tt.settings(snapshot_interval_h=.5))
    assert data["surfaces"] and max(s["rgba"][..., 3].max() for s in data["surfaces"]) <= .650001
    assert "not measured membrane" in data["checks"]["interpretation"]
    with pytest.raises(ValueError, match="style"):
        tt.analyse(labels, raw, display_range=(0., 1.), frame_hours=hours, measurements=table, style="dots",
                   source_frame_offset=1)


def test_misaligned_inputs_are_refused(recording):
    raw, labels, hours, table = tower_inputs(recording)
    with pytest.raises(ValueError, match="align"):
        tt.analyse(labels, raw, display_range=(0., 1.), frame_hours=hours[:-1], measurements=table,
                   source_frame_offset=1)


def test_declared_figure_prepares_and_draws_both_views(recording, tmp_path):
    from matplotlib import pyplot as plt
    from pymicroglia.registry import CLAIM_TEMPLATES
    from pymicroglia.visualisation.figures import load
    from pymicroglia.visualisation.panels.tracking_tower import tower
    rp, lp, _, _ = recording
    run = tmp_path / "saved"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({"movies": [{"stem": "sample", "spec": {"source_frame_offset": 1},
        "summary": {"scale": {"minutes_per_frame": 15}},
        "provenance": {"inputs": {"raw": {"path": str(rp)}, "labels": {"path": str(lp)}}}}]}), encoding="utf-8")
    spec = load()["tracking_tower"]
    assert {view.key for view in spec.views} == {"line", "perimeter"} and "{source}" in CLAIM_TEMPLATES["tracking_tower"]
    options = {o.name: o.default for o in spec.options} | {"display_preset": "none",
                                                           "tower_settings": {"snapshot_interval_h": .5}}
    data, statistics = tt.prepare(Tables(run), options)
    assert statistics is None and set(data) == {"line", "perimeter"}
    for key in ("line", "perimeter"):
        figure = plt.figure()
        drawn = tower(figure, data[key])
        assert len(drawn.data) and drawn.extra["checks"]["settings"] == asdict(tt.settings(snapshot_interval_h=.5))
        plt.close(figure)


def test_emphasis_keeps_every_track_and_fades_the_others(recording):
    """Correction reviewer (Jamie 2026-11-17): the full tower with every line, followed cells emphasised."""
    raw, labels, hours, table = tower_inputs(recording)
    kw = dict(display_range=(0., 192.), frame_hours=hours, measurements=table, source_frame_offset=1,
              tower=tt.settings(snapshot_interval_h=.5))
    plain = tt.analyse(labels, raw.astype(float), **kw)
    same = tt.analyse(labels, raw.astype(float), emphasis={19}, **kw)   # only 19 has links
    faded = tt.analyse(labels, raw.astype(float), emphasis={26}, **kw)
    flat = lambda d, k: [v for g in d["tracks"] for v in g[k]]
    assert flat(same, "widths") == flat(plain, "widths") and flat(same, "colours") == flat(plain, "colours")
    assert len(flat(faded, "widths")) == len(flat(plain, "widths"))
    s = tt.TowerSettings()
    assert np.allclose(flat(faded, "widths"), [w * s.faded_width for w in flat(plain, "widths")])
    spread = lambda c: np.ptp(np.asarray(c, float))
    assert all(spread(f) < spread(p) for f, p in zip(flat(faded, "colours"), flat(plain, "colours")) if spread(p) > 0)
