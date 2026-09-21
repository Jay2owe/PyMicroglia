"""Stand-in measurement modules and a synthetic movie for the measure chassis.

Stage 03 of the Motion port lands the chassis without a single science
module: ``pymicroglia.measure.modules`` is empty until stage 04. The chassis
still has to be tested -- the join, the folds, the roll-ups, the writer, the
pooling -- so these tests register a handful of small stand-ins through the
same decorators a real module uses and unregister them afterwards.

They are deliberately shaped like the real ones: one measurement that folds
into the cell-frame table and writes a per-frame file of its own, one that
folds into the per-cell roll-up, one that needs an extra channel, and one
derived module that reads what an earlier module wrote and copies a tracker
table through. ``movie()`` is Motion's ``test_column_declarations._movie``,
so the shapes the chassis is exercised over are the ones it was written for.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from pymicroglia.measure import (ChannelStack, Column, MeasurementContext,
                                 ObjectStack, Output, Scale, declare)

__all__ = ["STUB_NAMES", "register_stubs", "forget_stubs", "registered", "stubs",
           "movie", "channel", "objects", "side", "FIXTURE", "fixture_copy", "measured"]

#: Every stand-in, in registration order.
STUB_NAMES = ("stub_area", "stub_tracks", "stub_channel", "stub_regimes")


def _stub_area(context: MeasurementContext) -> dict:
    import pandas as pd

    scale = float(context.module_params("stub_area").get("scale", 1.0))
    rows = []
    for index in range(context.n_frames):
        frame = context.labels[index]
        for identity in context.identities:
            inside = frame == identity
            count = int(inside.sum())
            if not count:
                continue
            ys, xs = np.nonzero(inside)
            rows.append({"identity": identity, "frame_index": index,
                         "area_px": count * scale,
                         "centroid_x": float(xs.mean())})
    frames = pd.DataFrame({
        "frame_index": np.arange(context.n_frames),
        "labelled_px": [int((context.labels[i] > 0).sum())
                        for i in range(context.n_frames)],
    })
    return {"stub_area": pd.DataFrame(rows), "stub_frames": frames,
            "stub_stack": context.labels > 0}


def _stub_tracks(context: MeasurementContext) -> dict:
    import pandas as pd

    rows = []
    for identity in context.identities:
        centres = []
        for index in range(context.n_frames):
            ys, xs = np.nonzero(context.labels[index] == identity)
            if len(xs):
                centres.append((ys.mean(), xs.mean()))
        steps = (np.diff(np.asarray(centres), axis=0) if len(centres) > 1
                 else np.zeros((0, 2)))
        rows.append({"identity": identity,
                     "track_len_px": float(np.hypot(steps[:, 0], steps[:, 1]).sum())})
    return {"stub_tracks": pd.DataFrame(rows)}


def _stub_channel(context: MeasurementContext) -> dict:
    import pandas as pd

    rows = []
    for name, stack in context.channels.items():
        for index in range(context.n_frames):
            for identity in context.identities:
                inside = context.labels[index] == identity
                if not inside.any():
                    continue
                rows.append({"identity": identity, "frame_index": index,
                             "channel": name,
                             "channel_mean": float(np.nanmean(stack.values[index][inside]))})
    return {"stub_channels": pd.DataFrame(rows)}


def _stub_regimes(cell_frame, context: MeasurementContext) -> dict:
    import pandas as pd

    threshold = float(context.module_params("stub_regimes").get("threshold", 100.0))
    regimes = cell_frame[["identity", "frame_index"]].copy()
    regimes["state_number"] = (cell_frame["area_px"] > threshold).astype(int)
    copied = pd.DataFrame({"identity": context.identities,
                           "mechanism": ["stub"] * len(context.identities)})
    return {"stub_regimes": regimes, "history_stub_copy": copied}


def register_stubs() -> None:
    """Register every stand-in, exactly as a real module registers itself."""
    declare.measurement(
        "stub_area", "one row per cell per frame: how many pixels it covers",
        defaults={"scale": 1.0},
        produces=(Column("area_px", "Area", "px2"),
                  Column("centroid_x", "Centroid x", "px"),
                  Column("labelled_px", "Labelled pixels", "px2", "field")),
        writes=(Output("stub_area", grain=("identity", "frame_index"), fold=True),
                Output("stub_frames", grain=("frame_index",))),
    )(_stub_area)
    declare.measurement(
        "stub_tracks", "one row per cell: the length of its path",
        produces=(Column("track_len_px", "Track length", "px", "motility"),),
        writes=(Output("stub_tracks", grain=("identity",), fold=True),),
    )(_stub_tracks)
    declare.measurement(
        "stub_channel", "one row per cell per frame per channel: mean brightness",
        requires=("labels", "channels"),
        produces=(Column("channel", "Channel", "", "reporter"),
                  Column("channel_mean", "Channel mean", "a.u.", "reporter")),
        writes=(Output("stub_channels", grain=("identity", "frame_index", "channel")),),
    )(_stub_channel)
    declare.derived(
        "stub_regimes", "one row per cell per frame: a state from the area",
        needs_columns=("area_px",),
        defaults={"threshold": 100.0},
        produces=(Column("state_number", "State", "", "regime"),
                  Column("mechanism", "Mechanism", "", "tracker")),
        writes=(Output("stub_regimes", grain=("identity", "frame_index"), fold=True),
                Output("history_stub_copy", grain=(), origin="tracker", optional=True)),
    )(_stub_regimes)


def forget_stubs() -> None:
    for name in STUB_NAMES:
        declare.forget(name)


@contextmanager
def registered():
    register_stubs()
    try:
        yield
    finally:
        forget_stubs()


@pytest.fixture
def stubs():
    """The stand-ins registered for one test and gone afterwards."""
    with registered():
        yield STUB_NAMES


# ------------------------------------------------------------- the movie

def movie() -> MeasurementContext:
    """Three cells on a small field, long enough for a rhythm to be fitted.

    Motion's ``test_column_declarations._movie``, unchanged: forty-eight
    frames at 30 minutes, every optional stack supplied, one extra channel,
    one object set and one side table of each key.
    """
    n_frames, height, width = 48, 40, 40
    labels = np.zeros((n_frames, height, width), dtype=np.uint16)
    hours = np.arange(n_frames) / 2
    for frame in range(n_frames):
        drift = int(2 * np.sin(2 * np.pi * hours[frame] / 24))
        labels[frame, 8 + frame % 3:16 + frame % 3, 8 + drift:16 + drift] = 1
        labels[frame, 24:32, 22 + frame % 4:30 + frame % 4] = 2
        labels[frame, 4:9, 30:35] = 3

    noise = np.random.default_rng(0).random((n_frames, height, width))
    signal = 100 + 30 * np.sin(2 * np.pi * hours / 24)
    raw = (labels > 0) * signal[:, None, None] + noise * 5
    full = np.iinfo(np.uint16).max
    evidence = np.zeros((n_frames, 5, height, width), dtype=np.uint16)
    evidence[:, :, 9:15, 9:15] = full
    ramp = (np.arange(36).reshape(6, 6) % 10) + 1
    evidence[:, 4, 9:15, 9:15] = np.rint(ramp * full / 10).astype(np.uint16)
    flagged = np.zeros((n_frames, height, width), dtype=bool)
    flagged[:, 8:10, 8:10] = True

    return MeasurementContext(
        stem="declarations", labels=labels, raw=raw, scale=Scale(30.0),
        identities=[1, 2, 3], unclaimed=(labels == 0) & (noise > 0.95),
        evidence=evidence, inferred=flagged, unresolved=flagged, added=flagged,
        channels={"extra": channel(labels, noise, n_frames)},
        objects={"scenery": objects(n_frames, height, width)},
        side=side(n_frames),
    )


def objects(n_frames: int, height: int, width: int) -> ObjectStack:
    """Two reference shapes, one of which moves and one of which does not."""
    values = np.zeros((n_frames, height, width), dtype=np.int32)
    for frame in range(n_frames):
        values[frame, 18:26, 2:8] = 1
        drift = frame % 5
        values[frame, 0:4, 30 + drift:36 + drift] = 2
    return ObjectStack(name="scenery", values=values, static=False,
                       path="synthetic", description="two shapes, for the declarations")


def side(n_frames: int) -> dict:
    """One side table of each key, in the shape the loader hands them over."""
    import pandas as pd

    return {
        "acquisition": pd.DataFrame({
            "frame_index": np.arange(n_frames),
            "acquisition_focus": np.linspace(1.0, 0.5, n_frames),
            "acquisition_suspect": (np.arange(n_frames) == 11).astype(int),
        }),
        "genotype": pd.DataFrame({
            "identity": [1, 2, 3],
            "genotype_call": ["wt", "ko", "wt"],
        }),
    }


def channel(labels: np.ndarray, noise: np.ndarray, n_frames: int) -> ChannelStack:
    """A second imaging channel that fades, differs between cells and has holes."""
    fade = np.linspace(1.0, 0.55, n_frames)[:, None, None]
    values = (300.0 + 90.0 * (labels == 1) + 180.0 * (labels == 2) + 25.0 * noise) * fade
    values = values.astype(np.float32)
    values[:, 0:3, 0:3] = float(np.iinfo(np.uint16).max)
    values[:, :, -2:] = np.nan
    return ChannelStack(
        name="extra", values=values, source_dtype="uint16",
        saturation_value=float(np.iinfo(np.uint16).max),
        path="synthetic", description="a second channel, for the declarations",
    )


# ------------------------------------------------------- the fixture run

#: The stage-01 fixture: three tracked cells, 48 frames at 30 min, with every
#: extra input the configuration knows how to declare.
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "motion_parity"

#: The two blocks of the fixture's configuration that name stage-04 modules:
#: ``metric_groups`` lists columns only the real modules write, ``modules``
#: sets options on modules that are not registered until then. Both are
#: refused on the way in, by design, so the chassis run drops them.
STAGE_04_BLOCKS = ("metric_groups", "modules")


def fixture_copy(folder) -> Path:
    """The fixture's inputs and configuration copied beside each other.

    Copied rather than pointed at, so a run can never write into the package's
    own test data, and so the relative paths in the configuration resolve the
    way ``load_config`` resolves them: against the file.
    """
    import json
    import shutil

    target = Path(folder) / "parity"
    shutil.copytree(FIXTURE / "inputs", target / "inputs", dirs_exist_ok=True)
    data = json.loads((FIXTURE / "config.json").read_text(encoding="utf-8"))
    for block in STAGE_04_BLOCKS:
        data.pop(block, None)
    path = target / "config.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def measured(folder, *, run_label: str = "chassis", modules=STUB_NAMES, **overrides):
    """One chassis run over the fixture with the stand-ins registered.

    Returns the run folder, the manifest ``measure`` returned and the
    configuration it was built from. Callers set ``PYMICROGLIA_STORE`` first.
    """
    from pymicroglia.measure import load_config, measure

    config = load_config(fixture_copy(folder))
    settings = dict(
        output_dir=Path(folder) / "outputs", run_label=run_label,
        enabled_modules=list(modules), frame_interval_min=config.frame_interval_min,
        conditions=config.conditions, windows=[w.as_dict() for w in config.windows],
        contrasts=config.contrasts, claim="measured the fixture with stand-ins")
    settings.update(overrides)
    with registered():
        manifest = measure(config.movies, **settings)
    return Path(manifest["run"]["folder"]), manifest, config
