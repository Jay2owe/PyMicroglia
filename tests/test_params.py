"""The parameter reader: on synthetic fixtures, then against the real scripts."""

from __future__ import annotations

import pytest

from conftest import PROTOCOL_FILES, WITH_METHOD_VERSION
from pymicroglia import params


# ── the reader itself, no real protocol needed ───────────────────────────────
def test_reads_a_python_block(fixtures):
    block = params.harvest(fixtures / "fixture_engine.py")
    assert block is not None
    assert block.language == "python"
    assert block.method_version == "2026-08-19-fixture"
    assert block.constants == (
        "DEFAULT_SERIES",
        "DEFAULT_THRESHOLD_SIGMA",
        "DEFAULT_MASK_GROWTH_PX",
        "DEFAULT_WRITE_MASK",
        "DEFAULT_VIDEO_LUT",
        "DEFAULT_UNDERLAY_GREY",
        "DEFAULT_DISPLAY_PERCENTILES",
    )


def test_names_drop_the_default_prefix(fixtures):
    block = params.harvest(fixtures / "fixture_engine.py")
    assert [doc.name for doc in block.params][:3] == [
        "series", "threshold_sigma", "mask_growth_px",
    ]


def test_types_and_defaults_come_from_the_literal(fixtures):
    block = params.harvest(fixtures / "fixture_engine.py")
    by_name = {doc.name: doc for doc in block.params}
    assert (by_name["series"].type, by_name["series"].default) == ("int", 0)
    assert (by_name["threshold_sigma"].type, by_name["threshold_sigma"].default) == ("float", 12.0)
    assert (by_name["write_mask"].type, by_name["write_mask"].default) == ("bool", True)
    assert by_name["display_percentiles"].type == "tuple"
    assert by_name["display_percentiles"].default == (0.5, 99.5)


def test_grey_levels_stay_strings(fixtures):
    """trace_panel_figure holds greys as levels like "0.72", not hex.

    Stage 09 maps them onto the kit's GREY_LEVEL. Converting here would lose the
    fact that they were levels.
    """
    block = params.harvest(fixtures / "fixture_engine.py")
    grey = {doc.name: doc for doc in block.params}["underlay_grey"]
    assert (grey.type, grey.default) == ("str", "0.72")


def test_comment_split_is_quote_aware(fixtures):
    """A hex default must not be truncated at its own '#'."""
    block = params.harvest(fixtures / "fixture_engine.py")
    lut = {doc.name: doc for doc in block.params}["video_lut"]
    assert lut.default == "#a340d1"
    assert lut.description.startswith("A hex value")


def test_multi_line_description_is_joined_including_caution(fixtures):
    block = params.harvest(fixtures / "fixture_engine.py")
    sigma = {doc.name: doc for doc in block.params}["threshold_sigma"]
    assert sigma.description == (
        "A pixel is a spike when it exceeds the neighbour maximum by this many "
        "robust temporal noise sigmas. CAUTION: lowering this starts replacing "
        "real bright transients."
    )


def test_description_above_the_assignment_is_found(fixtures):
    block = params.harvest(fixtures / "fixture_engine.py")
    grey = {doc.name: doc for doc in block.params}["underlay_grey"]
    assert grey.description == (
        "The description for this one sits above it instead of beside it."
    )


@pytest.mark.parametrize(
    "constant, unit",
    [
        ("DEFAULT_MASK_GROWTH_PX", "px"),
        ("DEFAULT_THRESHOLD_SIGMA", "sigma"),
        ("DEFAULT_BLACK_POINT_PCT", "%"),
        ("DEFAULT_FRAME_INTERVAL_H", "h"),
        ("DEFAULT_FRAME_INTERVAL_SECONDS", "s"),
        ("DEFAULT_SERIES", "-"),
    ],
)
def test_units_are_inferred_from_the_name(constant, unit):
    assert params.infer_units(constant) == unit


def test_reads_an_imagej_macro(fixtures):
    block = params.harvest(fixtures / "fixture_macro.ijm")
    assert block.language == "ijm"
    assert "DEFAULT_PYTHON_ENGINE" in block.constants
    engine = block.by_constant("DEFAULT_PYTHON_ENGINE")
    assert engine.default == "fixture_engine.py"
    assert engine.description == "Sibling engine to call."


def test_reads_a_powershell_runner(fixtures):
    block = params.harvest(fixtures / "fixture_runner.ps1")
    assert block.language == "powershell"
    assert block.constants == ("OutputRoot", "Overwrite")
    assert block.by_constant("Overwrite").default is False


def test_no_block_returns_none(fixtures):
    assert params.harvest(fixtures / "fixture_noblock.py") is None


def test_harvest_many_dedupes_first_wins(fixtures):
    """A protocol is often an engine plus a wrapper that repeats a parameter."""
    merged = params.harvest_many([
        fixtures / "fixture_engine.py",
        fixtures / "fixture_macro.ijm",
        fixtures / "fixture_runner.ps1",
    ])
    assert merged.constants.count("DEFAULT_THRESHOLD_SIGMA") == 1
    # The Python engine came first, so its value is the one kept.
    assert merged.by_constant("DEFAULT_THRESHOLD_SIGMA").default == 12.0
    assert len(merged) == 7 + 2 + 2   # engine + macro's two new + runner's two
    assert merged.method_version == "2026-08-19-fixture"


def test_nothing_is_imported_or_executed(fixtures, monkeypatch):
    """Reading a block must not import the script it reads.

    The real engines import numpy, tifffile, scipy, scikit-image and matplotlib
    at module scope. A reader that imported them would cost seconds per call and
    would fail entirely when an optional dependency is missing.
    """
    import sys

    class Refuse:
        def find_module(self, name, path=None):  # pragma: no cover - py<3.12 shim
            return self.find_spec(name, path)

        def find_spec(self, name, path=None, target=None):
            if name.startswith("fixture_") or name in {"numpy", "tifffile", "scipy"}:
                raise AssertionError(f"the reader imported {name!r}")
            return None

    monkeypatch.setattr(sys, "meta_path", [Refuse(), *sys.meta_path])
    for name in [n for n in sys.modules if n.startswith("fixture_")]:
        monkeypatch.delitem(sys.modules, name, raising=False)

    block = params.harvest(fixtures / "fixture_engine.py")
    assert len(block) == 7


# ── verification against the real scripts; skipped when they are absent ──────
# These confirm the reader copes with the real files. They deliberately do NOT
# assert exact parameter counts against Protocols/INDEX.md: the catalogue this
# package ships is its own, free to diverge, and pinning numbers here would turn
# every edit to a source script into a failing test. The fixture tests above are
# what lock the reader's behaviour down.
@pytest.mark.parametrize("protocol", sorted(PROTOCOL_FILES))
def test_every_protocol_reads(protocols, protocol):
    _, relatives = PROTOCOL_FILES[protocol]
    block = params.harvest_many(protocols / rel for rel in relatives)
    assert len(block) > 0, f"{protocol}: no parameters read"
    for constant, doc in zip(block.constants, block.params):
        assert doc.name and doc.type, f"{protocol}.{constant} came back malformed"


def test_caution_survives_from_the_real_engine(protocols):
    """The one setting the engine warns about, read out of the live block.

    ``seed_z`` replaced ``threshold_sigma`` when the method was rewritten on
    2026-08-20. The sentence is what matters: an engine that says a setting
    decides what counts as data is telling a reader something no signature can.
    """
    block = params.harvest(protocols / "Analysis/microglia_cosmic_ray_removal.py")
    cut = block.by_constant("DEFAULT_SEED_Z")
    assert cut.name == "seed_z"
    assert cut.default == 12.0
    assert "CAUTION" in cut.description
    assert cut.description.endswith(
        "it is the setting that decides what counts as data.")


@pytest.mark.parametrize("protocol", sorted(PROTOCOL_FILES))
def test_method_version_is_read_or_empty(protocols, protocol):
    _, relatives = PROTOCOL_FILES[protocol]
    block = params.harvest_many(protocols / rel for rel in relatives)
    if protocol in WITH_METHOD_VERSION:
        assert block.method_version, f"{protocol} declares METHOD_VERSION but none was read"
    else:
        assert block.method_version == ""


def test_almost_everything_is_documented(protocols):
    """The reader should find prose for essentially every real parameter.

    A threshold rather than an exact list: a handful of parameters share a
    sentence with a neighbour in the source, and chasing that is not worth a
    failing test.
    """
    total = documented = 0
    for _, relatives in PROTOCOL_FILES.values():
        block = params.harvest_many(protocols / rel for rel in relatives)
        total += len(block)
        documented += sum(1 for doc in block.params if doc.description)
    assert documented / total > 0.98, f"only {documented}/{total} parameters got prose"
