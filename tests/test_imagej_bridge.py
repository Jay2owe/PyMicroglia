"""The one door to Fiji, and the promise that it being shut costs nothing.

Every test here runs on a machine with no Fiji, because that is the machine
this package has to work on: a server, a laptop on a train, a fresh checkout.
The bridge is for one thing — a person tracing an outline — and if it is
unreachable the only casualty must be that one step.

No test opens a window. Fiji is stood in for by a fake that answers the three
calls the bridge makes, so the interactive path is exercised end to end without
anything modal appearing on somebody's screen.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pymicroglia import imagej, roi
from tests_support import two_channel_stack


@pytest.fixture(autouse=True)
def forget_the_bridge():
    """Each test decides for itself whether Fiji is there."""
    imagej.reset_cache()
    yield
    imagej.reset_cache()


@pytest.fixture
def local_store(tmp_path, monkeypatch):
    monkeypatch.setenv("PYMICROGLIA_STORE", str(tmp_path / "cache"))
    monkeypatch.setenv("PYMICROGLIA_DECISIONS", str(tmp_path / "decisions"))
    return tmp_path


class FakeFiji:
    """Enough of ImageJAI to drive the bridge, and nothing that draws.

    ``outlines`` is what the next read finds. Setting it to ``[]`` is a person
    who has not drawn anything yet, which is the state the poll loop spends
    almost all of its time in.
    """

    HOST = "localhost"
    PORT = 7746

    def __init__(self, outlines=(), answering: bool = True):
        self.outlines = list(outlines)
        self.answering = answering
        self.opened: list[str] = []
        self.scripts: list[str] = []

    def ping(self):
        return {"ok": True} if self.answering else {"ok": False}

    def open_image(self, path, series=None, timeout=120):
        self.opened.append(str(path))
        return {"ok": True}

    def run_jython(self, code, timeout=180):
        self.scripts.append(code)
        # The read snippet is handed the file it must write; writing it here is
        # what makes this a stand-in for Fiji rather than a stub of the bridge.
        # Only that snippet writes: a bare OUTPUT assignment in some other
        # script is not a request for a file.
        for line in (code.splitlines() if "json.dump" in code else ()):
            if line.startswith("OUTPUT = "):
                target = Path(json.loads(line.split(" = ", 1)[1].replace("'", '"')))
                target.write_text(json.dumps({"rois": self.outlines}),
                                  encoding="utf-8")
        return {"ok": True}


def attach(monkeypatch, fake: FakeFiji) -> FakeFiji:
    monkeypatch.setattr(imagej, "bridge", lambda: fake)
    monkeypatch.setattr(imagej, "agent_dir", lambda: Path("nowhere"))
    return fake


def an_outline(kind: int = 7) -> dict:
    return {"name": "SCN", "type": kind,
            "x": [10.0, 40.0, 40.0, 10.0], "y": [10.0, 10.0, 40.0, 40.0]}


# ------------------------------------------------------------- nothing present
def test_a_missing_bridge_is_a_clean_answer_not_a_crash(monkeypatch):
    monkeypatch.setattr(imagej, "agent_dir", lambda: None)
    imagej.reset_cache()

    assert imagej.bridge() is None
    assert imagej.available() is False

    report = imagej.status()
    assert report["ok"] is False
    assert report["agent"] is None
    assert imagej.AGENT_ENV in report["reason"]


def test_an_unreachable_fiji_says_what_to_do(monkeypatch):
    attach(monkeypatch, FakeFiji(answering=False))
    report = imagej.status()

    assert report["ok"] is False
    assert "TCP command server" in report["reason"]
    assert report["port"] == 7746


def test_the_manual_step_refuses_with_the_message_the_plan_names(monkeypatch,
                                                                 local_store):
    monkeypatch.setattr(imagej, "agent_dir", lambda: None)
    imagej.reset_cache()
    source = two_channel_stack(local_store, frames=3, height=32, width=32)

    with pytest.raises(imagej.ImageJUnavailable) as raised:
        imagej.draw_roi(source)

    message = str(raised.value)
    assert "only needed for drawing an ROI by hand" in message
    assert "every analysis step runs without it" in message
    assert "pymicroglia doctor" in message


def test_every_analysis_action_still_runs_with_no_fiji(monkeypatch, local_store):
    """Gate 3. The scientific half of the package does not know Fiji exists."""
    monkeypatch.setattr(imagej, "agent_dir", lambda: None)
    imagej.reset_cache()

    from pymicroglia import doctor, run_action

    source = two_channel_stack(local_store, frames=5, height=32, width=32)
    results = local_store / "results"
    run_action("remove_cosmic_rays", source=str(source), signal_channel=1,
               output_dir=str(results), output_roots=[results])

    assert list(results.glob("*.tif"))
    report = doctor()
    assert report["ok"] is True and report["hand_roi_available"] is False
    assert report["imagej"]["ok"] is False


def test_doctor_says_where_it_looked():
    """An absent bridge is checkable rather than silent."""
    from pymicroglia import doctor

    report = doctor()
    assert "imagej" in report and "hand_roi_available" in report
    assert set(report["imagej"]) >= {"ok", "agent", "needed_for"}
    assert report["ok"] is True, "a missing Fiji is never a complaint"


# ----------------------------------------------------------- sending it Jython
def test_a_windows_path_reaches_jython_as_a_literal(monkeypatch):
    r"""``X:\Unicode`` interpolated into a script body is a broken escape.

    Arguments are emitted as ``repr`` assignments at the top of the snippet for
    the same reason the equivalent-script builder does it.
    """
    fake = attach(monkeypatch, FakeFiji())
    awkward = r"X:\Unicode Folder\temp\MCG 04.tif"
    imagej.run_jython("pass", OUTPUT=awkward)

    header = fake.scripts[-1].splitlines()[0]
    assert header == f"OUTPUT = {awkward!r}"
    namespace: dict = {}
    exec(header, namespace)  # noqa: S102 - proving the literal round-trips
    assert namespace["OUTPUT"] == awkward


def test_a_name_that_is_not_a_variable_is_refused(monkeypatch):
    attach(monkeypatch, FakeFiji())
    with pytest.raises(ValueError, match="usable Jython variable name"):
        imagej.run_jython("pass", **{"not a name": 1})


def test_the_read_snippet_names_no_macro_and_no_protocol_folder():
    """The only Jython this package sends is written in this package."""
    for snippet in (imagej.READ_OUTLINES, imagej.ANNOUNCE):
        assert ".ijm" not in snippet
        assert "Protocols" not in snippet


def test_reading_outlines_survives_a_snippet_that_wrote_nothing(monkeypatch):
    attach(monkeypatch, FakeFiji(outlines=[]))
    assert imagej.read_outlines() == []


# --------------------------------------------------------------- what came back
def test_a_freehand_outline_comes_back_as_a_polygon():
    polygon = imagej._as_polygon(an_outline(kind=7))
    assert polygon.name == "SCN" and len(polygon) == 4


def test_a_rectangle_is_refused_rather_than_rasterised():
    """The same rule the stored-file reader applies: a rectangle is a handful
    of parameters, and turning one into a mask means choosing a rasterisation."""
    with pytest.raises(ValueError, match="no vertices to read"):
        imagej._as_polygon({"type": 1, "x": [], "y": []})


def test_two_points_are_not_an_outline():
    with pytest.raises(ValueError, match="at least three vertices"):
        imagej._as_polygon({"type": 0, "x": [1.0, 2.0], "y": [1.0, 2.0]})


# ------------------------------------------------------------- drawing one, once
def test_an_outline_drawn_once_is_never_asked_for_again(monkeypatch, local_store):
    """Gate 4. Keyed on the source alone, so it survives a parameter change and
    a METHOD_VERSION bump — a person answered this, and a new engine version is
    not a reason to ask them again."""
    fake = attach(monkeypatch, FakeFiji(outlines=[an_outline()]))
    source = two_channel_stack(local_store, frames=4, height=64, width=64)

    first = imagej.draw_roi(source, frame=0, channel=1, poll_s=0.01)
    assert len(first) == 4
    assert fake.opened, "the frame should have been opened in Fiji"

    fake.opened.clear()
    fake.outlines = []                       # nothing drawn this time
    again = imagej.draw_roi(source, frame=0, channel=1, poll_s=0.01)

    assert list(again.x) == list(first.x)
    assert not fake.opened, "a stored answer must not reopen Fiji"


def test_a_timeout_records_nothing_at_all(monkeypatch, local_store):
    """A person on a large stack takes minutes. Timing out must lose nothing —
    a half-written decision would be worse than no decision."""
    from pymicroglia import store

    attach(monkeypatch, FakeFiji(outlines=[]))
    source = two_channel_stack(local_store, frames=4, height=64, width=64)

    with pytest.raises(imagej.ImageJTimeout) as raised:
        imagej.draw_roi(source, timeout_s=0.05, poll_s=0.01)

    assert "nothing was recorded" in str(raised.value)
    assert store.decision("hand_roi_hand", source) is None


def test_only_one_frame_is_handed_over(monkeypatch, local_store):
    """Not the stack. Opening ten gigabytes so somebody can trace one outline
    would cost minutes for nothing."""
    import tifffile

    fake = attach(monkeypatch, FakeFiji(outlines=[an_outline()]))
    source = two_channel_stack(local_store, frames=6, height=48, width=48)
    imagej.draw_roi(source, frame=2, channel=1, poll_s=0.01)

    handed = Path(fake.opened[-1])
    assert handed.is_file()
    assert tifffile.imread(handed).shape == (48, 48)
    assert "_t2_c1" in handed.name


# ------------------------------------------------------------ the SCN decision
def test_the_scn_tracing_lands_under_the_decision_the_pipeline_reads(
        monkeypatch, local_store):
    """One question, one answer, one key. Two keys for the same tracing would
    mean an unattended run asking again for something already answered."""
    attach(monkeypatch, FakeFiji(outlines=[an_outline()]))
    source = two_channel_stack(local_store, frames=4, height=64, width=64)
    results = local_store / "results"

    drawn = roi.draw_scn_roi(source, frame=1, channel=1,
                             output_dir=results, timeout_s=1.0)
    assert drawn["asked"] is True

    settled = roi.scn_roi(source)
    assert settled["source"] == "decision"
    assert settled["value"]["x"] == [10.0, 40.0, 40.0, 10.0]

    again = roi.draw_scn_roi(source, frame=1, channel=1)
    assert again["asked"] is False, "the automatic path must not ask twice"


def test_the_written_roi_records_the_frame_it_was_drawn_on(monkeypatch,
                                                           local_store):
    """A pipeline reads it back to pick which registration shift to apply.

    Left at zero — which is what this package wrote before — every outline
    claims frame 1, and one drawn on frame 60 comes back a few pixels off the
    structure it was traced around.
    """
    attach(monkeypatch, FakeFiji(outlines=[an_outline()]))
    source = two_channel_stack(local_store, frames=4, height=64, width=64)
    results = local_store / "results"

    drawn = roi.draw_scn_roi(source, frame=1, channel=1, output_dir=results)
    stored = roi.read_roi(drawn["roi_file"])

    assert stored.position["frame"] == 2, "ImageJ counts frames from one"
    assert stored.position["channel"] == 2
    assert list(stored.x) == [10.0, 40.0, 40.0, 10.0]


def test_a_roi_with_no_position_still_reads(local_store):
    """Every ROI written before this stage has zeros there, and must not break."""
    polygon = roi.Polygon(name="plain", x=[1.0, 9.0, 9.0], y=[1.0, 1.0, 9.0])
    path = roi.write_roi(local_store / "plain.roi", polygon)
    back = roi.read_roi(path)

    assert back.position["frame"] == 0
    assert list(back.x) == [1.0, 9.0, 9.0]


# ------------------------------------------------------------- where it started
def test_a_run_through_the_bridge_is_recorded_as_having_come_from_fiji(
        local_store, monkeypatch):
    """Gate 5. One field differs; everything else must be identical, so a run
    started at the bench and a run started in a terminal are comparable without
    anybody having to know which was which."""
    from pymicroglia._optional import kit

    if kit() is None:  # pragma: no cover
        pytest.skip("analysis_kit is not installed in this interpreter")

    from pymicroglia import run_recorded

    source = two_channel_stack(local_store, frames=5, height=32, width=32)
    shared = dict(source=str(source), signal_channel=1)

    from_python = run_recorded(
        "remove_cosmic_rays", output_dir=str(local_store / "a"),
        output_roots=[local_store / "a"], **shared)["record"]
    from_fiji = imagej.run_action(
        "remove_cosmic_rays", output_dir=str(local_store / "b"),
        output_roots=[local_store / "b"], **shared)

    assert from_python["entry_path"] == "python"

    records = sorted((local_store / "b" / ".analysis-kit" / "records").glob("*.json"))
    recorded = json.loads(records[0].read_text(encoding="utf-8"))
    assert recorded["entry_path"] == "imagej"
    assert from_fiji is not None

    # Everything that is not the run itself, or where it wrote, must match.
    varies = {"run_id", "started", "duration_s", "entry_path", "params",
              "outputs", "artefacts", "digest", "script", "metrics"}
    for field in set(from_python) - varies:
        assert recorded[field] == from_python[field], field
