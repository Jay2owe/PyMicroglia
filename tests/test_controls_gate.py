"""No control, no rhythm result — and no way round it.

The gate this file tests is the project's most expensive lesson made
structural. In the reference dataset the structural channel showed a 22.8 h
sinusoid at Lomb-Scargle power 0.966, and the same rhythm was present off
tissue where there is no sample, in a second channel, and in image sharpness.
It was a daily focus cycle. A pipeline that reported the period without the
control would have been reporting the microscope.

So `test_rhythm` refuses a source with no control artefact, the refusal names
the finding, and there is no keyword that switches it off. The last of those is
what the test on the signature is for: the failure being guarded against is
somebody adding one later.
"""

from __future__ import annotations

import inspect
import sys

import numpy as np
import pytest

from pymicroglia import controls, rhythm, tracing


@pytest.fixture
def store_root(tmp_path, monkeypatch):
    monkeypatch.setenv("PYMICROGLIA_STORE", str(tmp_path / "cache"))
    monkeypatch.setenv("PYMICROGLIA_DECISIONS", str(tmp_path / "decisions"))
    return tmp_path / "cache"


@pytest.fixture
def rhythmic_trace():
    """Five days at half-hourly sampling, with a clean 24 h rhythm on it."""
    times = np.arange(0.0, 120.0, 0.5)
    rng = np.random.default_rng(4)
    values = (1000.0 + 120.0 * np.sin(2 * np.pi * times / 24.0)
              + rng.normal(0, 8, len(times)))
    return times, values


# ------------------------------------------------------------------ the gate
def test_a_rhythm_without_a_control_is_refused(tmp_path, store_root,
                                               rhythmic_trace):
    """Gate 5. The whole stage, in one assertion."""
    from tests_support import two_channel_stack

    times, values = rhythmic_trace
    source = two_channel_stack(tmp_path)

    with pytest.raises(rhythm.ControlMissing) as raised:
        rhythm.test_rhythm(source, traces=values[None, :], times_h=times)

    message = str(raised.value)
    assert "22.8 h" in message
    assert "off tissue" in message
    assert "focus cycle" in message
    assert "run_controls" in message


def test_there_is_no_way_to_skip_the_control():
    """A flag would be used, once, in a hurry — and the number would look real.

    Checked on the signature and the source, because the failure is somebody
    *adding* the escape rather than the escape existing today.
    """
    escapes = ("skip_control", "force", "no_control", "ignore_control",
               "allow_missing_control")

    # no public function in the module takes one
    for name in dir(rhythm):
        function = getattr(rhythm, name)
        if not callable(function) or name.startswith("__"):
            continue
        try:
            parameters = set(inspect.signature(function).parameters)
        except (TypeError, ValueError):
            continue
        offending = parameters & set(escapes)
        assert not offending, f"rhythm.{name} grew {sorted(offending)}"

    # and none appears as a keyword or annotation anywhere in the code. The
    # words themselves are in the prose — they have to be, because a reader
    # needs to know what is refused — so the scan is for the syntax.
    code = " ".join(line for line in inspect.getsource(rhythm).splitlines()
                    if not line.strip().startswith(("#", '"', "'")))
    for escape in escapes:
        for syntax in (f"{escape}=", f"{escape}:", f"{escape} ="):
            assert syntax not in code, f"rhythm.py uses {syntax!r}"


def test_with_a_control_present_the_result_carries_it(tmp_path, store_root,
                                                      rhythmic_trace):
    """Gate 6: the control travels attached, not beside.

    A figure or a run record cannot then show a period without also showing
    what the control measured — there is no arrangement of the data in which
    one is present and the other is not.

    What travels is the **readings**, not a verdict. Auto-Organotypic stopped
    issuing one on 2026-08-24 and said why: a slice that fills its well and
    glows puts its own light off tissue, in the other channel and in the
    sharpness at once, so all three legs fire on a recording that is entirely
    healthy. The tell is that a real rhythm tracks its own recording's period
    well by well while an instrument cycle is common-mode across the plate,
    and a per-recording pass or fail cannot see that. So this asserts the
    numbers arrive, and no longer that somebody has judged them.
    """
    from tests_support import two_channel_stack

    from pymicroglia import store

    times, values = rhythmic_trace
    source = two_channel_stack(tmp_path)

    store.put(controls.CONTROL_STAGE, source, {"synthetic": True},
              kind="scalars",
              value={"findings": [], "region_names": ["tissue"],
                     "rhythmic_off_tissue": False},
              name="instrumental_control", output_dir=tmp_path / "out",
              method_version=controls.METHOD_VERSION)

    result = rhythm.test_rhythm(source, traces=values[None, :], times_h=times,
                                labels=["cell_1"])

    assert result.control, "the result arrived with an empty control"
    assert result.control["region_names"] == ["tissue"]
    assert result.periods[0]["period_hours"] == pytest.approx(24.0, abs=0.5)

    # and it is in the serialised form too, which is what a record writes
    payload = result.as_dict()
    assert payload["instrumental_control"] == result.control
    assert set(payload) >= {"periods", "cosinor", "instrumental_control"}


def test_a_control_that_found_something_still_returns_but_carries_it(
        tmp_path, store_root, rhythmic_trace):
    """A finding is an observation, not an error and not a refusal.

    The point is that nobody can read the period without reading what the
    control saw — not that the numbers are withheld. Withholding them would
    push somebody towards computing the period another way.

    Note the shape of the assertion: the finding is carried through verbatim,
    and nothing here decides what it means. Deciding is the reader's, with the
    rest of the plate in front of them.
    """
    from tests_support import two_channel_stack

    from pymicroglia import store

    times, values = rhythmic_trace
    source = two_channel_stack(tmp_path)
    store.put(controls.CONTROL_STAGE, source, {"synthetic": True},
              kind="scalars",
              value={"findings": ["a rhythm is present off tissue"],
                     "rhythmic_off_tissue": True},
              name="instrumental_control", output_dir=tmp_path / "out",
              method_version=controls.METHOD_VERSION)

    result = rhythm.test_rhythm(source, traces=values[None, :], times_h=times)

    assert result.control_findings == ["a rhythm is present off tissue"]
    assert result.control["rhythmic_off_tissue"] is True
    assert np.isfinite(result.periods[0]["period_hours"])


def test_a_control_stored_before_the_verdict_was_dropped_is_still_read():
    """Gate 6b: an artefact written under the old shape is not orphaned.

    ``METHOD_VERSION`` is unchanged across the move, deliberately, because it
    reaches the artefact key. So controls stored when this recorded ``reasons``
    and a ``passes`` boolean are still on disk and still valid measurements.
    Auto-Organotypic reads the observations back out of ``reasons``; this
    asserts PyMicroglia gets them through the delegation.
    """
    old_shape = rhythm.RhythmResult(
        labels=["cell_1"],
        control={"passes": False,
                 "reasons": ["a rhythm is present off tissue"]})
    assert old_shape.control_findings == ["a rhythm is present off tissue"]


# ------------------------------------------- nothing is reimplemented here
def test_rhythm_implements_no_period_statistic_of_its_own():
    """Gate 7. A third Lomb-Scargle in this lab is the drift to avoid.

    The search is for the *implementations*, not the words: the module names
    them in prose, and it must, because a reader needs to know what it is
    delegating to.
    """
    code = " ".join(line for line in inspect.getsource(rhythm).splitlines()
                    if not line.strip().startswith("#")
                    and not line.strip().startswith('"'))
    for forbidden in ("signal.lombscargle(", "lombscargle(", "np.polyfit(",
                      "chi2.sf(", "np.linalg.lstsq(", "def _lomb",
                      "def _cosinor"):
        assert forbidden not in code, (
            f"rhythm.py implements {forbidden!r} instead of borrowing it from "
            "circadian_workbench")

    # and the delegation is real, not nominal
    assert "circadian_workbench" in inspect.getsource(rhythm)
    found = rhythm.periodogram(np.arange(0, 96, 0.5),
                               np.sin(2 * np.pi * np.arange(0, 96, 0.5) / 24))
    assert found["source"] == "circadian_workbench.analysis.periodograms"


def test_a_missing_workbench_raises_the_named_error(monkeypatch):
    """Gate 8. Importing PyMicroglia still works; calling in is what fails.

    An ``AttributeError`` from deep inside would send somebody looking for a
    bug in this package. The message has to say what to install.
    """
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "circadian_workbench" or name.startswith("circadian_workbench."):
            raise ImportError("no module named circadian_workbench")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    for module in list(sys.modules):
        if module.startswith("circadian_workbench"):
            monkeypatch.delitem(sys.modules, module, raising=False)

    with pytest.raises(ImportError) as raised:
        rhythm.periodogram([0.0, 1.0], [1.0, 2.0])

    message = str(raised.value)
    assert "Auto-Organotypic[rhythm]" in message
    assert "hard optional" in message


def test_doctor_reports_whether_rhythm_analysis_is_available():
    """Gate 9. Better to learn this before a six-hour run, not at the end."""
    from pymicroglia import knowledge

    report = knowledge.doctor()
    assert "circadian_workbench" in report
    assert "circadian_api_version" in report
    assert "rhythm_analysis_available" in report
    assert report["rhythm_analysis_available"] is (
        report["circadian_api_version"] is not None)


# ----------------------------------------------------------- decoy placement
def test_decoys_off_tissue_are_refused_not_warned_about():
    """Gate 4, and the reason it is a refusal.

    Off tissue the mask-minus-ring baseline is about zero by construction, so
    dF/F divides by nothing, the decoy amplitudes blow up, and a real cell has
    to beat a distribution of near-singular ratios. A warning is something a
    batch run scrolls past.
    """
    labels = np.zeros((60, 60), np.int32)
    labels[20:30, 20:30] = 1
    tissue = np.zeros((60, 60), bool)
    tissue[5:55, 5:55] = True

    with pytest.raises(controls.OffTissueDecoys) as raised:
        controls.decoy_test(None, 0, labels, tissue, np.arange(10.0),
                            on_tissue=False)

    message = str(raised.value)
    assert "ON tissue" in message
    assert "divides by nothing" in message
    assert "no flag for this" in message


def test_decoys_are_generated_only_inside_the_tissue_mask():
    """Every decoy, every pixel — not a sample of them."""
    labels = np.zeros((120, 120), np.int32)
    labels[50:60, 50:60] = 1
    tissue = np.zeros((120, 120), bool)
    tissue[20:100, 20:100] = True

    free = controls.free_tissue(labels, tissue)
    assert not (free & (labels > 0)).any()

    rng = np.random.default_rng(1)
    produced = list(controls.decoy_masks(200, free, (120, 120), count=25,
                                         rng=rng, avoid=labels > 0))
    assert len(produced) == 25
    for _, mask in produced:
        assert (mask & ~tissue).sum() == 0, "a decoy left the tissue mask"
        assert not (mask & (labels > 0)).any(), "a decoy overlapped an object"


def test_there_is_nowhere_on_tissue_says_so_rather_than_going_outside():
    labels = np.ones((40, 40), np.int32)
    tissue = np.ones((40, 40), bool)
    with pytest.raises(controls.OffTissueDecoys) as raised:
        list(controls.decoy_masks(50, controls.free_tissue(labels, tissue),
                                  (40, 40)))
    assert "nowhere on tissue" in str(raised.value)


def test_the_decoy_verdict_is_read_in_absolute_counts(tmp_path, store_root):
    """The record says which quantity decided, because dF/F cannot decide.

    dF/F is reported alongside and is not the verdict, for the reason the
    module docstring gives. A reader should not have to know that from prose.
    """
    from tests_support import two_channel_stack

    from pymicroglia import segmentation, series

    source = two_channel_stack(tmp_path, frames=12)
    found = segmentation.segment(source, output_dir=tmp_path / "out",
                                 channels="dluc=0,struct=1",
                                 stationarity_check=False)

    with series.open_series(source) as opened:
        times = np.asarray(opened.meta.times_h, float)
        result = controls.decoy_test(opened, 0, found.labels,
                                     found.reference.tissue, times,
                                     baseline_h=2.0, count=12, seed=2)

    assert result.records
    measured = [row for row in result.records if "p_counts" in row]
    assert measured, f"nothing was measurable: {result.notes}"
    for row in measured:
        assert row["verdict_read_in"] == "absolute counts"
        assert "amp_dff" in row          # reported...
        assert row["admissible"] == (row["p_counts"] < controls.DECOY_P)


# ---------------------- the detrend the instrumental control is read on
def _drifting_daily(power_of_rhythm: float = 30.0):
    """A slow drift with a modest 24 h cycle riding on it.

    Shaped like a real region mean: a long thermal or bleaching trend an order
    of magnitude larger than the daily variation sitting on top of it.
    """
    hours = np.arange(0.0, 240.0, 0.5)
    drift = 3000.0 + 400.0 * np.sin(2 * np.pi * hours / 900.0)
    return hours, drift + power_of_rhythm * np.sin(2 * np.pi * hours / 24.0)


def test_the_control_detrends_before_it_looks_for_a_period():
    """The failure this guards is quiet and wrong in the dangerous direction.

    A raw region mean is dominated by its own slow drift, so a *normalised*
    periodogram of one reports where that drift lands rather than whether
    anything is periodic. Reading the instrumental control off raw means said
    "no daily cycle" on the reference recording, whose structural channel has a
    22.8 h sinusoid at Lomb-Scargle power 0.966 — the single thing this control
    exists to catch, missed.
    """
    hours, values = _drifting_daily()

    raw = rhythm.periodogram(hours, values, period_range=(15.0, 40.0))
    trimmed, residual = controls._detrended(hours, values, 24.0)
    detrended = rhythm.periodogram(trimmed, residual, period_range=(15.0, 40.0))

    assert detrended["peak_power"] > raw["peak_power"], (
        "detrending must make the daily cycle easier to see, not harder")
    assert abs(detrended["peak_period_hours"] - 24.0) < 1.5, (
        f"the detrended peak should land near 24 h, not "
        f"{detrended['peak_period_hours']:.1f} h")


def test_the_detrended_residual_drops_the_reflected_ends():
    """The half-window at each end is made of reflected samples, not data."""
    hours, values = _drifting_daily()
    length = tracing.window_length(24.0, hours)
    trimmed, residual = controls._detrended(hours, values, 24.0)

    assert len(trimmed) == len(residual)
    assert len(residual) == len(values) - 2 * (length // 2)
    np.testing.assert_allclose(trimmed, hours[length // 2:
                                              len(values) - length // 2])


def test_the_findings_name_the_bioluminescence_channel_when_told_one():
    """How strongly the dLuc channel itself carries the cycle is a field.

    It decides what may be reported at all, so it is read off the top of the
    result rather than hunted for in a table of rows. What it is *not* is a
    pass mark: the companion ``dluc_clean`` boolean went when Auto-Organotypic
    stopped judging, and the power it was derived from stayed.
    """
    hours, values = _drifting_daily()
    frames = len(hours)
    control = controls.ControlResult(
        times_h=hours, region_names=["region"],
        region=np.stack([[values], [values]]),
        off_tissue=np.stack([values, values]),
        sharpness=np.stack([[values], [values]]))

    findings = controls._verdict(control, period_range=(15.0, 40.0),
                                 baseline_h=24.0, dluc_channel=0)
    assert "dluc_roi_power" in findings
    assert 0.0 <= findings["dluc_roi_power"] <= 1.0
    assert "dluc_clean" not in findings, (
        "a pass mark is back on the instrumental control; see "
        "auto_organotypic.instrumental for why there is not one")
    assert len(control.times_h) == frames

    without = controls._verdict(control, period_range=(15.0, 40.0),
                                baseline_h=24.0)
    assert "dluc_roi_power" not in without


# ── the judgement Auto-Organotypic stopped making ───────────────────────────
def _flat(level: float = 3000.0):
    """A series with no daily cycle in it at all."""
    hours = np.arange(0.0, 240.0, 0.5)
    rng = np.random.default_rng(4)
    return hours, level + rng.normal(0.0, 5.0, hours.shape)


def _control_of(off_tissue, region, sharpness):
    hours = off_tissue[0]
    return controls.ControlResult(
        times_h=hours, region_names=["region"],
        region=np.stack([[region[1]]]),
        off_tissue=np.stack([off_tissue[1]]),
        sharpness=np.stack([[sharpness[1]]]))


def test_a_rhythm_off_tissue_is_still_called_the_microscope():
    """The judgement moved to this package; the answer must not have moved with it.

    Auto-Organotypic reported ``instrumental_rhythm_detected`` and ``dluc_clean``
    until it stopped judging per recording. Reading those names through
    ``.get`` with a default would have survived the rename silently and said
    "no instrumental cycle, channel clean" for every recording ever run — the
    one answer that lets a period be reported whatever the control saw.
    """
    daily = _drifting_daily()
    findings = controls._verdict(_control_of(daily, daily, daily),
                                 period_range=(15.0, 40.0), baseline_h=24.0,
                                 dluc_channel=0)
    read = controls.read_findings(findings)

    assert read["instrumental"] is True
    assert read["passes"] is False
    assert read["reasons"], "a finding that changes what may be reported, unsaid"
    assert read["dluc_carries_it"] is True


def test_a_recording_with_nothing_off_tissue_passes():
    flat = _flat()
    daily = _drifting_daily()
    findings = controls._verdict(_control_of(flat, daily, flat),
                                 period_range=(15.0, 40.0), baseline_h=24.0,
                                 dluc_channel=0)
    read = controls.read_findings(findings)

    assert read["instrumental"] is False
    assert read["passes"] is True


def test_the_threshold_is_the_one_the_control_was_run_at():
    """``ls_rhythmic`` decides what "carrying it" means, and is not hard-coded."""
    findings = {"dluc_roi_power": 0.6}
    assert controls.read_findings(findings, rhythmic_power=0.5)["dluc_carries_it"]
    assert not controls.read_findings(findings,
                                      rhythmic_power=0.9)["dluc_carries_it"]


def test_the_pipeline_reads_the_control_through_the_same_function():
    """Two callers, one conclusion.

    The pipeline blocks a periodicity claim on this and the ``run_controls``
    action reports it; the two disagreeing about one recording is worse than
    either being wrong.
    """
    from pymicroglia.pipelines import dluc_single_cell

    findings = {"rhythmic_off_tissue": [0], "dluc_roi_power": 0.9,
                "findings": ["a rhythm is present off tissue"]}
    settings = {"ls_rhythmic": 0.5}
    instrumental, clean = dluc_single_cell._read_control(findings, settings)
    read = controls.read_findings(findings, rhythmic_power=0.5)

    assert instrumental is read["instrumental"]
    assert clean is not read["dluc_carries_it"]
    assert (instrumental, clean) == (True, False)


def test_the_control_action_runs_and_reports_what_it_found(tmp_path, monkeypatch):
    """The action end to end, because nothing else here calls it.

    It read two keys Auto-Organotypic had removed and would have raised
    ``KeyError`` on the first real call. Every test around it passed, because
    every one of them stopped at the signature. This one does not.
    """
    from pymicroglia import segmentation
    from tests_support import oscillating_stack

    monkeypatch.setenv("PYMICROGLIA_STORE", str(tmp_path / "cache"))
    monkeypatch.setenv("PYMICROGLIA_DECISIONS", str(tmp_path / "decisions"))

    source = str(oscillating_stack(tmp_path, frames=48, height=96, width=96))
    segmentation.segment(source, output_dir=str(tmp_path / "out"))
    found = controls.run_controls(source, output_dir=str(tmp_path / "out"),
                                  ndecoy=20)

    assert found["ok"] is True
    assert isinstance(found["passes"], bool)
    assert isinstance(found["reasons"], list)
    assert found["objects_tested"] >= 1
    # Every cell in this fixture oscillates on the same 24 h clock, and so does
    # the background it sits on, so the honest reading of it is "common-mode".
    assert found["passes"] is False
    assert any("off tissue" in reason for reason in found["reasons"])


def test_a_whole_run_reaches_a_conclusion_about_the_control(tmp_path, monkeypatch):
    """The pipeline's own control branch, which every other run test skips.

    ``skip_control=True`` is how the rest of the suite keeps its runs short, so
    the branch that decides whether a period may be reported at all was reached
    by nothing. With the dropped keys read through ``.get`` it would have
    written "the instrumental control passed" onto a recording whose every
    channel carries the same daily cycle.
    """
    from pymicroglia.pipelines import dluc_single_cell
    from tests_support import oscillating_stack

    monkeypatch.setenv("PYMICROGLIA_STORE", str(tmp_path / "cache"))
    monkeypatch.setenv("PYMICROGLIA_DECISIONS", str(tmp_path / "decisions"))

    manifest = dluc_single_cell.run(
        oscillating_stack(tmp_path, frames=48, height=96, width=96),
        output_dir=tmp_path / "out", if_exists="error",
        t0=None, t1=None, baselines=(6.0,), detrends=("cubic",), ndecoy=20,
        skip_videos=True, skip_control=False)

    control = [note for note in manifest["review"] if note["gate"] == "control"]
    assert len(control) == 1, control
    assert "passed" not in control[0]["headline"], (
        "a recording whose off-tissue pixels carry the same daily cycle was "
        "reported as having no instrumental rhythm")
    assert "instrumental cycle is present" in control[0]["headline"]
