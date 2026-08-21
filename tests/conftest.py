"""Shared test fixtures.

Two kinds of test live in this suite.

**Self-contained tests** run always. They use the small synthetic scripts in
``tests/fixtures/`` and exercise the reader without needing any real protocol to
be present.

**Source-verification tests** compare what PyMicroglia reads against the real
scripts in ``Protocols/Analysis``. They are how a copy is verified against the
thing it was copied from — and they **skip** when that folder is not there,
because the package must work without it. Point ``PYMICROGLIA_PROTOCOLS`` at a
non-existent path to prove that:

    PYMICROGLIA_PROTOCOLS=/nowhere python -m pytest

The path knowledge lives here, in the tests, and never in ``src/pymicroglia``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

PROTOCOLS_ENV = "PYMICROGLIA_PROTOCOLS"

FIXTURES = Path(__file__).parent / "fixtures"


def protocols_root() -> Path | None:
    """The ``Protocols`` folder to verify against, or ``None`` if unavailable."""
    override = os.environ.get(PROTOCOLS_ENV)
    if override is not None:
        candidate = Path(override)
        return candidate if candidate.is_dir() else None
    default = Path(__file__).resolve().parents[2] / "Protocols"
    return default if default.is_dir() else None


@pytest.fixture(scope="session")
def protocols() -> Path:
    root = protocols_root()
    if root is None:
        pytest.skip(
            "Protocols folder not available; source-verification skipped. "
            "This is the expected result when the package is used on its own."
        )
    return root


@pytest.fixture(scope="session")
def fixtures() -> Path:
    return FIXTURES


@pytest.fixture(autouse=True, scope="session")
def quarantined_run_index(tmp_path_factory):
    """Keep the suite out of the real global run index.

    Every captured run appends one line to that index, which lives in the shared
    Dropbox config folder and is what "have we done this before?" reads. A suite
    that appended to it would bury the real runs under a few hundred runs over
    synthetic 48x48 stacks, each pointing at a temporary folder that no longer
    exists.

    Redirected by moving the kit's own default root for the session, so it holds
    however a test reaches the audit layer, and put back afterwards.
    """
    from pymicroglia._optional import kit

    installed = kit()
    if installed is None:  # pragma: no cover - the suite runs without the kit
        yield None
        return

    index = installed.audit.index
    root = tmp_path_factory.mktemp("global_run_index")
    original, index.DEFAULT_ROOT = index.DEFAULT_ROOT, root
    try:
        yield root
    finally:
        index.DEFAULT_ROOT = original


#: Which files carry each registered protocol's parameters, copied from
#: ``Protocols/INDEX.md``. Paths are relative to the ``Protocols`` folder.
#:
#: The counts are what the block grammar yields from the live files today. Note
#: ``microglia_bioluminescence_display``: INDEX.md says 19 and the files say 20,
#: because ``..._analysis.ijm`` gained ``DEFAULT_PYTHON_ENGINE`` after INDEX was
#: last regenerated. The files are the truth; INDEX is one line stale.
PROTOCOL_FILES: dict[str, tuple[int, tuple[str, ...]]] = {
    "cry1_dluc_photon_pipeline": (22, (
        "Analysis/cry1_dluc_photon_pipeline.py",
    )),
    "dLuc_single_cell_analysis": (52, (
        "Analysis/dLuc_single_cell_analysis/dluc_pipeline.py",
        "Analysis/dLuc_single_cell_analysis/run_dluc_analysis.ps1",
        "Analysis/dLuc_single_cell_analysis/run_dluc_batch.ps1",
        "Analysis/dLuc_single_cell_analysis/setup_environment.ps1",
    )),
    "microglia_bioluminescence_display": (20, (
        "Analysis/microglia_bioluminescence_display.py",
        "Analysis/microglia_bioluminescence_display_analysis.ijm",
    )),
    "microglia_cosmic_ray_removal": (16, (
        "Analysis/microglia_cosmic_ray_removal.py",
        "Analysis/microglia_cosmic_ray_removal_analysis.ijm",
    )),
    "microglia_phase_correlation_registration": (7, (
        "Analysis/microglia_phase_correlation_registration.py",
        "Analysis/microglia_phase_correlation_registration_analysis.ijm",
    )),
    "microglia_raw_registered_stack_export": (4, (
        "Analysis/microglia_raw_registered_stack_export.py",
        "Analysis/microglia_raw_registered_stack_analysis.ijm",
    )),
    "microglia_red_only_video_export": (12, (
        "Analysis/microglia_red_only_video_export.py",
    )),
    "microglia_static_background_removal": (15, (
        "Analysis/microglia_static_background_removal.py",
        "Analysis/microglia_static_background_removal_analysis.ijm",
    )),
    "microglia_timestamped_composite_video_export": (11, (
        "Analysis/microglia_timestamped_composite_video_export.py",
    )),
    "phase_green_red_timelapse_pipeline": (8, (
        "Analysis/phase_green_red_timelapse_pipeline.py",
        "Analysis/phase_green_red_registration_analysis.ijm",
    )),
    "phase_green_red_video_export": (22, (
        "Analysis/phase_green_red_video_export.py",
    )),
    "tiff_stack_to_mp4": (24, (
        "Analysis/tiff_stack_to_mp4.py",
        "Analysis/tiff_stack_to_mp4.ijm",
    )),
    "trace_panel_figure": (54, (
        "Analysis/trace_panel_figure/run_trace_panel_figure.ps1",
        "Analysis/trace_panel_figure/trace_panel_figure.py",
    )),
}

#: Protocols whose engine declares a METHOD_VERSION. The other five do not, and
#: the reader must return "" for them rather than raising.
WITH_METHOD_VERSION = {
    "cry1_dluc_photon_pipeline",
    "microglia_cosmic_ray_removal",
    "microglia_phase_correlation_registration",
    "microglia_raw_registered_stack_export",
    "microglia_red_only_video_export",
    "microglia_timestamped_composite_video_export",
    "phase_green_red_timelapse_pipeline",
    "tiff_stack_to_mp4",
}
