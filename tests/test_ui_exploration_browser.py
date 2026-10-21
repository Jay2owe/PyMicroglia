"""Opt-in real-browser gate; uses the dedicated already running workbench."""
import os
from pathlib import Path
import sys
import pytest


@pytest.mark.skipif(not os.environ.get("PYMICROGLIA_UI_TEST_URL"), reason="Set the live workbench URL to run browser acceptance")
def test_live_workbench_browser():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from verify_exploration_ui import verify
    folder = Path(os.environ["PYMICROGLIA_UI_TEST_FIXTURE"])
    assert verify(os.environ["PYMICROGLIA_UI_TEST_URL"], folder)["passed"]
