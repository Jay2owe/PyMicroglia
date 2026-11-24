"""The mask annotation apps stop themselves once their page stops sending heartbeats."""

import importlib.util
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import urlopen

import pytest

DEVELOPMENT = Path(__file__).resolve().parents[1] / "development"
COPIES = [DEVELOPMENT / "cd68_lv200_mask_unet/annotation/code/annotate_app.py",
          DEVELOPMENT / "single_frame_mask_unet/MCG_mask_annotation/code/annotate_app.py"]


def _load(path):
    if not path.is_file():
        pytest.skip(f"{path.name} copy is not present")
    spec = importlib.util.spec_from_file_location(f"annotate_{abs(hash(path))}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _serve(module, timeout, grace):
    server = ThreadingHTTPServer(("127.0.0.1", 0), module.Handler)
    module.stop_when_idle(server, timeout, grace, poll=.02)
    runner = threading.Thread(target=server.serve_forever, daemon=True)
    runner.start()
    return server, runner


def _beat(server):
    with urlopen(f"http://127.0.0.1:{server.server_address[1]}/heartbeat", timeout=2) as response:
        assert response.status == 200


@pytest.mark.parametrize("path", COPIES, ids=["current", "old"])
def test_heartbeats_keep_it_up_and_silence_stops_it(path):
    module = _load(path)
    server, runner = _serve(module, timeout=1, grace=1)
    try:
        for _ in range(12):          # 2.4 s of beats, well past the 1 s limits
            _beat(server)
            time.sleep(.2)
        assert runner.is_alive()
        runner.join(5)               # silence: the watcher calls server.shutdown()
        assert not runner.is_alive()
    finally:
        server.shutdown() if runner.is_alive() else None
        server.server_close()


@pytest.mark.parametrize("path", COPIES, ids=["current", "old"])
def test_no_first_heartbeat_stops_after_grace(path):
    module = _load(path)
    started = time.monotonic()
    server, runner = _serve(module, timeout=.05, grace=.3)
    try:
        runner.join(3)
        assert not runner.is_alive() and time.monotonic() - started >= .3
    finally:
        server.server_close()


@pytest.mark.parametrize("path", COPIES, ids=["current", "old"])
def test_zero_disables(path):
    module = _load(path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), module.Handler)
    try:
        assert module.stop_when_idle(server, 0, 0) is None
    finally:
        server.server_close()


@pytest.mark.parametrize("path", COPIES, ids=["current", "old"])
def test_page_sends_heartbeats(path):
    assert 'fetch("/heartbeat"' in path.with_name("ui.html").read_text(encoding="utf-8") \
        if path.is_file() else pytest.skip("copy not present")
