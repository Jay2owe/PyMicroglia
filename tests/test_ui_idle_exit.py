"""The browser launcher stops once its page has stopped sending heartbeats."""

import threading
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient

from pymicroglia.ui import desktop, server
from pymicroglia.ui.jobs import JobManager


class AliveThread:
    def is_alive(self):
        return True


def _app(busy=False):
    jobs = SimpleNamespace(busy=lambda: busy)
    return SimpleNamespace(state=SimpleNamespace(last_heartbeat=None, jobs=jobs))


def test_silence_after_grace_stops_the_wait():
    app = _app()
    started = time.monotonic()
    assert desktop._wait_for_page(AliveThread(), app, timeout=.05, grace=.1, poll=.01)
    assert time.monotonic() - started >= .1


def test_heartbeats_keep_it_alive_then_silence_stops_it():
    app = _app()
    stop = threading.Event()

    def beat():
        while not stop.is_set():
            app.state.last_heartbeat = time.monotonic()
            time.sleep(.02)

    beater = threading.Thread(target=beat, daemon=True)
    beater.start()
    result = {}
    waiter = threading.Thread(target=lambda: result.update(closed=desktop._wait_for_page(
        AliveThread(), app, timeout=.15, grace=.15, poll=.01)), daemon=True)
    waiter.start()
    waiter.join(.6)
    assert waiter.is_alive()  # still beating, still waiting
    stop.set()
    waiter.join(2)
    assert result == {"closed": True}


def test_running_job_blocks_exit():
    now = time.monotonic()
    app = _app(busy=True)
    app.state.last_heartbeat = now - 1000
    assert not desktop._page_closed(app, now - 1000, now, timeout=90, grace=300)
    app.state.jobs.busy = lambda: False
    assert desktop._page_closed(app, now - 1000, now, timeout=90, grace=300)


def test_grace_applies_before_first_heartbeat_and_zero_disables():
    now = time.monotonic()
    app = _app()
    assert not desktop._page_closed(app, now - 200, now, timeout=90, grace=300)
    assert desktop._page_closed(app, now - 301, now, timeout=90, grace=300)
    assert not desktop._page_closed(app, now - 10_000, now, timeout=0, grace=300)


def test_timeout_comes_from_flag_then_environment(monkeypatch):
    monkeypatch.delenv("PYMICROGLIA_UI_IDLE_TIMEOUT", raising=False)
    assert desktop._seconds(None, "PYMICROGLIA_UI_IDLE_TIMEOUT", 90) == 90
    monkeypatch.setenv("PYMICROGLIA_UI_IDLE_TIMEOUT", "0")
    assert desktop._seconds(None, "PYMICROGLIA_UI_IDLE_TIMEOUT", 90) == 0
    assert desktop._seconds(5, "PYMICROGLIA_UI_IDLE_TIMEOUT", 90) == 5


def test_job_manager_reports_pending_jobs():
    manager = JobManager(runner=lambda **kwargs: {})
    try:
        assert not manager.busy()
        manager.jobs["a"] = {"id": "a", "state": "running"}
        assert manager.busy()
        manager.jobs["a"]["state"] = "completed"
        assert not manager.busy()
    finally:
        manager.close()


def test_heartbeat_endpoint_records_time_and_pings_motion(monkeypatch):
    pinged = []
    monkeypatch.setattr(server, "_ping_motion", pinged.append)
    app = server.create_app(motion_url="http://127.0.0.1:12345")
    assert app.state.last_heartbeat is None
    response = TestClient(app).get("/api/heartbeat")
    assert response.status_code == 200 and response.json()["busy"] is False
    assert app.state.last_heartbeat is not None
    for _ in range(50):
        if pinged:
            break
        time.sleep(.01)
    assert pinged == ["http://127.0.0.1:12345"]


def test_motion_ping_ignores_an_unreachable_service():
    server._ping_motion("http://127.0.0.1:9")  # nothing listens; must not raise


def test_pages_load_the_heartbeat_script():
    client = TestClient(server.create_app())
    for page in ("/", "/review", "/correction-review"):
        assert "/assets/heartbeat.js" in client.get(page).text
    assert "/api/heartbeat" in client.get("/assets/heartbeat.js").text
