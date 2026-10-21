"""The native host owns and closes both local application servers."""

import sys
from types import SimpleNamespace

from pymicroglia.ui import desktop


def test_native_window_owns_both_servers(monkeypatch):
    servers = []
    windows = []

    class FakeThread:
        def join(self, timeout=None):
            pass

    def fake_server(app, port):
        server = SimpleNamespace(should_exit=False)
        servers.append(server)
        return server, FakeThread()

    monkeypatch.setattr(desktop, "_server", fake_server)
    monkeypatch.setattr(desktop, "_port", iter((12001, 12002)).__next__)
    monkeypatch.setattr(desktop, "_ready", lambda *args, **kwargs: True)
    monkeypatch.setattr(desktop, "_motion_checkout", lambda value: None)
    monkeypatch.setitem(sys.modules, "webview", SimpleNamespace(
        settings={}, create_window=lambda *args, **kwargs: windows.append((args, kwargs)),
        start=lambda: None))
    assert desktop.launch() == 0
    assert len(servers) == 2 and all(server.should_exit for server in servers)
    assert windows[0][0][1] == "http://127.0.0.1:12002"
    assert isinstance(windows[0][1]["js_api"], desktop.Bridge)
