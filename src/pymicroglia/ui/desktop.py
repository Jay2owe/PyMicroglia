"""One native window for Auto-Organotypic, Motion and PyMicroglia."""

from __future__ import annotations

import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from urllib.parse import urlencode
from urllib.request import urlopen
import webbrowser


class Bridge:
    """The only desktop-only operation: choose a local results folder."""

    def choose_folder(self, start: str | None = None) -> str | None:
        import webview
        window = webview.active_window()
        if window is None:
            return None
        picked = window.create_file_dialog(webview.FOLDER_DIALOG,
                                           directory=str(start or ""))
        return str(picked[0]) if picked else None

    def choose_model(self, start: str | None = None) -> str | None:
        import webview
        window = webview.active_window()
        if window is None:
            return None
        picked = window.create_file_dialog(
            webview.OPEN_DIALOG, directory=str(start or ""),
            file_types=("PyTorch model (*.pt;*.pth)",))
        return str(picked[0]) if picked else None


def _port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _server(app, port: int):
    import uvicorn
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                           log_level="warning", access_log=False,
                                           timeout_graceful_shutdown=5))
    thread = threading.Thread(target=server.run, daemon=True, name=f"pymicroglia-ui-{port}")
    thread.start()
    return server, thread


def _ready(url: str, path: str, timeout: float = 30) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            with urlopen(url + path, timeout=.5) as response:
                if response.status == 200:
                    return True
        except OSError:
            time.sleep(.1)
    return False


def _motion_checkout(value: str | None) -> Path | None:
    if value:
        path = Path(value).expanduser().resolve()
        if not (path / "motion_edit" / "ui_cli.py").is_file():
            raise ValueError("--motion-root must name the Motion checkout")
        return path
    # Convenient when launched from this project's checkout. This is an
    # optional service process, never an import dependency of PyMicroglia.
    candidate = Path.cwd().parent / "Motion"
    return candidate.resolve() if (candidate / "motion_edit" / "ui_cli.py").is_file() else None


def launch(*, browser: bool = False, project: str | None = None,
           motion_root: str | None = None, motion_url: str | None = None,
           serve_only: bool = False) -> int:
    try:
        from auto_organotypic.server import create_app as auto_app
    except ImportError as exc:
        print(f"Auto-Organotypic is required: {exc}", file=sys.stderr)
        return 1
    from .server import create_app

    owned_servers = []
    motion_process = None
    try:
        auto_port = _port()
        auto_base = f"http://127.0.0.1:{auto_port}"
        server, thread = _server(auto_app(), auto_port)
        owned_servers.append((server, thread))
        if not _ready(auto_base, "/api/health"):
            raise RuntimeError("Auto-Organotypic did not start")
        auto_url = auto_base + "/workflow"
        if project:
            auto_url += "?" + urlencode({"folder": str(Path(project).expanduser().resolve())})

        if not motion_url:
            checkout = _motion_checkout(motion_root)
            if checkout:
                edit_root = checkout / "manual_editing"
                motion_port = _port()
                motion_url = f"http://127.0.0.1:{motion_port}"
                flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
                motion_process = subprocess.Popen(
                    [sys.executable, "-m", "motion_edit.ui_cli", "--root", str(edit_root),
                     "--port", str(motion_port)], cwd=str(checkout),
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL, **flags)
                if not _ready(motion_url, "/api/actions", timeout=20):
                    motion_process.terminate()
                    motion_process = None
                    motion_url = None

        ui_port = _port()
        ui_url = f"http://127.0.0.1:{ui_port}"
        server, thread = _server(create_app(auto_url=auto_url, motion_url=motion_url), ui_port)
        owned_servers.append((server, thread))
        if not _ready(ui_url, "/api/config"):
            raise RuntimeError("PyMicroglia workbench did not start")
        if serve_only:
            print(f"PyMicroglia workbench: {ui_url}")
            try:
                while thread.is_alive():
                    time.sleep(.5)
            except KeyboardInterrupt:
                return 0
            return 0
        if browser:
            print(f"PyMicroglia workbench: {ui_url}")
            webbrowser.open(ui_url)
            try:
                while thread.is_alive():
                    time.sleep(.5)
            except KeyboardInterrupt:
                return 0
            return 0
        try:
            import webview
        except ImportError:
            print('Install "PyMicroglia[desktop]" or use pymicroglia ui --browser.', file=sys.stderr)
            return 1
        webview.settings["ALLOW_DOWNLOADS"] = True
        webview.create_window("PyMicroglia Workbench", ui_url, width=1440, height=900,
                              min_size=(1000, 650), js_api=Bridge(),
                              background_color="#0b1725")
        webview.start()
        return 0
    finally:
        for server, thread in reversed(owned_servers):
            server.should_exit = True
            thread.join(timeout=10)
        if motion_process and motion_process.poll() is None:
            motion_process.terminate()
            try:
                motion_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                motion_process.kill()
