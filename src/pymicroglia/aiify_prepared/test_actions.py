import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

def bundle():
    path = Path(__file__).with_name("actions.py")
    spec = importlib.util.spec_from_file_location("reviewed_shortcut", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_requires_live_binding():
    with pytest.raises(RuntimeError):
        bundle().workspace_overview()

def test_real_backend_readback(monkeypatch):
    monkeypatch.setenv("PYMICROGLIA_AI", "0")
    from pymicroglia.ui.server import create_app
    from pymicroglia.ai_assistant import overview
    app = create_app()
    try:
        app.state.aiify_overview = lambda: overview(app)
        module = bundle()
        module.bind(app)
        assert module.workspace_overview()["jobs"] == app.state.jobs.list()
        assert module.workspace_overview() == overview(app)
        assert not app.state.jobs.jobs
    finally:
        app.state.jobs.pool.shutdown(wait=True)
