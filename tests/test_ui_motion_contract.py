import pytest
from fastapi.testclient import TestClient

from pymicroglia.ui.server import create_app


def test_motion_proxy_is_optional_and_loopback_only():
    client = TestClient(create_app())
    assert client.get("/api/motion/actions").status_code == 503
    with pytest.raises(ValueError, match="loopback"):
        create_app(motion_url="https://example.org/")
    client = TestClient(create_app(motion_url="http://127.0.0.1:12345"))
    assert client.get("/api/motion/not-an-action").status_code == 404
    assert client.post("/api/motion/sources").status_code == 405


def test_frontend_contains_separate_motion_workspace():
    client = TestClient(create_app())
    html = client.get("/").text
    assert html.index("Auto-Organotypic</button>") < html.index("Motion tracks</button>") < html.index("PyMicroglia extras</button>")
    assert 'id="motion-sidebar"' in html
    assert 'id="micro-sidebar"' in html
