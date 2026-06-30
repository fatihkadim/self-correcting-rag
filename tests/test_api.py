from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# TestClient — SelfCorrectionController'ı mock'layarak app'i yükler
# ---------------------------------------------------------------------------

# main.py modül seviyesinde `pipeline = SelfCorrectionController()` çağırdığı
# için, import öncesinde __init__'i patch'lememiz gerekir.
with patch("app.agent.controller.SelfCorrectionController.__init__", lambda self: None):
    with patch("app.agent.controller.SelfCorrectionController.run") as _mock_run:
        from app.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Test 1 — /health endpoint'i
# ---------------------------------------------------------------------------

def test_health_endpoint_returns_200():
    """/health endpoint'i 200 ve 'healthy' status dönmeli."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "server time" in data


# ---------------------------------------------------------------------------
# Test 2 — /files endpoint'i
# ---------------------------------------------------------------------------

def test_files_endpoint_returns_200():
    """/files endpoint'i 200 ve files listesi dönmeli."""
    response = client.get("/files")
    assert response.status_code == 200
    data = response.json()
    assert "files" in data
    assert isinstance(data["files"], list)


# ---------------------------------------------------------------------------
# Test 3 — /query endpoint'i (geçersiz body)
# ---------------------------------------------------------------------------

def test_query_endpoint_rejects_empty_body():
    """/query boş body ile 422 dönmeli."""
    response = client.post("/query", json={})
    assert response.status_code == 422


def test_query_endpoint_rejects_missing_content_type():
    """/query content-type olmadan 422 dönmeli."""
    response = client.post("/query")
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Test 4 — /upload endpoint'i (geçersiz dosya türü)
# ---------------------------------------------------------------------------

def test_upload_rejects_non_pdf():
    """/upload PDF olmayan dosyayı reddetmeli."""
    response = client.post(
        "/upload",
        files={"file": ("test.txt", b"hello world", "text/plain")},
    )
    assert response.status_code == 200  # Uygulama 200 döndürüp error mesajı veriyor
    data = response.json()
    assert "error" in data or data.get("status") == "error"
