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
# Test 4 — /query girdi doğrulama ve hata yönetimi
# ---------------------------------------------------------------------------

import sys
import types

import pytest
from fastapi import HTTPException

import app.main as main_module
from app.core.config import settings
from app.main import safe_filename


@pytest.mark.parametrize("question", ["", "   ", "ab", "x" * (settings.max_question_length + 1)])
def test_query_rejects_invalid_question_length(question):
    """Boş, sadece boşluk, çok kısa veya çok uzun soru 422 dönmeli."""
    response = client.post("/query", json={"question": question})
    assert response.status_code == 422


def test_query_internal_error_returns_500_without_details(monkeypatch):
    """Pipeline hatası 500 döner; iç hata mesajı istemciye sızmaz."""
    def boom(request):
        raise RuntimeError("gizli-iç-detay")
    monkeypatch.setattr(main_module.pipeline, "run", boom)

    response = client.post("/query", json={"question": "Python nedir?"})

    assert response.status_code == 500
    assert "gizli-iç-detay" not in response.text


# ---------------------------------------------------------------------------
# Test 5 — safe_filename
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("../../etc/passwd.pdf", "passwd.pdf"),
    (r"..\..\windows\evil.PDF", "evil.pdf"),
    ("rapor<script>.pdf", "rapor_script_.pdf"),
    ("Türkçe Doküman (v2).txt", "Türkçe Doküman (v2).txt"),
    ("CON.txt", "_CON.txt"),
])
def test_safe_filename_sanitizes(raw, expected):
    """Dizin bileşenleri atılır, tehlikeli karakterler temizlenir."""
    assert safe_filename(raw) == expected


@pytest.mark.parametrize("raw, status", [
    ("notes.exe", 415),
    ("noext", 415),
    ("<>.pdf", 400),
    (".pdf", 415),
    (None, 415),
])
def test_safe_filename_rejects(raw, status):
    with pytest.raises(HTTPException) as exc:
        safe_filename(raw)
    assert exc.value.status_code == status


# ---------------------------------------------------------------------------
# Test 6 — /upload
# ---------------------------------------------------------------------------

PDF_BYTES = b"%PDF-1.4 dummy content"


@pytest.fixture
def upload_env(tmp_path, monkeypatch):
    """data/raw yerine geçici dizin; ingest Qdrant/embedder yerine fake."""
    monkeypatch.setattr(main_module, "RAW_DIR", str(tmp_path))
    fake_ingest = MagicMock(return_value=5)
    monkeypatch.setitem(sys.modules, "scripts.ingest",
                        types.SimpleNamespace(ingest_documents=fake_ingest))
    return tmp_path, fake_ingest


def _upload(name, content, mime="application/pdf"):
    return client.post("/upload", files={"file": (name, content, mime)})


def test_upload_success_writes_sanitized_file(upload_env):
    raw_dir, fake_ingest = upload_env
    response = _upload("../../secret.pdf", PDF_BYTES)

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert (raw_dir / "secret.pdf").read_bytes() == PDF_BYTES
    assert not (raw_dir.parent / "secret.pdf").exists()
    fake_ingest.assert_called_once()
    assert [p.name for p in raw_dir.iterdir()] == ["secret.pdf"]  # .part dosyası kalmaz


def test_upload_accepts_txt(upload_env):
    raw_dir, _ = upload_env
    response = _upload("notlar.txt", "merhaba dünya".encode(), "text/plain")
    assert response.status_code == 200
    assert (raw_dir / "notlar.txt").exists()


def test_upload_rejects_unsupported_extension(upload_env):
    response = _upload("script.exe", b"MZ...", "application/octet-stream")
    assert response.status_code == 415


def test_upload_rejects_fake_pdf(upload_env):
    """Uzantısı .pdf olan ama PDF imzası taşımayan dosya reddedilir."""
    raw_dir, fake_ingest = upload_env
    response = _upload("fake.pdf", b"<html>not a pdf</html>")
    assert response.status_code == 415
    assert list(raw_dir.iterdir()) == []
    fake_ingest.assert_not_called()


def test_upload_rejects_binary_txt(upload_env):
    response = _upload("binary.txt", b"abc\x00def", "text/plain")
    assert response.status_code == 415


def test_upload_rejects_empty_file(upload_env):
    raw_dir, _ = upload_env
    response = _upload("bos.txt", b"", "text/plain")
    assert response.status_code == 400
    assert list(raw_dir.iterdir()) == []


def test_upload_rejects_oversized_file(upload_env, monkeypatch):
    raw_dir, fake_ingest = upload_env
    monkeypatch.setattr(settings, "max_upload_mb", 1)
    big = b"%PDF-" + b"0" * (1024 * 1024 + 10)

    response = _upload("big.pdf", big)

    assert response.status_code == 413
    assert list(raw_dir.iterdir()) == []
    fake_ingest.assert_not_called()


def test_upload_rejects_duplicate(upload_env):
    raw_dir, fake_ingest = upload_env
    (raw_dir / "var.pdf").write_bytes(PDF_BYTES)

    response = _upload("var.pdf", PDF_BYTES)

    assert response.status_code == 409
    fake_ingest.assert_not_called()


def test_upload_no_extractable_text_returns_422_and_removes_file(upload_env):
    raw_dir, fake_ingest = upload_env
    fake_ingest.return_value = 0

    response = _upload("taranmis.pdf", PDF_BYTES)

    assert response.status_code == 422
    assert list(raw_dir.iterdir()) == []


def test_upload_ingest_error_returns_500_without_details(upload_env):
    raw_dir, fake_ingest = upload_env
    fake_ingest.side_effect = RuntimeError("qdrant-bağlantı-detayı")

    response = _upload("doc.pdf", PDF_BYTES)

    assert response.status_code == 500
    assert "qdrant-bağlantı-detayı" not in response.text
    assert list(raw_dir.iterdir()) == []


def test_upload_rejects_same_content_under_different_name(upload_env):
    """Aynı içerik farklı adla yüklenirse 409 döner, ingest çağrılmaz."""
    raw_dir, fake_ingest = upload_env
    (raw_dir / "orijinal.pdf").write_bytes(PDF_BYTES)

    response = _upload("kopya.pdf", PDF_BYTES)

    assert response.status_code == 409
    assert "orijinal.pdf" in response.json()["detail"]
    assert sorted(p.name for p in raw_dir.iterdir()) == ["orijinal.pdf"]
    fake_ingest.assert_not_called()
