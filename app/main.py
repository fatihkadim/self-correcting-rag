from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import hashlib
import re
import time
import os
import uuid
from app.core.config import settings
from app.core.logger import get_logger
from app.schemas.query import QueryRequest, QueryResponse
from app.agent.controller import SelfCorrectionController

logger = get_logger(__name__)

pipeline = SelfCorrectionController()

app = FastAPI(title="Self Correction RAG")

# Mount static files directory
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Mount assets directory from root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
assets_dir = os.path.join(project_root, "assets")
os.makedirs(assets_dir, exist_ok=True)
app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

RAW_DIR = os.path.join(project_root, "data", "raw")
ALLOWED_EXTENSIONS = {".pdf", ".txt"}
MAX_FILENAME_STEM = 150
UPLOAD_READ_CHUNK = 1024 * 1024
# Windows'ta dosya adı olarak kullanılamayan aygıt adları
_RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                   *(f"LPT{i}" for i in range(1, 10))}
_UNSAFE_CHARS_RE = re.compile(r"[^\w\-. ()]")


def safe_filename(filename: str | None) -> str:
    """Kullanıcıdan gelen dosya adını data/raw/ içinde güvenle kullanılabilir hale getirir.

    Dizin bileşenleri atılır (path traversal), izin verilmeyen karakterler `_` olur.
    Uzantı desteklenmiyorsa 415, geriye kullanılabilir bir ad kalmazsa 400 fırlatır.
    """
    name = os.path.basename((filename or "").replace("\\", "/"))
    stem, ext = os.path.splitext(name)
    ext = ext.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Sadece PDF ve TXT dosyaları desteklenmektedir.")

    stem = _UNSAFE_CHARS_RE.sub("_", stem).strip(" .")[:MAX_FILENAME_STEM]
    if not stem or not stem.strip("_"):
        raise HTTPException(status_code=400, detail="Geçersiz dosya adı.")
    if stem.upper() in _RESERVED_NAMES:
        stem = f"_{stem}"
    return stem + ext


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(UPLOAD_READ_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _find_duplicate_content(digest: str) -> str | None:
    """Aynı içeriğe sahip, daha önce yüklenmiş dosyanın adını döndürür."""
    for name in os.listdir(RAW_DIR):
        path = os.path.join(RAW_DIR, name)
        if (os.path.splitext(name)[1].lower() in ALLOWED_EXTENSIONS and os.path.isfile(path)
                and _file_sha256(path) == digest):
            return name
    return None


def _validate_content(ext: str, head: bytes) -> None:
    """Dosya içeriğinin uzantısıyla uyumlu olduğunu kontrol eder."""
    if ext == ".pdf" and not head.startswith(b"%PDF-"):
        raise HTTPException(status_code=415, detail="Dosya geçerli bir PDF değil.")
    if ext == ".txt" and b"\x00" in head:
        raise HTTPException(status_code=415, detail="Dosya düz metin (TXT) değil.")


@app.get("/")
def read_root():
    return FileResponse(os.path.join(static_dir, "index.html"))

@app.get("/health")
def check_health():
    return {
        "status":"healthy",
        "server time": time.time()
    }

@app.post("/query",response_model=QueryResponse)
def query(req: QueryRequest):
    try:
        return pipeline.run(request=req)
    except Exception:
        # Ayrıntı sadece loglanır; iç hata mesajları istemciye sızdırılmaz.
        logger.exception("Sorgu işlenirken hata oluştu")
        raise HTTPException(status_code=500, detail="Sorgu işlenirken bir hata oluştu.")

# Senkron endpoint: FastAPI thread pool'da çalıştırır, uzun süren ingest
# işlemi event loop'u bloklamaz.
@app.post("/upload")
def upload_document(file: UploadFile = File(...)):
    filename = safe_filename(file.filename)
    ext = os.path.splitext(filename)[1]
    os.makedirs(RAW_DIR, exist_ok=True)
    file_path = os.path.join(RAW_DIR, filename)

    # Aynı dosyayı tekrar yüklemek Qdrant'ta mükerrer chunk'lar oluşturur.
    if os.path.exists(file_path):
        raise HTTPException(status_code=409, detail=f"{filename} zaten yüklü.")

    max_bytes = settings.max_upload_mb * 1024 * 1024
    tmp_path = os.path.join(RAW_DIR, f".upload-{uuid.uuid4().hex}.part")
    try:
        size = 0
        digest = hashlib.sha256()
        with open(tmp_path, "wb") as f:
            while chunk := file.file.read(UPLOAD_READ_CHUNK):
                if size == 0:
                    _validate_content(ext, chunk)
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Dosya boyutu {settings.max_upload_mb} MB sınırını aşıyor.",
                    )
                digest.update(chunk)
                f.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail="Dosya boş.")
        # Farklı adla yüklenen aynı içerik de Qdrant'ta mükerrer chunk oluşturur.
        duplicate = _find_duplicate_content(digest.hexdigest())
        if duplicate:
            raise HTTPException(status_code=409, detail=f"Bu içerik zaten {duplicate} olarak yüklü.")
        try:
            os.rename(tmp_path, file_path)
        except FileExistsError:
            raise HTTPException(status_code=409, detail=f"{filename} zaten yüklü.")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    from scripts.ingest import ingest_documents
    try:
        chunk_count = ingest_documents([Path(file_path)])
    except Exception:
        logger.exception("Ingest başarısız: %s", filename)
        os.remove(file_path)
        raise HTTPException(status_code=500, detail="Dosya işlenirken bir hata oluştu.")

    if not chunk_count:
        os.remove(file_path)
        raise HTTPException(status_code=422, detail="Dosyadan metin çıkarılamadı.")

    return {"status": "success", "message": f"{filename} başarıyla yüklendi ve işlendi."}

@app.get("/files")
def get_files():
    if not os.path.exists(RAW_DIR):
        return {"files": []}

    files = []
    for f in os.listdir(RAW_DIR):
        if os.path.splitext(f)[1].lower() in ALLOWED_EXTENSIONS:
            files.append(f)
    return {"files": sorted(files)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
