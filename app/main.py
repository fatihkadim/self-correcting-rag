from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import time
import os
from app.schemas.query import QueryRequest, QueryResponse
from app.agent.controller import SelfCorrectionController

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
    return pipeline.run(request=req)

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        return {"error": "Sadece PDF dosyaları desteklenmektedir."}
    
    raw_dir = os.path.join(project_root, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    file_path = os.path.join(raw_dir, file.filename)
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Run ingestion
    from scripts.ingest import ingest_documents
    from pathlib import Path
    try:
        ingest_documents([Path(file_path)])
        return {"status": "success", "message": f"{file.filename} başarıyla yüklendi ve işlendi."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/files")
def get_files():
    raw_dir = os.path.join(project_root, "data", "raw")
    if not os.path.exists(raw_dir):
        return {"files": []}
    
    files = []
    for f in os.listdir(raw_dir):
        if f.lower().endswith('.pdf') or f.lower().endswith('.txt'):
            files.append(f)
    return {"files": sorted(files)}





if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
