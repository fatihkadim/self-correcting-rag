from fastapi import FastAPI
import time
from app.schemas.query import QueryRequest, QueryResponse
from app.agent.controller import SelfCorrectionController

pipeline = SelfCorrectionController()

app = FastAPI(title="Self Correction RAG")
 
@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/health")
def check_health():
    return {
        "status":"healthy",
        "server time": time.time()
    }
@app.post("/query",response_model=QueryResponse)
def query(req: QueryRequest):
    return pipeline.run(request=req)




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
