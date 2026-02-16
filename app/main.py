from fastapi import FastAPI
import time
from schemas import QueryRequest,QueryResponse
from retrieval import MockRetriever

app = FastAPI(title="Self Correction RAG")
 
@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/health")
def check_health():
    return {
        "status":"heatlhy",
        "server time": time.time()
    }
@app.post("/query",response_model=QueryResponse)
def query(req: QueryRequest):
    retriever = MockRetriever()
    result = retriever.search(req.question)
    contents = "\n".join([c.content for c in result.chunks])
    return QueryResponse(answer=contents)





if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
