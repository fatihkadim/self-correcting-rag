from pydantic import BaseModel

class Chunk(BaseModel):
    id:str
    content:str
    source:str
    similiarty_score:float | None = None

class RetrievalResult(BaseModel):
    query:str
    chunks:list[Chunk]
    retrieval_time:float


