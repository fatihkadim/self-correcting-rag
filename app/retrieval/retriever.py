from app.schemas.retrieval import RetrievalResult, Chunk
import time

class MockRetriever:
    def __init__(self):
        self.chunk1 = Chunk(id="12",content="Python 1991 yılında yaratıldı",source="wiki.txt",similiarty_score=1.45)
        self.chunk2 = Chunk(id="doc_45",content="FASTAPI modern bir framework",source="fastapi.txt",similiarty_score=3.2)
        self.chunk3 = Chunk(id="doc_41",content="Python AI alanında çok güçlü bir dil",source="python.txt",similiarty_score=0.12)
        self.documents = [self.chunk1,self.chunk2,self.chunk3]
    
    def search(self,query:str, top_k:int = 2):
        start = time.time()
        chunks = self.documents[:top_k]
        all_time = (time.time() - start) * 1000

        return RetrievalResult(query=query,chunks=chunks,retrieval_time=all_time)