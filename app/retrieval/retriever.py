from app.schemas.retrieval import RetrievalResult, Chunk
from app.retrieval.embedder import Embedder
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from app.core.config import settings
from enum import Enum
import time

COLLECTION_NAME="SCR2"
class RetrievalMode(str,Enum):
    HIGH_RECALL = "high_recall"
    HIGH_PRECISION = "high_precision"

class QdrantRetriever:
    def __init__(self):
        self.embedder = Embedder()
        self.client = QdrantClient(url=settings.qdrant_url)
            

    def search(self,query:str,mode=RetrievalMode.HIGH_RECALL):
        
        if mode == RetrievalMode.HIGH_RECALL:
            top_k = 10
            threshold = 0.3
        else:
            top_k = 3
            threshold = 0.4
        
        start = time.time()
        encoded_query = self.embedder.embed_single(query)
        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=encoded_query,
            limit=top_k,
            score_threshold=threshold
        ).points

        all_time = (time.time() - start) * 1000
        chunks = [Chunk(id=str(r.id),content=r.payload["content"],source=r.payload["source"],similarity_score=r.score) for r in results]

        return RetrievalResult(query=query,chunks=chunks,retrieval_time=all_time)



