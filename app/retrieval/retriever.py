from app.schemas.retrieval import RetrievalResult, Chunk
from app.retrieval.embedder import get_embedder
from app.retrieval.reranker import get_reranker
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
        self.embedder = get_embedder()
        self.client = QdrantClient(url=settings.qdrant_url)
        self.reranker = get_reranker()
            

    def search(self,query:str,mode=RetrievalMode.HIGH_RECALL):
        
        if mode == RetrievalMode.HIGH_RECALL:
            top_k_retrieval = 25
            top_k_final = 10
            threshold = 0.25
        else:
            top_k_retrieval = 15
            top_k_final = 3
            threshold = 0.35
        
        start = time.time()
        encoded_query = self.embedder.embed_single(query)
        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=encoded_query,
            limit=top_k_retrieval,
            score_threshold=threshold
        ).points

        chunks = [Chunk(id=str(r.id),content=r.payload["content"],source=r.payload["source"],similarity_score=r.score) for r in results]

        # Rerank the retrieved chunks
        reranked_chunks = self.reranker.rerank(query, chunks, top_k=top_k_final)

        all_time = (time.time() - start) * 1000

        return RetrievalResult(query=query,chunks=reranked_chunks,retrieval_time=all_time)

    def rank_texts(self, query: str, texts: list[str], top_k: int) -> list[str]:
        """Verilen metinleri cross-encoder ile sorguya göre sıralayıp ilk top_k'yı döndürür."""
        return self.reranker.rank_texts(query, texts, top_k)



