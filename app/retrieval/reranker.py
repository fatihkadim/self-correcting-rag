from functools import lru_cache

from sentence_transformers import CrossEncoder
from app.schemas.retrieval import Chunk

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL):
        # Use a small and fast cross-encoder for reranking
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: list[Chunk], top_k: int = 5) -> list[Chunk]:
        if not chunks:
            return []

        pairs = [[query, chunk.content] for chunk in chunks]
        scores = self.model.predict(pairs)

        # Add rerank scores to chunks and sort
        for chunk, score in zip(chunks, scores):
            chunk.similarity_score = float(score)  # Overwrite similarity score with rerank score

        chunks.sort(key=lambda x: x.similarity_score, reverse=True)
        return chunks[:top_k]

    def rank_texts(self, query: str, texts: list[str], top_k: int) -> list[str]:
        """Metinleri sorguya göre sıralar ve en alakalı top_k tanesini döndürür."""
        if not texts:
            return []
        scores = self.model.predict([[query, t] for t in texts])
        ranked = sorted(zip(texts, scores), key=lambda x: float(x[1]), reverse=True)
        return [t for t, _ in ranked[:top_k]]


@lru_cache(maxsize=None)
def get_reranker(model_name: str = DEFAULT_RERANKER_MODEL) -> Reranker:
    """Süreç genelinde tek bir Reranker örneği."""
    return Reranker(model_name)
