from sentence_transformers import CrossEncoder
from app.schemas.retrieval import Chunk
from typing import List

class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        # Use a small and fast cross-encoder for reranking
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: List[Chunk], top_k: int = 5) -> List[Chunk]:
        if not chunks:
            return []
            
        pairs = [[query, chunk.content] for chunk in chunks]
        scores = self.model.predict(pairs)
        
        # Add rerank scores to chunks and sort
        for chunk, score in zip(chunks, scores):
            chunk.similarity_score = float(score)  # Overwrite similarity score with rerank score
            
        chunks.sort(key=lambda x: x.similarity_score, reverse=True)
        return chunks[:top_k]
