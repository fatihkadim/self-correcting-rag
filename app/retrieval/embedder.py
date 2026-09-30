from functools import lru_cache

from sentence_transformers import SentenceTransformer

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class Embedder:
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, batch_size=32).tolist()

    def embed_single(self, text: str) -> list[float]:
        return self.model.encode(text).tolist()


@lru_cache(maxsize=None)
def get_embedder(model_name: str = DEFAULT_EMBEDDING_MODEL) -> Embedder:
    """Süreç genelinde tek bir Embedder örneği (model belleğe bir kez yüklenir)."""
    return Embedder(model_name)
