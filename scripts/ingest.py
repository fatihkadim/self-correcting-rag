from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from app.retrieval.chunking import TextChunker
from app.retrieval.embedder import Embedder
from app.core.config import settings

COLLECTION_NAME = "SCR2"
client = QdrantClient(url=settings.qdrant_url)

RAW_DIR = Path("data/raw")
txt_files = list(RAW_DIR.glob("*.txt"))

embedder = Embedder()
chunker = TextChunker(chunk_size=500, chunk_overlap=50)


def ingest_documents():
    if not txt_files:
        print("data/raw/ içinde .txt dosyası bulunamadı!")
        return

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

    all_chunks = []
    for doc in txt_files:
        raw_text = doc.read_text(encoding="utf-8")
        chunks = chunker.split(raw_text, source=doc.name)
        all_chunks.extend(chunks)

    texts = [chunk.content for chunk in all_chunks]
    embedded_docs = embedder.embed(texts)

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=i,
                payload={
                    "content": all_chunks[i].content,
                    "source": all_chunks[i].source
                },
                vector=embedded_docs[i]
            )
            for i in range(len(all_chunks))
        ]
    )
    print(f"{len(all_chunks)} chunk yüklendi.")


if __name__ == "__main__":
    ingest_documents()
