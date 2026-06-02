import sys
import os
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from app.retrieval.chunking import TextChunker
from app.retrieval.embedder import Embedder
from app.core.config import settings
import fitz
import uuid

COLLECTION_NAME = "SCR2"
client = QdrantClient(url=settings.qdrant_url)

RAW_DIR = Path("data/raw")

embedder = Embedder()
chunker = TextChunker(chunk_size=1500, chunk_overlap=200)


def ingest_documents(target_files=None):
    if target_files is None:
        txt_files = list(RAW_DIR.glob("*.txt"))
        pdf_files = list(RAW_DIR.glob("*.pdf"))
        target_files = txt_files + pdf_files

    if not target_files:
        print("data/raw/ içinde .txt veya .pdf dosyası bulunamadı!")
        return

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

    all_chunks = []
    for doc in target_files:
        print(f"İşleniyor: {doc.name}")
        raw_text = ""
        
        if doc.suffix.lower() == '.txt':
            raw_text = doc.read_text(encoding="utf-8", errors="ignore")
        elif doc.suffix.lower() == '.pdf':
            try:
                doc_pdf = fitz.open(str(doc))
                for page in doc_pdf:
                    text = page.get_text()
                    if text:
                        raw_text += text + "\n"
                doc_pdf.close()
            except Exception as e:
                print(f"PDF okuma hatası {doc.name}: {e}")
                continue

        if not raw_text.strip():
            continue

        chunks = chunker.split(raw_text, source=doc.name)
        all_chunks.extend(chunks)

    if not all_chunks:
        print("Çıkarılabilir metin bulunamadı!")
        return

    texts = [chunk.content for chunk in all_chunks]
    embedded_docs = embedder.embed(texts)

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=uuid.uuid4().hex,
                payload={
                    "content": all_chunks[i].content,
                    "source": all_chunks[i].source
                },
                vector=embedded_docs[i]
            )
            for i in range(len(all_chunks))
        ]
    )
    print(f"{len(all_chunks)} chunk Qdrant'a başarıyla yüklendi.")


if __name__ == "__main__":
    ingest_documents()
