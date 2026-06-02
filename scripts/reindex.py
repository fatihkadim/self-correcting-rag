import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient
from app.core.config import settings
from scripts.ingest import ingest_documents

def reindex_all():
    client = QdrantClient(url=settings.qdrant_url)
    collection_name = "SCR2"
    
    # 1. Koleksiyonu tamamen sil (içindeki bozuk verilerden kurtul)
    if client.collection_exists(collection_name):
        print(f"Koleksiyon '{collection_name}' Qdrant'tan siliniyor...")
        client.delete_collection(collection_name=collection_name)
    
    # 2. Tüm dosyaları (yeni PyMuPDF mantığı ile) yeniden ingest et
    print("PyMuPDF (fitz) ile tüm dosyalar yeniden indeksleniyor...")
    ingest_documents()
    print("İşlem tamamlandı!")

if __name__ == "__main__":
    reindex_all()
