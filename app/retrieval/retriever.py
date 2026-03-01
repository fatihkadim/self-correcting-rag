from app.schemas.retrieval import RetrievalResult, Chunk
from app.retrieval.embedder import Embedder
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from app.core.config import settings
import time

COLLECTION_NAME="SCR"
class MockRetriever:
    def __init__(self):
        self.chunk1 = Chunk(id="12",content="Python 1991 yılında yaratıldı",source="wiki.txt",similarity_score=1.45)
        self.chunk2 = Chunk(id="doc_45",content="FASTAPI modern bir framework",source="fastapi.txt",similarity_score=3.2)
        self.chunk3 = Chunk(id="doc_41",content="Python AI alanında çok güçlü bir dil",source="python.txt",similarity_score=0.12)
        self.documents = [self.chunk1,self.chunk2,self.chunk3]
    
    def search(self,query:str, top_k:int = 2):
        start = time.time()
        chunks = self.documents[:top_k]
        all_time = (time.time() - start) * 1000

        return RetrievalResult(query=query,chunks=chunks,retrieval_time=all_time)


class QdrantRetriever:
    def __init__(self):
        self.embedder = Embedder()

        self.client = QdrantClient(url=settings.qdrant_url)
        
        # Sadece koleksiyon yoksa oluştur ve mock verileri yükle
        if not self.client.collection_exists(COLLECTION_NAME):
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
            mock = MockRetriever()
            texts = [chunk.content for chunk in mock.documents]
            vectors = self.embedder.embed(texts)

            self.client.upsert(
                collection_name=COLLECTION_NAME,
                points=[
                    PointStruct(
                        id=1,
                        payload={
                            "content": mock.documents[0].content,
                            "source": mock.documents[0].source,
                        },
                        vector=vectors[0]
                    ),
                    PointStruct(
                        id=2,
                        payload={
                            "content": mock.documents[1].content,
                            "source": mock.documents[1].source,
                        },
                        vector=vectors[1]
                    ),
                    PointStruct(
                        id=3,
                        payload={
                            "content": mock.documents[2].content,
                            "source": mock.documents[2].source,
                        },
                        vector=vectors[2]
                    ),
                ],
            )



    def search(self,query:str, top_k:int=2):
        start = time.time()
        encoded_query = self.embedder.embed_single(query)
        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=encoded_query,
            limit=top_k,
        ).points

        all_time = (time.time() - start) * 1000
        chunks = [Chunk(id=str(r.id),content=r.payload["content"],source=r.payload["source"],similarity_score=r.score) for r in results]

        return RetrievalResult(query=query,chunks=chunks,retrieval_time=all_time)



