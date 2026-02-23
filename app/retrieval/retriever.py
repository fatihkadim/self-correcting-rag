from app.schemas.retrieval import RetrievalResult, Chunk
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import time

COLLECTION_NAME="SCR"
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


class QdrantRetriever:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.client = QdrantClient(":memory:")
        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        mock = MockRetriever()
        texts = [chunk.content for chunk in mock.documents]
        vectors = self.model.encode(texts).tolist()

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
        encoded_query = self.model.encode(query).tolist()
        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=encoded_query,
            limit=top_k,
        ).points

        all_time = (time.time() - start) * 1000
        chunks = [Chunk(id=str(r.id),content=r.payload["content"],source=r.payload["source"],similarity_score=r.score) for r in results]

        return RetrievalResult(query=query,chunks=chunks,retrieval_time=all_time)
