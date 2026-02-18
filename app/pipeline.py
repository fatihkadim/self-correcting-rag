from app.retrieval.retriever import MockRetriever
from app.schemas.query import QueryRequest,QueryResponse
from app.core.logger import get_logger

logger = get_logger(__name__)

class RAGPipeline:
    def __init__(self):
        self.retriever = MockRetriever()

    
    def run(self,request: QueryRequest) -> QueryResponse:
        logger.info("New question")
        retrieval_result = self.retriever.search(request.question)
        logger.info(f"Retrievel bitti. {len(retrieval_result.chunks)} parça bilgi bulundu")        
        context = "\n".join([chunk.content for chunk in retrieval_result.chunks])
        logger.info(f"Buldugum bilgiler şunlar: {context}")
        answer = f"Bulunan bilgilere göre cevap: {context[:40]}"
        return QueryResponse(
            answer=answer,
            sources=[chunk.model_dump() for chunk in retrieval_result.chunks])
        