from app.generation.llm import LLMClient
from app.retrieval.retriever import QdrantRetriever
from app.schemas.query import QueryRequest,QueryResponse
from app.core.logger import get_logger
from app.generation.answer import AnswerGenerator
from app.claims.extractor import ClaimExtractor
logger = get_logger(__name__)

class RAGPipeline:
    def __init__(self):
        self.retriever = QdrantRetriever()
        self.answer_generator = AnswerGenerator(llm_client=LLMClient())
        self.claim_extractor = ClaimExtractor(llm_client=LLMClient())
    def run(self, request: QueryRequest) -> QueryResponse:
        logger.info(f"Yeni soru: {request.question}")
        retrieval_result = self.retriever.search(request.question)
        logger.info(f"Retrieval bitti. {len(retrieval_result.chunks)} parça bilgi bulundu")
        answer = self.answer_generator.generate(request.question, retrieval_result)
        claims = self.claim_extractor.extract(answer)
        return QueryResponse(
            answer=answer,
            sources=[chunk.model_dump() for chunk in retrieval_result.chunks],
            claims=claims,
        )
