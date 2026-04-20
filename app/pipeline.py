from app.generation.llm import LLMClient
from app.retrieval.retriever import QdrantRetriever
from app.schemas.query import QueryRequest,QueryResponse
from app.core.logger import get_logger
from app.generation.answer import AnswerGenerator
from app.claims.extractor import ClaimExtractor
from app.verification.verifier import VerificationEngine
logger = get_logger(__name__)

class RAGPipeline:
    def __init__(self):
        self.retriever = QdrantRetriever()
        self.answer_generator = AnswerGenerator(llm_client=LLMClient())
        self.claim_extractor = ClaimExtractor(llm_client=LLMClient())
        self.verifier = VerificationEngine(retriever=self.retriever)

    def run(self, request: QueryRequest) -> QueryResponse:
        logger.info(f"Yeni soru: {request.question}")
        retrieval_result = self.retriever.search(request.question)
        logger.info(f"Retrieval bitti. {len(retrieval_result.chunks)} parça bilgi bulundu")
        answer = self.answer_generator.generate(request.question, retrieval_result)
        claims_result = self.claim_extractor.extract(answer)
        claims_list = [c.model_dump() for c in claims_result.claims] if claims_result else []
        verification_result = self.verifier.verify_claims(claims_result.claims) if claims_result else None
        return QueryResponse(
            answer=answer,
            sources=[chunk.model_dump() for chunk in retrieval_result.chunks],
            claims=claims_list,
            verification=verification_result.model_dump() if verification_result else None,
        )
