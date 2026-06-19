from app.generation.llm import LLMClient
from app.retrieval.retriever import QdrantRetriever
from app.schemas.query import QueryRequest,QueryResponse
from app.core.logger import get_logger
from app.generation.answer import AnswerGenerator
from app.claims.extractor import ClaimExtractor
from app.verification.verifier import VerificationEngine
from app.generation.repair import AnswerRepair
logger = get_logger(__name__)

class RAGPipeline:
    def __init__(self):
        self.retriever = QdrantRetriever()
        self.answer_generator = AnswerGenerator(llm_client=LLMClient())
        self.claim_extractor = ClaimExtractor(llm_client=LLMClient())
        self.verifier = VerificationEngine(retriever=self.retriever)
        self.answer_repair = AnswerRepair(llm_client=LLMClient())

    def run(self, request: QueryRequest) -> QueryResponse:
        logger.info(f"Yeni soru: {request.question}")
        retrieval_result = self.retriever.search(request.question)
        logger.info(f"Retrieval bitti. {len(retrieval_result.chunks)} parça bilgi bulundu")
        answer = self.answer_generator.generate(request.question, retrieval_result)
        claims_result = self.claim_extractor.extract(answer)
        claims_list = [c.model_dump() for c in claims_result.claims] if claims_result else []
        original_contexts = [chunk.content for chunk in retrieval_result.chunks]
        verification_result = self.verifier.verify_claims(claims_result.claims, original_contexts) if claims_result else None
        
        final_answer = answer 
        
        if verification_result is not None:
            if verification_result.supported_counts < verification_result.total_claims:
                logger.info("Bazı iddialar doğrulanamadı. Cevap onarılıyor (Answer Repair)...")
                final_answer = self.answer_repair.repair(
                    question=request.question, 
                    verification_result=verification_result,
                    original_answer=answer,
                    original_contexts=original_contexts
                )
            else:
                logger.info("Tüm iddialar doğrulandı. Onarıma (Repair) gerek yok.")
        else:
            logger.warning("Verification result alınamadı.")

        return QueryResponse(
            answer=final_answer,
            sources=[chunk.model_dump() for chunk in retrieval_result.chunks],
            claims=claims_list,
            verification=verification_result.model_dump() if verification_result else None,
        )
