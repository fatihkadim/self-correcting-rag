from app.core.logger import get_logger
from app.generation.repair import AnswerRepair
from app.verification.verifier import VerificationEngine
from app.claims.extractor import ClaimExtractor
from app.retrieval.retriever import QdrantRetriever
from app.generation.answer import AnswerGenerator
from app.generation.llm import LLMClient
from app.agent.policies import PolicyEngine
from app.core.config import settings
from app.schemas.query import QueryRequest, QueryResponse
from app.agent.policies import PolicyEngine, Decisions

logger = get_logger(__name__)

class SelfCorrectionController():
    def __init__(self):
        self.retriever = QdrantRetriever()
        self.answer_generator = AnswerGenerator(llm_client=LLMClient())
        self.claim_extractor = ClaimExtractor(llm_client=LLMClient())
        self.verifier = VerificationEngine(retriever=self.retriever)
        self.answer_repair = AnswerRepair(llm_client=LLMClient())
        self.decider = PolicyEngine()

    def run(self,request:QueryRequest) -> QueryResponse:
        attempt= 0
        while attempt < settings.max_retry:
            retrieval_result = self.retriever.search(request.question)
            answer = self.answer_generator.generate(request.question, retrieval_result)

            claims_result = self.claim_extractor.extract(answer)
            claims_list = [c.model_dump() for c in claims_result.claims] if claims_result else []
            verification_result = self.verifier.verify_claims(claims_result.claims) if claims_result else None

            if verification_result is None:
                return QueryResponse(answer=answer,
                                    sources=[chunk.model_dump() for chunk in retrieval_result.chunks],
                                    claims=claims_list,
                                    verification=None,
                                    )

            decision = self.decider.decide(verification_result)    
            if decision == Decisions.ACCEPT:
                return QueryResponse(answer=answer,
                                    sources=[chunk.model_dump() for chunk in retrieval_result.chunks],
                                    claims=claims_list,
                                    verification=verification_result.model_dump() if verification_result else None,
                                    )
            elif decision == Decisions.REPAIR:
                repaired_answer = self.answer_repair.repair(request.question,verification_result)
                return QueryResponse(answer=repaired_answer,
                                    sources=[chunk.model_dump() for chunk in retrieval_result.chunks],
                                    claims=claims_list,
                                    verification=verification_result.model_dump() if verification_result else None,
                                    )
            elif decision == Decisions.RETRY:
                logger.info("Cevap yanlislandi, tekrar deneniyor...")

            attempt += 1
        return QueryResponse(answer=answer,
                                    sources=[chunk.model_dump() for chunk in retrieval_result.chunks],
                                    claims=claims_list,
                                    verification=verification_result.model_dump() if verification_result else None,
                                    )
            
