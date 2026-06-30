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
from app.retrieval.query_rewriter import QueryRewriter

logger = get_logger(__name__)

class SelfCorrectionController():
    def __init__(self):
        self.retriever = QdrantRetriever()
        self.answer_generator = AnswerGenerator(llm_client=LLMClient())
        self.claim_extractor = ClaimExtractor(llm_client=LLMClient())
        self.verifier = VerificationEngine(retriever=self.retriever)
        self.answer_repair = AnswerRepair(llm_client=LLMClient())
        self.decider = PolicyEngine()
        self.query_rewriter = QueryRewriter(llm_client=LLMClient())

    def run(self,request:QueryRequest) -> QueryResponse:
        attempt= 0
        current_question = request.question
        while attempt < settings.max_retry:
            retrieval_result = self.retriever.search(current_question)
            answer = self.answer_generator.generate(request.question, retrieval_result)

            claims_result = self.claim_extractor.extract(answer)
            claims_list = [c.model_dump() for c in claims_result.claims] if claims_result else []
            # Orijinal context'leri verification'a geçir (Problem 1)
            original_contexts = [chunk.content for chunk in retrieval_result.chunks]
            verification_result = self.verifier.verify_claims(claims_result.claims, original_contexts) if claims_result else None

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
                repaired_answer = self.answer_repair.repair(
                    request.question, verification_result,
                    original_answer=answer,
                    original_contexts=original_contexts
                )
                # Güvenlik kontrolü: repair sonucu orijinalden kötüyse geri al
                if self._is_degraded(repaired_answer, answer):
                    logger.warning("Repair cevabı bozdu, orijinal cevap korunuyor.")
                    repaired_answer = answer
                return QueryResponse(answer=repaired_answer,
                                    sources=[chunk.model_dump() for chunk in retrieval_result.chunks],
                                    claims=claims_list,
                                    verification=verification_result.model_dump() if verification_result else None,
                                    )
            elif decision == Decisions.RETRY:
                logger.info("Cevap yanlislandi, tekrar deneniyor...")
                # Soruyu reformüle et (Problem 2)
                current_question = self.query_rewriter.rewrite(request.question, attempt)

            attempt += 1
        return QueryResponse(answer=answer,
                                    sources=[chunk.model_dump() for chunk in retrieval_result.chunks],
                                    claims=claims_list,
                                    verification=verification_result.model_dump() if verification_result else None,
                                    )

    def _is_degraded(self, new_answer: str, original_answer: str) -> bool:
        """Checks whether the answer has degraded after repair."""
        fallback_phrases = [
            # Turkish fallbacks
            "cevap bulunamamıştır",
            "güvenilir bir kanıt bulunamadı",
            "cevap bulunamadı",
            "yeterli bilgi yoktur",
            # English fallbacks
            "do not contain an answer",
            "no reliable evidence was found",
            "insufficient information",
            "cannot be answered",
        ]
        new_lower = new_answer.lower()
        orig_lower = original_answer.lower()
        # Orijinal zaten fallback ise, repair'ı bozmayız
        orig_is_fallback = any(phrase in orig_lower for phrase in fallback_phrases)
        if orig_is_fallback:
            return False
        # Repair sonucu fallback olduysa → bozulmuş
        if any(phrase in new_lower for phrase in fallback_phrases):
            return True
        # Repair sonucu orijinalden çok kısaysa → bozulmuş
        if len(new_answer.strip()) < len(original_answer.strip()) * 0.3:
            return True
        return False

            
