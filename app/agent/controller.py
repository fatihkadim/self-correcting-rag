from dataclasses import dataclass

from app.core.logger import get_logger
from app.generation.repair import AnswerRepair, NO_ANSWER_MESSAGE
from app.verification.verifier import VerificationEngine
from app.claims.extractor import ClaimExtractor
from app.retrieval.retriever import QdrantRetriever
from app.generation.answer import AnswerGenerator
from app.generation.llm import LLMClient
from app.core.config import settings
from app.schemas.claims import ClaimExtractionResult
from app.schemas.query import QueryRequest, QueryResponse, AnswerStatus
from app.schemas.retrieval import RetrievalResult
from app.schemas.verification import VerificationResult
from app.agent.policies import PolicyEngine, Decisions
from app.retrieval.query_rewriter import QueryRewriter

logger = get_logger(__name__)


@dataclass
class _Candidate:
    """Kabul edilmemiş ama fallback olarak kullanılabilecek bir cevap."""
    answer: str
    retrieval: RetrievalResult
    claims: ClaimExtractionResult
    verification: VerificationResult


class SelfCorrectionController():
    def __init__(self):
        llm = LLMClient()
        self.retriever = QdrantRetriever()
        self.answer_generator = AnswerGenerator(llm_client=llm)
        self.claim_extractor = ClaimExtractor(llm_client=llm)
        self.verifier = VerificationEngine(retriever=self.retriever, llm_client=llm)
        self.answer_repair = AnswerRepair(llm_client=llm)
        self.decider = PolicyEngine()
        self.query_rewriter = QueryRewriter(llm_client=llm)

    def run(self, request: QueryRequest) -> QueryResponse:
        question = request.question
        max_attempts = max(1, settings.max_retry)
        search_query = question
        candidates: list[_Candidate] = []
        retrieval_result: RetrievalResult | None = None

        for attempt in range(1, max_attempts + 1):
            retrieval_result = self.retriever.search(search_query)
            contexts = [chunk.content for chunk in retrieval_result.chunks]
            answer = self.answer_generator.generate(question, retrieval_result)

            claims_result, verification_result = self._verify(answer, contexts)
            if claims_result is None:
                logger.warning("Claim extraction başarısız; cevap doğrulanmadan dönüyor.")
                return self._response(answer, retrieval_result, None, None,
                                      AnswerStatus.UNVERIFIED, attempt)

            decision = self.decider.decide(verification_result)
            logger.info("attempt=%d decision=%s", attempt, decision.value)

            if decision == Decisions.ACCEPT:
                return self._response(answer, retrieval_result, claims_result, verification_result,
                                      AnswerStatus.VERIFIED, attempt)

            if decision == Decisions.REPAIR:
                repaired = self.answer_repair.repair(question, verification_result, original_answer=answer)
                # Onarılan cevap da doğrulanır; kullanıcıya gösterilen metin ile
                # döndürülen doğrulama sonucu her zaman aynı cevaba ait olur.
                r_claims, r_verification = self._verify(repaired, contexts)
                if r_claims is not None and self._is_acceptable_repair(r_verification):
                    return self._response(repaired, retrieval_result, r_claims, r_verification,
                                          AnswerStatus.REPAIRED, attempt)
                logger.warning("Onarılan cevap doğrulamadan geçemedi.")
                if r_claims is not None:
                    candidates.append(_Candidate(repaired, retrieval_result, r_claims, r_verification))

            candidates.append(_Candidate(answer, retrieval_result, claims_result, verification_result))

            if attempt < max_attempts:
                logger.info("Cevap kabul edilmedi, soru yeniden yazılarak tekrar deneniyor...")
                search_query = self.query_rewriter.rewrite(question, attempt)

        return self._fallback(candidates, retrieval_result, max_attempts)

    def _verify(self, answer: str, contexts: list[str]
                ) -> tuple[ClaimExtractionResult | None, VerificationResult | None]:
        claims_result = self.claim_extractor.extract(answer)
        if claims_result is None:
            return None, None
        return claims_result, self.verifier.verify_claims(claims_result.claims, contexts)

    @staticmethod
    def _is_acceptable_repair(result: VerificationResult) -> bool:
        """Onarılan cevapta çürütülmüş iddia olmamalı ve en az bir iddia doğrulanmış olmalı.

        Hiç iddia içermeyen onarım (ör. "kaynaklarda cevap yok") da kabul edilir.
        """
        if result.refuted_counts > 0:
            return False
        return result.total_claims == 0 or result.supported_counts > 0

    def _fallback(self, candidates: list[_Candidate], last_retrieval: RetrievalResult,
                  attempts: int) -> QueryResponse:
        """Deneme hakları bittiğinde, çürütülmüş iddia içermeyen en iyi adayı döndürür.

        Böyle bir aday yoksa doğrulanmamış bir cevap göstermek yerine
        güvenilir cevap bulunamadığını bildirir.
        """
        safe = [
            c for c in candidates
            if c.verification.refuted_counts == 0 and c.verification.supported_counts > 0
        ]
        if safe:
            best = max(safe, key=lambda c: (
                c.verification.supported_counts / c.verification.total_claims,
                c.verification.overall_confidence,
            ))
            logger.info("Deneme hakları bitti; kısmen doğrulanmış en iyi cevap döndürülüyor.")
            return self._response(best.answer, best.retrieval, best.claims, best.verification,
                                  AnswerStatus.PARTIALLY_VERIFIED, attempts)

        logger.warning("Deneme hakları bitti; güvenilir cevap bulunamadı.")
        return self._response(NO_ANSWER_MESSAGE, last_retrieval, None, None,
                              AnswerStatus.NO_ANSWER, attempts)

    @staticmethod
    def _response(answer: str, retrieval: RetrievalResult,
                  claims: ClaimExtractionResult | None, verification: VerificationResult | None,
                  status: AnswerStatus, attempts: int) -> QueryResponse:
        return QueryResponse(
            answer=answer,
            status=status,
            attempts=attempts,
            sources=[chunk.model_dump() for chunk in retrieval.chunks],
            claims=[c.model_dump() for c in claims.claims] if claims else [],
            verification=verification.model_dump() if verification else None,
        )
