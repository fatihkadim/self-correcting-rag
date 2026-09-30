from concurrent.futures import ThreadPoolExecutor

from app.verification.judge import LLMJudge
from app.core.config import settings
from app.generation.llm import LLMClient
from app.retrieval.retriever import QdrantRetriever, RetrievalMode
from app.schemas.claims import Claim
from app.schemas.verification import (
    ClaimVerification,
    VerificationResult,
    VerificationStatus,
)
from app.core.logger import get_logger


class VerificationEngine:
    def __init__(
        self,
        retriever: QdrantRetriever | None = None,
        judge: LLMJudge | None = None,
        llm_client: LLMClient | None = None,
    ):
        self.retriever = retriever or QdrantRetriever()
        self.judge = judge or LLMJudge(llm_client or LLMClient())
        self.logger = get_logger(__name__)

    def verify_claims(self, claims: list[Claim], original_contexts: list[str] | None = None) -> VerificationResult:
        verifications: list[ClaimVerification] = []
        claim_texts = [claim.claim for claim in claims]

        # Kanıt toplama sıralı (yerel embedder/reranker hızlı); süreyi belirleyen
        # LLM judge çağrıları paralel yapılır. map() sırayı korur.
        evidences = [self._gather_evidence(text, original_contexts) for text in claim_texts]
        if claims:
            workers = min(settings.verify_max_workers, len(claims))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                judge_results = list(pool.map(self.judge.judge, claim_texts, evidences))
        else:
            judge_results = []

        for claim, evidence, judge_result in zip(claims, evidences, judge_results):
            judge_result = judge_result or {}

            status = str(judge_result.get("status", "unknown")).lower()
            if status not in {"supported", "refuted", "unknown"}:
                status = "unknown"

            self.logger.info(
                "verified claim=%s evidence_count=%s status=%s",
                claim.claim,
                len(evidence),
                status,
            )

            verifications.append(
                ClaimVerification(
                    claim=claim.claim,
                    status=VerificationStatus(status),
                    confidence=self._safe_confidence(judge_result.get("confidence")),
                    evidence=evidence,
                    reasoning=str(judge_result.get("reasoning", "")),
                )
            )

        total_claims = len(verifications)
        supported_counts = sum(v.status == VerificationStatus.SUPPORTED for v in verifications)
        refuted_counts = sum(v.status == VerificationStatus.REFUTED for v in verifications)
        unknown_counts = sum(v.status == VerificationStatus.UNKNOWN for v in verifications)
        overall_confidence = (
            sum(v.confidence for v in verifications) / total_claims if total_claims else 0.0
        )

        return VerificationResult(
            verifications=verifications,
            total_claims=total_claims,
            supported_counts=supported_counts,
            refuted_counts=refuted_counts,
            unknown_counts=unknown_counts,
            overall_confidence=overall_confidence,
        )

    def _gather_evidence(self, claim: str, original_contexts: list[str] | None) -> list[str]:
        """Claim'e özel arama sonuçları + orijinal context'ler; sınırı aşarsa
        claim'e en alakalı max_evidence_per_claim parça tutulur (judge prompt'u küçülür)."""
        retrieval_result = self.retriever.search(claim, mode=RetrievalMode.HIGH_PRECISION)
        evidence = [chunk.content for chunk in retrieval_result.chunks]
        for ctx in original_contexts or []:
            if ctx not in evidence:
                evidence.append(ctx)

        limit = settings.max_evidence_per_claim
        if len(evidence) > limit:
            evidence = self.retriever.rank_texts(claim, evidence, top_k=limit)
        return evidence

    def _safe_confidence(self, value: object) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, confidence))

