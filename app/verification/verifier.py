from app.verification.judge import LLMJudge
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

    def verify_claims(self, claims: list[Claim]) -> VerificationResult:
        verifications: list[ClaimVerification] = []

        for claim in claims:
            retrieval_result = self.retriever.search(claim.claim, mode=RetrievalMode.HIGH_PRECISION)
            evidence = [chunk.content for chunk in retrieval_result.chunks]
            judge_result = self.judge.judge(claim.claim, evidence) or {}

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

    def _safe_confidence(self, value: object) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, confidence))

