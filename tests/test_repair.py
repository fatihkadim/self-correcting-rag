from app.generation.repair import AnswerRepair
from app.schemas.verification import (
    ClaimVerification,
    VerificationResult,
    VerificationStatus,
)


# ---------------------------------------------------------------------------
# Fake LLM
# ---------------------------------------------------------------------------

class FakeLLM:
    def __init__(self, response: str = "Onarılmış cevap."):
        self.response = response
        self.call_count = 0

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        self.call_count += 1
        return self.response


# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------

def _make_verification_result(
    statuses: list[VerificationStatus],
    evidences: list[list[str]] | None = None,
) -> VerificationResult:
    evidences = evidences or [["kanıt"] for _ in statuses]
    verifications = [
        ClaimVerification(
            claim=f"claim-{i}",
            status=s,
            confidence=0.9 if s == VerificationStatus.SUPPORTED else 0.4,
            evidence=evidences[i],
            reasoning="reasoning",
        )
        for i, s in enumerate(statuses)
    ]
    total = len(verifications)
    supported = sum(1 for v in verifications if v.status == VerificationStatus.SUPPORTED)
    refuted = sum(1 for v in verifications if v.status == VerificationStatus.REFUTED)
    unknown = total - supported - refuted
    confidence = sum(v.confidence for v in verifications) / total if total else 0.0
    return VerificationResult(
        verifications=verifications,
        total_claims=total,
        supported_counts=supported,
        refuted_counts=refuted,
        unknown_counts=unknown,
        overall_confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Test 1 — Supported claim'ler varsa LLM'e gönderir
# ---------------------------------------------------------------------------

def test_repair_calls_llm_when_supported_claims_exist():
    """En az bir supported claim varsa LLM çağrılmalı."""
    fake_llm = FakeLLM("Düzeltilmiş cevap.")
    repair = AnswerRepair(llm_client=fake_llm)

    vr = _make_verification_result(
        [VerificationStatus.SUPPORTED, VerificationStatus.REFUTED]
    )
    result = repair.repair("Soru?", vr, original_answer="Orijinal", original_contexts=["ctx"])

    assert result == "Düzeltilmiş cevap."
    assert fake_llm.call_count == 1


# ---------------------------------------------------------------------------
# Test 2 — Hiç supported claim yoksa fallback mesaj döner
# ---------------------------------------------------------------------------

def test_repair_returns_fallback_when_no_supported_claims():
    """Tüm claim'ler refuted/unknown ise LLM çağrılmadan uyarı mesajı döner."""
    fake_llm = FakeLLM("Bu çağrılmamalı.")
    repair = AnswerRepair(llm_client=fake_llm)

    vr = _make_verification_result(
        [VerificationStatus.REFUTED, VerificationStatus.UNKNOWN]
    )
    result = repair.repair("Soru?", vr)

    assert "güvenilir" in result.lower() or "kanıt" in result.lower()
    assert fake_llm.call_count == 0


# ---------------------------------------------------------------------------
# Test 3 — Evidence'lar düzgün toplanıyor mu
# ---------------------------------------------------------------------------

def test_repair_collects_evidence_from_supported_claims_only():
    """Sadece supported claim'lerin evidence'ları kullanılmalı."""
    fake_llm = FakeLLM("Onarılmış.")
    repair = AnswerRepair(llm_client=fake_llm)

    vr = _make_verification_result(
        [VerificationStatus.SUPPORTED, VerificationStatus.REFUTED],
        evidences=[["doğru kanıt"], ["yanlış kanıt"]],
    )
    result = repair.repair("Soru?", vr)

    # LLM'e gönderilen prompt'ta "doğru kanıt" olmalı
    assert fake_llm.call_count == 1
