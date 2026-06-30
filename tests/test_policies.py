from app.agent.policies import PolicyEngine, Decisions
from app.schemas.verification import (
    ClaimVerification,
    VerificationResult,
    VerificationStatus,
)


def _make_result(
    supported: int = 0, refuted: int = 0, unknown: int = 0
) -> VerificationResult:
    """Belirli sayıda supported/refuted/unknown claim içeren VerificationResult üretir."""
    verifications = []
    for _ in range(supported):
        verifications.append(
            ClaimVerification(
                claim="supported claim",
                status=VerificationStatus.SUPPORTED,
                confidence=0.9,
                evidence=["some evidence"],
                reasoning="ok",
            )
        )
    for _ in range(refuted):
        verifications.append(
            ClaimVerification(
                claim="refuted claim",
                status=VerificationStatus.REFUTED,
                confidence=0.8,
                evidence=["contradicting evidence"],
                reasoning="wrong",
            )
        )
    for _ in range(unknown):
        verifications.append(
            ClaimVerification(
                claim="unknown claim",
                status=VerificationStatus.UNKNOWN,
                confidence=0.3,
                evidence=[],
                reasoning="no info",
            )
        )
    total = supported + refuted + unknown
    overall = (
        sum(v.confidence for v in verifications) / total if total else 0.0
    )
    return VerificationResult(
        verifications=verifications,
        total_claims=total,
        supported_counts=supported,
        refuted_counts=refuted,
        unknown_counts=unknown,
        overall_confidence=overall,
    )


engine = PolicyEngine()


# ---------------------------------------------------------------------------
# ACCEPT senaryoları
# ---------------------------------------------------------------------------

def test_accept_when_all_claims_supported():
    """Tüm claim'ler supported → ACCEPT."""
    result = _make_result(supported=3)
    assert engine.decide(result) == Decisions.ACCEPT


def test_accept_when_zero_claims():
    """Hiç claim yoksa → ACCEPT (boş sonuç)."""
    result = _make_result()  # 0 claim
    assert engine.decide(result) == Decisions.ACCEPT


# ---------------------------------------------------------------------------
# RETRY senaryoları
# ---------------------------------------------------------------------------

def test_retry_when_zero_supported():
    """Sıfır supported (hepsi refuted/unknown) → RETRY."""
    result = _make_result(refuted=2, unknown=1)
    assert engine.decide(result) == Decisions.RETRY


def test_retry_when_all_refuted():
    """Tüm claim'ler refuted → RETRY."""
    result = _make_result(refuted=3)
    assert engine.decide(result) == Decisions.RETRY


def test_retry_when_all_unknown():
    """Tüm claim'ler unknown → RETRY."""
    result = _make_result(unknown=2)
    assert engine.decide(result) == Decisions.RETRY


# ---------------------------------------------------------------------------
# REPAIR senaryoları
# ---------------------------------------------------------------------------

def test_repair_when_partial_support():
    """Kısmi supported → REPAIR."""
    result = _make_result(supported=2, refuted=1)
    assert engine.decide(result) == Decisions.REPAIR


def test_repair_with_mixed_results():
    """Supported + refuted + unknown karışımı → REPAIR."""
    result = _make_result(supported=1, refuted=1, unknown=1)
    assert engine.decide(result) == Decisions.REPAIR


def test_repair_when_supported_and_unknown():
    """Supported + unknown (refuted yok) → REPAIR."""
    result = _make_result(supported=2, unknown=1)
    assert engine.decide(result) == Decisions.REPAIR


# ---------------------------------------------------------------------------
# CONFIDENCE THRESHOLD senaryoları
# ---------------------------------------------------------------------------

def test_accept_high_confidence_mostly_supported():
    """5 claim'den 4'ü yüksek güvenle supported, 1 unknown → ACCEPT (≥80% threshold)."""
    verifications = []
    for _ in range(4):
        verifications.append(
            ClaimVerification(
                claim="high conf supported",
                status=VerificationStatus.SUPPORTED,
                confidence=0.9,
                evidence=["ev"],
                reasoning="ok",
            )
        )
    verifications.append(
        ClaimVerification(
            claim="unknown claim",
            status=VerificationStatus.UNKNOWN,
            confidence=0.3,
            evidence=[],
            reasoning="no info",
        )
    )
    result = VerificationResult(
        verifications=verifications,
        total_claims=5,
        supported_counts=4,
        refuted_counts=0,
        unknown_counts=1,
        overall_confidence=0.78,
    )
    assert engine.decide(result) == Decisions.ACCEPT


def test_repair_low_confidence_supported():
    """3 claim'den 2'si düşük güvenle supported, 1 unknown → REPAIR (threshold altında)."""
    verifications = []
    for _ in range(2):
        verifications.append(
            ClaimVerification(
                claim="low conf supported",
                status=VerificationStatus.SUPPORTED,
                confidence=0.5,  # threshold (0.7) altında
                evidence=["ev"],
                reasoning="ok",
            )
        )
    verifications.append(
        ClaimVerification(
            claim="unknown claim",
            status=VerificationStatus.UNKNOWN,
            confidence=0.3,
            evidence=[],
            reasoning="no info",
        )
    )
    result = VerificationResult(
        verifications=verifications,
        total_claims=3,
        supported_counts=2,
        refuted_counts=0,
        unknown_counts=1,
        overall_confidence=0.43,
    )
    assert engine.decide(result) == Decisions.REPAIR

