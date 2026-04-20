from app.schemas.claims import Claim, ClaimType
from app.schemas.verification import VerificationStatus
from app.verification.verifier import VerificationEngine


# ---------------------------------------------------------------------------
# Fake bağımlılıklar
# ---------------------------------------------------------------------------

class FakeChunk:
    def __init__(self, content: str):
        self.content = content


class FakeResult:
    def __init__(self, *contents: str):
        self.chunks = [FakeChunk(c) for c in contents]


class FakeRetriever:
    """Her zaman bir kanıt döndüren sahte retriever."""

    def search(self, query, mode=None):
        return FakeResult(f"evidence for {query}")


class EmptyRetriever:
    """Hiç kanıt döndürmeyen sahte retriever (boş evidence senaryosu)."""

    def search(self, query, mode=None):
        return FakeResult()


class FakeJudge:
    """Her zaman 'supported' döndüren sahte judge."""

    def judge(self, claim, evidence):
        return {"status": "supported", "confidence": 0.9, "reasoning": "matches the evidence"}


class RefutedJudge:
    """Her zaman 'refuted' döndüren sahte judge."""

    def judge(self, claim, evidence):
        return {"status": "refuted", "confidence": 0.85, "reasoning": "contradicts the evidence"}


class UnknownJudge:
    """Her zaman 'unknown' döndüren sahte judge."""

    def judge(self, claim, evidence):
        return {"status": "unknown", "confidence": 0.3, "reasoning": "insufficient evidence"}


class BrokenJudge:
    """Her zaman None döndüren (parse hatası simülasyonu) sahte judge."""

    def judge(self, claim, evidence):
        return None


# ---------------------------------------------------------------------------
# Test 1 — Happy path: desteklenen claim'ler
# ---------------------------------------------------------------------------

def test_verification_engine_aggregates_claim_results():
    engine = VerificationEngine(retriever=FakeRetriever(), judge=FakeJudge())

    result = engine.verify_claims([
        Claim(claim="Python was created in 1991.", type=ClaimType.TEMPORAL),
        Claim(claim="Python is a programming language.", type=ClaimType.FACTUAL),
    ])

    assert result.total_claims == 2
    assert result.supported_counts == 2
    assert result.refuted_counts == 0
    assert result.unknown_counts == 0
    assert result.overall_confidence == 0.9
    assert all(item.status == VerificationStatus.SUPPORTED for item in result.verifications)


# ---------------------------------------------------------------------------
# Test 2 — Refuted: çelişen kanıt senaryosu
# ---------------------------------------------------------------------------

def test_verification_engine_refuted_claim():
    engine = VerificationEngine(retriever=FakeRetriever(), judge=RefutedJudge())

    result = engine.verify_claims([
        Claim(claim="Python was invented by Microsoft.", type=ClaimType.FACTUAL),
    ])

    assert result.total_claims == 1
    assert result.refuted_counts == 1
    assert result.supported_counts == 0
    assert result.unknown_counts == 0
    assert result.verifications[0].status == VerificationStatus.REFUTED
    assert result.verifications[0].confidence == 0.85


# ---------------------------------------------------------------------------
# Test 3 — Unknown: yetersiz kanıt senaryosu
# ---------------------------------------------------------------------------

def test_verification_engine_unknown_claim():
    engine = VerificationEngine(retriever=FakeRetriever(), judge=UnknownJudge())

    result = engine.verify_claims([
        Claim(claim="Python will be discontinued in 2030.", type=ClaimType.TEMPORAL),
    ])

    assert result.total_claims == 1
    assert result.unknown_counts == 1
    assert result.supported_counts == 0
    assert result.refuted_counts == 0
    assert result.verifications[0].status == VerificationStatus.UNKNOWN
    assert result.verifications[0].confidence == 0.3


# ---------------------------------------------------------------------------
# Test 4 — Empty evidence: kanıt yok → unknown + düşük confidence fallback
# ---------------------------------------------------------------------------

def test_verification_engine_empty_evidence_falls_back_to_unknown():
    engine = VerificationEngine(retriever=EmptyRetriever(), judge=UnknownJudge())

    result = engine.verify_claims([
        Claim(claim="Some obscure fact with no supporting documents.", type=ClaimType.FACTUAL),
    ])

    assert result.total_claims == 1
    assert result.verifications[0].status == VerificationStatus.UNKNOWN
    assert len(result.verifications[0].evidence) == 0
    assert result.verifications[0].confidence < 0.5


# ---------------------------------------------------------------------------
# Test 5 — Broken judge (parse hatası): None döndüğünde unknown'a düşmeli
# ---------------------------------------------------------------------------

def test_verification_engine_handles_broken_judge_output():
    engine = VerificationEngine(retriever=FakeRetriever(), judge=BrokenJudge())

    result = engine.verify_claims([
        Claim(claim="Some claim that the judge cannot parse.", type=ClaimType.FACTUAL),
    ])

    assert result.total_claims == 1
    # judge None döndürdüğünde status "unknown", confidence 0.0 olmalı
    assert result.verifications[0].status == VerificationStatus.UNKNOWN
    assert result.verifications[0].confidence == 0.0
    assert result.overall_confidence == 0.0
