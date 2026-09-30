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


# ---------------------------------------------------------------------------
# Test — kanıt sınırı ve paralel judge
# ---------------------------------------------------------------------------

class RankingRetriever:
    """Claim'e özel 3 kanıt döndürür; rank_texts çağrılarını kaydeder."""

    def __init__(self):
        self.rank_calls = []

    def search(self, query, mode=None):
        return FakeResult(f"{query}-a", f"{query}-b", f"{query}-c")

    def rank_texts(self, query, texts, top_k):
        self.rank_calls.append((query, list(texts), top_k))
        return list(reversed(texts))[:top_k]


class RecordingJudge:
    def __init__(self):
        self.evidence_sizes = []

    def judge(self, claim, evidence):
        self.evidence_sizes.append(len(evidence))
        return {"status": "supported", "confidence": 0.9, "reasoning": "ok"}


def test_verification_caps_evidence_per_claim(monkeypatch):
    """Claim'e özel + orijinal context'ler sınırı aşarsa en alakalı parçalar tutulur."""
    from app.verification import verifier as verifier_module
    monkeypatch.setattr(verifier_module.settings, "max_evidence_per_claim", 4)
    retriever, judge = RankingRetriever(), RecordingJudge()
    engine = VerificationEngine(retriever=retriever, judge=judge)

    result = engine.verify_claims(
        [Claim(claim="c1", type=ClaimType.FACTUAL)],
        original_contexts=["ctx1", "ctx2", "ctx3", "c1-a"],  # "c1-a" tekrar eklenmez
    )

    assert judge.evidence_sizes == [4]
    assert len(result.verifications[0].evidence) == 4
    query, texts, top_k = retriever.rank_calls[0]
    assert (query, top_k) == ("c1", 4)
    assert texts == ["c1-a", "c1-b", "c1-c", "ctx1", "ctx2", "ctx3"]


def test_verification_does_not_rank_when_under_limit():
    retriever = RankingRetriever()
    engine = VerificationEngine(retriever=retriever, judge=FakeJudge())

    engine.verify_claims([Claim(claim="c1", type=ClaimType.FACTUAL)], original_contexts=["ctx1"])

    assert retriever.rank_calls == []


def test_verification_runs_judges_in_parallel_and_keeps_order():
    """Judge çağrıları eşzamanlı çalışır; sonuçlar claim sırasını korur."""
    import threading
    import time

    class SlowJudge:
        def __init__(self):
            self.active = 0
            self.max_active = 0
            self.lock = threading.Lock()

        def judge(self, claim, evidence):
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            # İlk claim en yavaş: sıra korunmasaydı sonuçlar karışırdı
            time.sleep(0.2 if claim == "claim-0" else 0.05)
            with self.lock:
                self.active -= 1
            status = "refuted" if claim == "claim-0" else "supported"
            return {"status": status, "confidence": 0.8, "reasoning": claim}

    judge = SlowJudge()
    engine = VerificationEngine(retriever=FakeRetriever(), judge=judge)
    claims = [Claim(claim=f"claim-{i}", type=ClaimType.FACTUAL) for i in range(4)]

    result = engine.verify_claims(claims)

    assert judge.max_active > 1
    assert [v.claim for v in result.verifications] == [c.claim for c in claims]
    assert [v.reasoning for v in result.verifications] == [c.claim for c in claims]
    assert result.verifications[0].status == VerificationStatus.REFUTED
    assert result.refuted_counts == 1 and result.supported_counts == 3


def test_verification_with_no_claims():
    engine = VerificationEngine(retriever=FakeRetriever(), judge=FakeJudge())
    result = engine.verify_claims([])
    assert result.total_claims == 0
    assert result.verifications == []
