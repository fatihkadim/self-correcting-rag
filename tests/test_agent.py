import json
from unittest.mock import patch, MagicMock

from app.agent.controller import SelfCorrectionController
from app.agent.policies import Decisions
from app.schemas.claims import Claim, ClaimType, ClaimExtractionResult
from app.schemas.query import QueryRequest, QueryResponse
from app.schemas.retrieval import Chunk, RetrievalResult
from app.schemas.verification import (
    ClaimVerification,
    VerificationResult,
    VerificationStatus,
)


# ---------------------------------------------------------------------------
# Yardımcılar — her test için tutarlı dummy veri üreticileri
# ---------------------------------------------------------------------------

def _retrieval_result(contents: list[str] | None = None) -> RetrievalResult:
    contents = contents or ["evidence-1", "evidence-2"]
    return RetrievalResult(
        query="test query",
        chunks=[
            Chunk(id=str(i), content=c, source="doc.pdf", similarity_score=0.8)
            for i, c in enumerate(contents)
        ],
        retrieval_time=10.0,
    )


def _claims_result(claims_text: list[str] | None = None) -> ClaimExtractionResult:
    claims_text = claims_text or ["Claim-A", "Claim-B"]
    return ClaimExtractionResult(
        claims=[Claim(claim=c, type=ClaimType.FACTUAL) for c in claims_text],
        original_answer="original",
        claim_count=len(claims_text),
    )


def _verification_result(
    statuses: list[VerificationStatus] | None = None,
) -> VerificationResult:
    statuses = statuses or [VerificationStatus.SUPPORTED, VerificationStatus.SUPPORTED]
    verifications = [
        ClaimVerification(
            claim=f"claim-{i}",
            status=s,
            confidence=0.9 if s == VerificationStatus.SUPPORTED else 0.3,
            evidence=["ev"],
            reasoning="ok",
        )
        for i, s in enumerate(statuses)
    ]
    total = len(verifications)
    supported = sum(1 for v in verifications if v.status == VerificationStatus.SUPPORTED)
    refuted = sum(1 for v in verifications if v.status == VerificationStatus.REFUTED)
    unknown = sum(1 for v in verifications if v.status == VerificationStatus.UNKNOWN)
    confidence = sum(v.confidence for v in verifications) / total if total else 0.0
    return VerificationResult(
        verifications=verifications,
        total_claims=total,
        supported_counts=supported,
        refuted_counts=refuted,
        unknown_counts=unknown,
        overall_confidence=confidence,
    )


def _make_controller() -> SelfCorrectionController:
    """Gerçek bağımlılıkları atlayarak Controller oluşturur."""
    with patch.object(SelfCorrectionController, "__init__", lambda self: None):
        ctrl = SelfCorrectionController()
    ctrl.retriever = MagicMock()
    ctrl.answer_generator = MagicMock()
    ctrl.claim_extractor = MagicMock()
    ctrl.verifier = MagicMock()
    ctrl.answer_repair = MagicMock()
    ctrl.decider = MagicMock()
    ctrl.query_rewriter = MagicMock()
    return ctrl


# ---------------------------------------------------------------------------
# Test 1 — Tüm claim'ler desteklendi → ACCEPT
# ---------------------------------------------------------------------------

def test_controller_accept_path():
    """Decider ACCEPT derse ilk cevap doğrudan döner."""
    ctrl = _make_controller()
    request = QueryRequest(question="Python nedir?")

    ctrl.retriever.search.return_value = _retrieval_result()
    ctrl.answer_generator.generate.return_value = "Python bir dildir."
    ctrl.claim_extractor.extract.return_value = _claims_result()
    ctrl.verifier.verify_claims.return_value = _verification_result()
    ctrl.decider.decide.return_value = Decisions.ACCEPT

    with patch("app.agent.controller.settings") as mock_settings:
        mock_settings.max_retry = 3
        response = ctrl.run(request)

    assert isinstance(response, QueryResponse)
    assert response.answer == "Python bir dildir."
    ctrl.answer_repair.repair.assert_not_called()
    ctrl.query_rewriter.rewrite.assert_not_called()


# ---------------------------------------------------------------------------
# Test 2 — Kısmen desteklendi → REPAIR
# ---------------------------------------------------------------------------

def test_controller_repair_path():
    """Decider REPAIR derse onarılmış cevap döner."""
    ctrl = _make_controller()
    request = QueryRequest(question="Python nedir?")

    ctrl.retriever.search.return_value = _retrieval_result()
    ctrl.answer_generator.generate.return_value = "Orijinal cevap."
    ctrl.claim_extractor.extract.return_value = _claims_result()
    ctrl.verifier.verify_claims.return_value = _verification_result(
        [VerificationStatus.SUPPORTED, VerificationStatus.REFUTED]
    )
    ctrl.decider.decide.return_value = Decisions.REPAIR
    ctrl.answer_repair.repair.return_value = "Onarılmış cevap."

    with patch("app.agent.controller.settings") as mock_settings:
        mock_settings.max_retry = 3
        response = ctrl.run(request)

    assert response.answer == "Onarılmış cevap."
    ctrl.answer_repair.repair.assert_called_once()


# ---------------------------------------------------------------------------
# Test 3 — Hiç desteklenmedi → RETRY → max_retry'a kadar dener
# ---------------------------------------------------------------------------

def test_controller_retry_then_exhaust():
    """Decider sürekli RETRY derse max_retry'a kadar deneyip son cevabı döner."""
    ctrl = _make_controller()
    request = QueryRequest(question="Bilinmeyen soru?")

    ctrl.retriever.search.return_value = _retrieval_result()
    ctrl.answer_generator.generate.return_value = "Cevap."
    ctrl.claim_extractor.extract.return_value = _claims_result()
    ctrl.verifier.verify_claims.return_value = _verification_result(
        [VerificationStatus.REFUTED]
    )
    ctrl.decider.decide.return_value = Decisions.RETRY
    ctrl.query_rewriter.rewrite.return_value = "Yeniden yazılmış soru"

    max_retry = 2
    with patch("app.agent.controller.settings") as mock_settings:
        mock_settings.max_retry = max_retry
        response = ctrl.run(request)

    assert isinstance(response, QueryResponse)
    assert ctrl.query_rewriter.rewrite.call_count == max_retry
    assert ctrl.retriever.search.call_count == max_retry


# ---------------------------------------------------------------------------
# Test 4 — Claim extraction None döndüğünde → hemen dön
# ---------------------------------------------------------------------------

def test_controller_no_claims_returns_immediately():
    """ClaimExtractor None döndürdüğünde verification yapılmadan cevap döner."""
    ctrl = _make_controller()
    request = QueryRequest(question="Basit soru?")

    ctrl.retriever.search.return_value = _retrieval_result()
    ctrl.answer_generator.generate.return_value = "Basit cevap."
    ctrl.claim_extractor.extract.return_value = None

    with patch("app.agent.controller.settings") as mock_settings:
        mock_settings.max_retry = 3
        response = ctrl.run(request)

    assert response.answer == "Basit cevap."
    assert response.verification is None
    ctrl.verifier.verify_claims.assert_not_called()
    ctrl.decider.decide.assert_not_called()


# ---------------------------------------------------------------------------
# Test 5 — Repair sonucu bozulursa orijinal cevap korunur
# ---------------------------------------------------------------------------

def test_controller_repair_degraded_keeps_original():
    """Repair fallback cevap üretirse orijinal cevap korunur."""
    ctrl = _make_controller()
    request = QueryRequest(question="Python nedir?")

    ctrl.retriever.search.return_value = _retrieval_result()
    ctrl.answer_generator.generate.return_value = "Python genel amaçlı bir dildir."
    ctrl.claim_extractor.extract.return_value = _claims_result()
    ctrl.verifier.verify_claims.return_value = _verification_result(
        [VerificationStatus.SUPPORTED, VerificationStatus.REFUTED]
    )
    ctrl.decider.decide.return_value = Decisions.REPAIR
    # Repair fallback cevap döndürüyor
    ctrl.answer_repair.repair.return_value = "Verilen kaynaklarda bu soruya cevap bulunamamıştır."

    with patch("app.agent.controller.settings") as mock_settings:
        mock_settings.max_retry = 3
        response = ctrl.run(request)

    # Orijinal cevap korunmalı
    assert response.answer == "Python genel amaçlı bir dildir."


# ---------------------------------------------------------------------------
# Test 6 — Geçerli repair sonucu kabul edilir
# ---------------------------------------------------------------------------

def test_controller_repair_valid_answer_accepted():
    """Repair iyi bir cevap üretirse kabul edilir."""
    ctrl = _make_controller()
    request = QueryRequest(question="Python nedir?")

    ctrl.retriever.search.return_value = _retrieval_result()
    ctrl.answer_generator.generate.return_value = "Orijinal cevap."
    ctrl.claim_extractor.extract.return_value = _claims_result()
    ctrl.verifier.verify_claims.return_value = _verification_result(
        [VerificationStatus.SUPPORTED, VerificationStatus.REFUTED]
    )
    ctrl.decider.decide.return_value = Decisions.REPAIR
    ctrl.answer_repair.repair.return_value = "Python yüksek seviyeli bir programlama dilidir [1]."

    with patch("app.agent.controller.settings") as mock_settings:
        mock_settings.max_retry = 3
        response = ctrl.run(request)

    # Repair'ın ürettiği iyi cevap kabul edilmeli
    assert response.answer == "Python yüksek seviyeli bir programlama dilidir [1]."

