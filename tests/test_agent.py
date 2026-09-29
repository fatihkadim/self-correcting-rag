from unittest.mock import patch, MagicMock

from app.agent.controller import SelfCorrectionController
from app.agent.policies import Decisions
from app.generation.repair import NO_ANSWER_MESSAGE
from app.schemas.claims import Claim, ClaimType, ClaimExtractionResult
from app.schemas.query import QueryRequest, QueryResponse, AnswerStatus
from app.schemas.retrieval import Chunk, RetrievalResult
from app.schemas.verification import (
    ClaimVerification,
    VerificationResult,
    VerificationStatus,
)

S, R, U = VerificationStatus.SUPPORTED, VerificationStatus.REFUTED, VerificationStatus.UNKNOWN


# ---------------------------------------------------------------------------
# Yardımcılar — her test için tutarlı dummy veri üreticileri
# ---------------------------------------------------------------------------

def _retrieval_result(contents: list[str] | None = None) -> RetrievalResult:
    contents = contents if contents is not None else ["evidence-1", "evidence-2"]
    return RetrievalResult(
        query="test query",
        chunks=[
            Chunk(id=str(i), content=c, source="doc.pdf", similarity_score=0.8)
            for i, c in enumerate(contents)
        ],
        retrieval_time=10.0,
    )


def _claims_result(claims_text: list[str] | None = None) -> ClaimExtractionResult:
    claims_text = claims_text if claims_text is not None else ["Claim-A", "Claim-B"]
    return ClaimExtractionResult(
        claims=[Claim(claim=c, type=ClaimType.FACTUAL) for c in claims_text],
        original_answer="original",
        claim_count=len(claims_text),
    )


def _verification_result(
    statuses: list[VerificationStatus] | None = None,
) -> VerificationResult:
    statuses = statuses if statuses is not None else [S, S]
    verifications = [
        ClaimVerification(
            claim=f"claim-{i}",
            status=s,
            confidence=0.9 if s == S else 0.3,
            evidence=["ev"],
            reasoning="ok",
        )
        for i, s in enumerate(statuses)
    ]
    total = len(verifications)
    return VerificationResult(
        verifications=verifications,
        total_claims=total,
        supported_counts=sum(v.status == S for v in verifications),
        refuted_counts=sum(v.status == R for v in verifications),
        unknown_counts=sum(v.status == U for v in verifications),
        overall_confidence=sum(v.confidence for v in verifications) / total if total else 0.0,
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
    ctrl.retriever.search.return_value = _retrieval_result()
    return ctrl


def _run(ctrl, question: str, max_retry: int = 3) -> QueryResponse:
    with patch("app.agent.controller.settings") as mock_settings:
        mock_settings.max_retry = max_retry
        return ctrl.run(QueryRequest(question=question))


# ---------------------------------------------------------------------------
# Test 1 — ACCEPT
# ---------------------------------------------------------------------------

def test_controller_accept_path():
    """Decider ACCEPT derse ilk cevap doğrudan döner."""
    ctrl = _make_controller()
    ctrl.answer_generator.generate.return_value = "Python bir dildir."
    ctrl.claim_extractor.extract.return_value = _claims_result()
    ctrl.verifier.verify_claims.return_value = _verification_result()
    ctrl.decider.decide.return_value = Decisions.ACCEPT

    response = _run(ctrl, "Python nedir?")

    assert isinstance(response, QueryResponse)
    assert response.answer == "Python bir dildir."
    assert response.status == AnswerStatus.VERIFIED
    assert response.attempts == 1
    ctrl.answer_repair.repair.assert_not_called()
    ctrl.query_rewriter.rewrite.assert_not_called()


# ---------------------------------------------------------------------------
# Test 2 — REPAIR: onarılan cevap tekrar doğrulanır
# ---------------------------------------------------------------------------

def test_controller_repair_path_reverifies_repaired_answer():
    """Onarılan cevap doğrulanır; dönen claims/verification onarılan cevaba aittir."""
    ctrl = _make_controller()
    ctrl.answer_generator.generate.return_value = "Orijinal cevap."
    ctrl.claim_extractor.extract.side_effect = [_claims_result(["Doğru", "Yanlış"]), _claims_result(["Doğru"])]
    repaired_verification = _verification_result([S])
    ctrl.verifier.verify_claims.side_effect = [_verification_result([S, R]), repaired_verification]
    ctrl.decider.decide.return_value = Decisions.REPAIR
    ctrl.answer_repair.repair.return_value = "Onarılmış cevap."

    response = _run(ctrl, "Python nedir?")

    assert response.answer == "Onarılmış cevap."
    assert response.status == AnswerStatus.REPAIRED
    assert [c["claim"] for c in response.claims] == ["Doğru"]
    assert response.verification == repaired_verification.model_dump()
    ctrl.answer_repair.repair.assert_called_once()
    assert ctrl.claim_extractor.extract.call_args_list[1].args[0] == "Onarılmış cevap."


def test_controller_repair_to_no_answer_is_accepted():
    """Repair 'kaynaklarda cevap yok' derse (0 claim) bu dürüst cevap döner;
    çürütülmüş iddia içeren orijinale geri dönülmez."""
    ctrl = _make_controller()
    ctrl.answer_generator.generate.return_value = "Python genel amaçlı bir dildir."
    ctrl.claim_extractor.extract.side_effect = [_claims_result(), _claims_result([])]
    ctrl.verifier.verify_claims.side_effect = [_verification_result([S, R]), _verification_result([])]
    ctrl.decider.decide.return_value = Decisions.REPAIR
    fallback = "The provided sources do not contain an answer to this question."
    ctrl.answer_repair.repair.return_value = fallback

    response = _run(ctrl, "Python nedir?")

    assert response.answer == fallback
    assert response.status == AnswerStatus.REPAIRED


def test_controller_repair_with_refuted_claim_is_not_returned():
    """Onarılan cevap hâlâ çürütülmüş iddia içeriyorsa kabul edilmez, tekrar denenir."""
    ctrl = _make_controller()
    ctrl.answer_generator.generate.side_effect = ["Cevap 1", "Cevap 2"]
    ctrl.claim_extractor.extract.return_value = _claims_result()
    ctrl.verifier.verify_claims.side_effect = [
        _verification_result([S, R]),  # cevap 1
        _verification_result([S, R]),  # onarılmış cevap 1 → hâlâ refuted
        _verification_result([S, S]),  # cevap 2
    ]
    ctrl.decider.decide.side_effect = [Decisions.REPAIR, Decisions.ACCEPT]
    ctrl.answer_repair.repair.return_value = "Onarılmış 1"
    ctrl.query_rewriter.rewrite.return_value = "Yeniden yazılmış"

    response = _run(ctrl, "Soru?", max_retry=2)

    assert response.answer == "Cevap 2"
    assert response.status == AnswerStatus.VERIFIED
    assert response.attempts == 2


# ---------------------------------------------------------------------------
# Test 3 — RETRY ve deneme hakkının tükenmesi
# ---------------------------------------------------------------------------

def test_controller_retry_exhausted_returns_no_answer():
    """Hiçbir deneme güvenli değilse doğrulanmamış cevap yerine 'cevap yok' döner."""
    ctrl = _make_controller()
    ctrl.answer_generator.generate.return_value = "Uydurma cevap."
    ctrl.claim_extractor.extract.return_value = _claims_result()
    ctrl.verifier.verify_claims.return_value = _verification_result([R])
    ctrl.decider.decide.return_value = Decisions.RETRY
    ctrl.query_rewriter.rewrite.return_value = "Yeniden yazılmış soru"

    max_retry = 3
    response = _run(ctrl, "Bilinmeyen soru?", max_retry=max_retry)

    assert response.answer == NO_ANSWER_MESSAGE
    assert response.status == AnswerStatus.NO_ANSWER
    assert response.verification is None
    assert response.attempts == max_retry
    assert ctrl.retriever.search.call_count == max_retry
    # Son denemeden sonra gereksiz rewrite yapılmaz
    assert ctrl.query_rewriter.rewrite.call_count == max_retry - 1


def test_controller_retry_uses_rewritten_query_for_retrieval():
    ctrl = _make_controller()
    ctrl.answer_generator.generate.return_value = "Cevap."
    ctrl.claim_extractor.extract.return_value = _claims_result()
    ctrl.verifier.verify_claims.return_value = _verification_result([U])
    ctrl.decider.decide.side_effect = [Decisions.RETRY, Decisions.ACCEPT]
    ctrl.query_rewriter.rewrite.return_value = "Yeniden yazılmış soru"

    _run(ctrl, "Orijinal soru?", max_retry=2)

    searched = [c.args[0] for c in ctrl.retriever.search.call_args_list]
    assert searched == ["Orijinal soru?", "Yeniden yazılmış soru"]
    # Cevap her zaman orijinal soruya göre üretilir
    assert all(c.args[0] == "Orijinal soru?" for c in ctrl.answer_generator.generate.call_args_list)


def test_controller_exhausted_returns_best_safe_candidate():
    """Deneme hakları bitince çürütülmüş iddia içermeyen en iyi aday döner."""
    ctrl = _make_controller()
    ctrl.answer_generator.generate.side_effect = ["Refuted içeren", "Kısmen doğrulanmış"]
    ctrl.claim_extractor.extract.return_value = _claims_result()
    ctrl.verifier.verify_claims.side_effect = [
        _verification_result([S, R]), _verification_result([R]),  # deneme 1 + onarımı (refuted)
        _verification_result([S, U]), _verification_result([R]),  # deneme 2 + onarımı (refuted)
    ]
    ctrl.decider.decide.return_value = Decisions.REPAIR
    ctrl.answer_repair.repair.return_value = "Onarım"
    ctrl.query_rewriter.rewrite.return_value = "q2"

    response = _run(ctrl, "Soru?", max_retry=2)

    assert response.answer == "Kısmen doğrulanmış"
    assert response.status == AnswerStatus.PARTIALLY_VERIFIED


def test_controller_max_retry_zero_still_runs_once():
    """max_retry <= 0 çökmeye (UnboundLocalError) yol açmamalı."""
    ctrl = _make_controller()
    ctrl.answer_generator.generate.return_value = "Cevap."
    ctrl.claim_extractor.extract.return_value = _claims_result()
    ctrl.verifier.verify_claims.return_value = _verification_result()
    ctrl.decider.decide.return_value = Decisions.ACCEPT

    response = _run(ctrl, "Soru?", max_retry=0)

    assert response.answer == "Cevap."
    assert ctrl.retriever.search.call_count == 1


# ---------------------------------------------------------------------------
# Test 4 — Claim extraction başarısız → doğrulanmamış olarak işaretlenir
# ---------------------------------------------------------------------------

def test_controller_extraction_failure_marks_unverified():
    ctrl = _make_controller()
    ctrl.answer_generator.generate.return_value = "Basit cevap."
    ctrl.claim_extractor.extract.return_value = None

    response = _run(ctrl, "Basit soru?")

    assert response.answer == "Basit cevap."
    assert response.status == AnswerStatus.UNVERIFIED
    assert response.verification is None
    ctrl.verifier.verify_claims.assert_not_called()
    ctrl.decider.decide.assert_not_called()
