from app.retrieval.query_rewriter import QueryRewriter


# ---------------------------------------------------------------------------
# Fake LLM
# ---------------------------------------------------------------------------

class FakeLLM:
    def __init__(self, response: str = "Reformüle edilmiş soru"):
        self.response = response

    def generate(self, prompt: str, system_prompt: str = "", **kwargs) -> str:
        return self.response


class BrokenLLM:
    """Her zaman hata fırlatan sahte LLM."""

    def generate(self, prompt: str, system_prompt: str = "", **kwargs) -> str:
        raise RuntimeError("LLM bağlantı hatası")


# ---------------------------------------------------------------------------
# Test 1 — Başarılı rewrite
# ---------------------------------------------------------------------------

def test_rewrite_returns_reformulated_question():
    """LLM başarılı döndüğünde reformüle edilmiş soru döner."""
    rewriter = QueryRewriter(llm_client=FakeLLM("Python neden popülerdir?"))

    result = rewriter.rewrite("Python nedir?", attempt=0)

    assert result == "Python neden popülerdir?"


def test_rewrite_strips_whitespace():
    """LLM yanıtındaki baştaki/sondaki boşluklar temizlenmeli."""
    rewriter = QueryRewriter(llm_client=FakeLLM("  Temiz soru?  "))

    result = rewriter.rewrite("Soru?", attempt=1)

    assert result == "Temiz soru?"
    assert not result.startswith(" ")


# ---------------------------------------------------------------------------
# Test 2 — Hata durumunda orijinal soruyu döner (graceful fallback)
# ---------------------------------------------------------------------------

def test_rewrite_returns_original_on_llm_failure():
    """LLM hata fırlatırsa orijinal soru değiştirilmeden döner."""
    rewriter = QueryRewriter(llm_client=BrokenLLM())

    result = rewriter.rewrite("Orijinal sorum?", attempt=0)

    assert result == "Orijinal sorum?"


# ---------------------------------------------------------------------------
# Test 3 — Farklı attempt değerleri
# ---------------------------------------------------------------------------

def test_rewrite_accepts_different_attempt_values():
    """attempt parametresi farklı değerlerle çağrılabilmeli."""
    rewriter = QueryRewriter(llm_client=FakeLLM("Yeni soru"))

    for attempt in [0, 1, 5]:
        result = rewriter.rewrite("Soru?", attempt=attempt)
        assert result == "Yeni soru"
