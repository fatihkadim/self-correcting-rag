import json
from app.claims.extractor import ClaimExtractor
from app.schemas.claims import Claim, ClaimType, ClaimExtractionResult


# ---------------------------------------------------------------------------
# Fake LLM — kontrollü çıktılar döndüren sahte LLM istemcisi
# ---------------------------------------------------------------------------

class FakeLLM:
    """generate() çağrıldığında önceden tanımlı yanıtı döndürür."""

    def __init__(self, response: str):
        self.response = response
        self.last_prompt = None
        self.last_system_prompt = None

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt
        return self.response


# ---------------------------------------------------------------------------
# 1. _clean_json yardımcı fonksiyonu
# ---------------------------------------------------------------------------

def test_clean_json_removes_markdown_fences():
    """```json ... ``` bloğundaki işaretler temizlenmeli."""
    extractor = ClaimExtractor(llm_client=FakeLLM(""))
    raw = '```json\n[{"claim": "test", "type": "factual"}]\n```'
    cleaned = extractor._clean_json(raw)
    assert "```" not in cleaned
    # Temizlenmiş metin geçerli JSON olmalı
    parsed = json.loads(cleaned)
    assert isinstance(parsed, list)


def test_clean_json_handles_plain_json():
    """Zaten temiz JSON ise değiştirmemeli."""
    extractor = ClaimExtractor(llm_client=FakeLLM(""))
    raw = '[{"claim": "test", "type": "factual"}]'
    cleaned = extractor._clean_json(raw)
    assert cleaned == raw.strip()


def test_clean_json_strips_whitespace():
    """Baştaki ve sondaki boşluklar temizlenmeli."""
    extractor = ClaimExtractor(llm_client=FakeLLM(""))
    raw = '   \n[{"claim": "test"}]\n   '
    cleaned = extractor._clean_json(raw)
    assert not cleaned.startswith(" ")
    assert not cleaned.endswith(" ")


# ---------------------------------------------------------------------------
# 2. extract — başarılı senaryo
# ---------------------------------------------------------------------------

def test_extract_returns_claims_from_valid_llm_response():
    """LLM geçerli JSON döndürdüğünde ClaimExtractionResult dönmeli."""
    llm_response = json.dumps([
        {"claim": "Python 1991'de yayınlandı.", "type": "temporal"},
        {"claim": "Python dinamik tiplidir.", "type": "factual"},
    ])
    fake_llm = FakeLLM(llm_response)
    extractor = ClaimExtractor(llm_client=fake_llm)

    result = extractor.extract("Python 1991'de yayınlanan dinamik tipli bir dildir.")

    assert result is not None
    assert isinstance(result, ClaimExtractionResult)
    assert result.claim_count == 2
    assert len(result.claims) == 2
    assert result.claims[0].type == ClaimType.TEMPORAL
    assert result.claims[1].type == ClaimType.FACTUAL


def test_extract_preserves_original_answer():
    """Orijinal cevap result.original_answer'da saklanmalı."""
    llm_response = json.dumps([
        {"claim": "Bir iddia.", "type": "factual"},
    ])
    extractor = ClaimExtractor(llm_client=FakeLLM(llm_response))
    answer = "Bu orijinal cevaptır."
    result = extractor.extract(answer)

    assert result is not None
    assert result.original_answer == answer


def test_extract_with_markdown_wrapped_response():
    """LLM ```json ... ``` ile sarılmış yanıt döndürdüğünde de çalışmalı."""
    claims_data = [{"claim": "Test iddiası.", "type": "causal"}]
    llm_response = f"```json\n{json.dumps(claims_data)}\n```"
    extractor = ClaimExtractor(llm_client=FakeLLM(llm_response))

    result = extractor.extract("Bir cevap.")
    assert result is not None
    assert result.claim_count == 1
    assert result.claims[0].type == ClaimType.CAUSAL


# ---------------------------------------------------------------------------
# 3. extract — hata senaryoları
# ---------------------------------------------------------------------------

def test_extract_returns_none_on_invalid_json():
    """LLM geçersiz JSON döndürdüğünde None dönmeli."""
    fake_llm = FakeLLM("bu geçerli json değil {{{")
    extractor = ClaimExtractor(llm_client=fake_llm)

    result = extractor.extract("Bir cevap.")
    assert result is None


def test_extract_returns_none_on_schema_mismatch():
    """JSON geçerli ama şema uyumsuz (type alanı eksik) → None."""
    llm_response = json.dumps([
        {"claim": "İddia var ama type yok."}
    ])
    extractor = ClaimExtractor(llm_client=FakeLLM(llm_response))

    result = extractor.extract("Bir cevap.")
    assert result is None


def test_extract_returns_none_on_empty_string():
    """LLM boş string döndürdüğünde None dönmeli."""
    extractor = ClaimExtractor(llm_client=FakeLLM(""))

    result = extractor.extract("Bir cevap.")
    assert result is None


def test_extract_handles_empty_claims_array():
    """LLM boş dizi döndürdüğünde geçerli ama 0 claim'li sonuç dönmeli."""
    extractor = ClaimExtractor(llm_client=FakeLLM("[]"))

    result = extractor.extract("Bir cevap.")
    assert result is not None
    assert result.claim_count == 0
    assert result.claims == []
