import json
import pytest
from app.claims.extractor import ClaimExtractor
from app.utils.llm_json import parse_llm_json
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

    def generate(self, prompt: str, system_prompt: str = "", **kwargs) -> str:
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt
        return self.response


# ---------------------------------------------------------------------------
# 1. parse_llm_json yardımcı fonksiyonu
# ---------------------------------------------------------------------------

def test_parse_llm_json_removes_markdown_fences():
    """```json ... ``` bloğundaki işaretler temizlenmeli."""
    raw = '```json\n[{"claim": "test", "type": "factual"}]\n```'
    assert parse_llm_json(raw) == [{"claim": "test", "type": "factual"}]


def test_parse_llm_json_handles_plain_json_and_whitespace():
    assert parse_llm_json('   \n{"claims": []}\n   ') == {"claims": []}


def test_parse_llm_json_ignores_surrounding_text():
    """JSON'un etrafındaki açıklama metni tolere edilmeli."""
    raw = 'İşte sonuç: {"status": "supported"} umarım yardımcı olur.'
    assert parse_llm_json(raw) == {"status": "supported"}


def test_parse_llm_json_raises_on_garbage():
    with pytest.raises(ValueError):
        parse_llm_json("bu geçerli json değil {{{")
    with pytest.raises(ValueError):
        parse_llm_json("")


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


def test_extract_defaults_missing_or_unknown_type_to_factual():
    """type eksik/bilinmiyorsa tüm extraction düşmemeli, 'factual' kabul edilmeli."""
    llm_response = json.dumps([
        {"claim": "İddia var ama type yok."},
        {"claim": "Tanım iddiası.", "type": "definition"},
    ])
    extractor = ClaimExtractor(llm_client=FakeLLM(llm_response))

    result = extractor.extract("Bir cevap.")
    assert result is not None
    assert result.claim_count == 2
    assert all(c.type == ClaimType.FACTUAL for c in result.claims)


def test_extract_skips_invalid_items():
    """Bozuk öğeler atlanmalı, geçerliler korunmalı."""
    llm_response = json.dumps({"claims": [
        {"claim": "Geçerli iddia.", "type": "factual"},
        {"type": "factual"},
        "düz metin",
    ]})
    extractor = ClaimExtractor(llm_client=FakeLLM(llm_response))

    result = extractor.extract("Bir cevap.")
    assert result is not None
    assert [c.claim for c in result.claims] == ["Geçerli iddia."]


def test_extract_accepts_json_object_format():
    """JSON mode çıktısı {"claims": [...]} biçiminde gelir."""
    llm_response = json.dumps({"claims": [{"claim": "A.", "type": "temporal"}]})
    extractor = ClaimExtractor(llm_client=FakeLLM(llm_response))

    result = extractor.extract("Bir cevap.")
    assert result.claim_count == 1


def test_extract_returns_none_when_all_items_invalid():
    extractor = ClaimExtractor(llm_client=FakeLLM(json.dumps([{"foo": 1}])))
    assert extractor.extract("Bir cevap.") is None


def test_extract_returns_none_when_llm_raises():
    class BrokenLLM:
        def generate(self, *args, **kwargs):
            raise RuntimeError("timeout")

    assert ClaimExtractor(llm_client=BrokenLLM()).extract("Bir cevap.") is None


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
