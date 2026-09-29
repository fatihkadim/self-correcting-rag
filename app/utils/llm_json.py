import json
import re

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_llm_json(text: str | None):
    """LLM çıktısından JSON ayrıştırır.

    Markdown code fence'lerini temizler; JSON'un önünde/arkasında açıklama
    metni varsa ilk `{`/`[` ile son `}`/`]` arasını dener.
    Ayrıştırılamazsa ValueError fırlatır.
    """
    if not text or not text.strip():
        raise ValueError("Boş LLM yanıtı")

    cleaned = _FENCE_RE.sub("", text.strip()).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    starts = [i for i in (cleaned.find("{"), cleaned.find("[")) if i != -1]
    end = max(cleaned.rfind("}"), cleaned.rfind("]"))
    if starts and end > min(starts):
        try:
            return json.loads(cleaned[min(starts):end + 1])
        except json.JSONDecodeError as e:
            raise ValueError(f"Geçersiz JSON: {e}") from e
    raise ValueError("Yanıtta JSON bulunamadı")
