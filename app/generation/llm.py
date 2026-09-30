import anthropic
from app.core.config import settings

# Anthropic API'de OpenAI'daki gibi şemasız bir "JSON mode" yok; yanıt
# parse_llm_json ile toleranslı ayrıştırıldığı için talimat yeterli.
JSON_MODE_INSTRUCTION = (
    "\n\nRespond with a single valid JSON object only. "
    "Do not wrap it in markdown code fences and do not add any text before or after it."
)


class LLMError(RuntimeError):
    """LLM kullanılabilir bir yanıt döndürmediğinde fırlatılır."""


class LLMClient:
    def __init__(self):
        # timeout + max_retries: SDK rate limit (429), 5xx ve bağlantı
        # hatalarında exponential backoff ile yeniden dener.
        self.client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key,
            timeout=settings.llm_timeout,
            max_retries=settings.llm_max_retries,
        )

    def generate(self, prompt: str, system_prompt: str, json_mode: bool = False) -> str:
        system = system_prompt + JSON_MODE_INSTRUCTION if json_mode else system_prompt
        try:
            response = self.client.messages.create(
                model=settings.model_name,
                max_tokens=settings.llm_max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                extra_body={"temperature": settings.temperature},
            )
        except anthropic.APIError as e:
            raise LLMError(f"LLM çağrısı başarısız: {e}") from e

        text = "".join(b.text for b in response.content if b.type == "text")
        if response.stop_reason == "refusal" or not text.strip():
            raise LLMError(
                f"LLM boş yanıt döndürdü (stop_reason={response.stop_reason})"
            )
        return text
