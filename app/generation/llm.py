from openai import OpenAI
from app.core.config import settings


class LLMError(RuntimeError):
    """LLM kullanılabilir bir yanıt döndürmediğinde fırlatılır."""


class LLMClient:
    def __init__(self):
        # timeout + max_retries: SDK rate limit, 5xx ve bağlantı hatalarında
        # exponential backoff ile yeniden dener.
        self.client = OpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.llm_timeout,
            max_retries=settings.llm_max_retries,
        )

    def generate(self, prompt: str, system_prompt: str, json_mode: bool = False) -> str:
        kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
        response = self.client.chat.completions.create(
            model=settings.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": prompt}
            ],
            temperature=settings.temperature,
            **kwargs,
        )
        content = response.choices[0].message.content
        if content is None:
            raise LLMError(
                f"LLM boş yanıt döndürdü (finish_reason={response.choices[0].finish_reason})"
            )
        return content
