from openai import OpenAI
from app.core.config import settings

DEFAULT_SYSTEM_PROMPT = "Sen bir araştırma asistanısın. SADECE verilen bilgilere dayanarak cevap ver. Bilmiyorsan 'bilmiyorum' de."

class LLMClient:
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)

    def generate(self, prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
        response = self.client.chat.completions.create(
            model=settings.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": prompt}
            ],
            temperature=settings.temperature
        )
        return response.choices[0].message.content






