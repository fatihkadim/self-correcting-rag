from openai import OpenAI
from app.core.config import settings

class LLMClient:
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)

    def generate(self, prompt: str, system_prompt: str ) -> str:
        response = self.client.chat.completions.create(
            model=settings.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": prompt}
            ],
            temperature=settings.temperature
        )
        return response.choices[0].message.content






