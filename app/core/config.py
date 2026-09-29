from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.1
    # Toplam deneme sayısı (ilk deneme dahil). En az 1 olmalı.
    max_retry: int = Field(default=2, ge=1)
    confidence_threshold: float = 0.7
    top_k: int = 5
    qdrant_url: str = "http://localhost:6333"

    # LLM çağrıları için ağ dayanıklılığı
    llm_timeout: float = 60.0
    llm_max_retries: int = 3


settings = Settings()
