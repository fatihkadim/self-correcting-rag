from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str
    model_name: str = "claude-haiku-4-5"
    temperature: float = 0.1
    # Toplam deneme sayısı (ilk deneme dahil). En az 1 olmalı.
    max_retry: int = Field(default=2, ge=1)
    confidence_threshold: float = 0.7
    top_k: int = 5
    qdrant_url: str = "http://localhost:6333"

    # LLM çağrıları için ağ dayanıklılığı
    llm_timeout: float = 60.0
    llm_max_retries: int = 3
    llm_max_tokens: int = 4096

    # Claim doğrulama: paralel judge çağrısı ve claim başına kanıt sınırı
    verify_max_workers: int = Field(default=5, ge=1)
    max_evidence_per_claim: int = Field(default=5, ge=1)

    # API girdi sınırları
    max_upload_mb: int = Field(default=20, ge=1)
    min_question_length: int = 3
    max_question_length: int = 1000


settings = Settings()
