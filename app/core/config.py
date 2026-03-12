from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    model_name: str
    temperature: float = 0.1
    max_retry: int
    confidence_threshold: float = 0.7
    top_k: int = 5
    qdrant_url: str = "http://localhost:6333"

    class Config:
        env_file = ".env"

settings = Settings()