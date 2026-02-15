from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    model_name: str
    temperature: float
    max_retry: int
    confidence_threshold: float = 0.7
    top_k: int = 5

    class Config:
        env_file = ".env"

settings = Settings()