from enum import Enum
from pydantic import BaseModel, Field, field_validator

from app.core.config import settings


class AnswerStatus(str, Enum):
    VERIFIED = "verified"                      # tüm/yeterli iddialar doğrulandı
    REPAIRED = "repaired"                      # cevap onarıldı ve onarılan hali doğrulandı
    PARTIALLY_VERIFIED = "partially_verified"  # deneme hakkı bitti; çürütülmüş iddia yok ama tamamı doğrulanamadı
    UNVERIFIED = "unverified"                  # iddialar çıkarılamadı, doğrulama yapılamadı
    NO_ANSWER = "no_answer"                    # güvenilir cevap bulunamadı


class QueryRequest(BaseModel):
    question: str = Field(
        min_length=settings.min_question_length,
        max_length=settings.max_question_length,
    )

    @field_validator("question", mode="before")
    @classmethod
    def strip_question(cls, v):
        # Sadece boşluktan oluşan sorular min_length kontrolüne takılsın
        return v.strip() if isinstance(v, str) else v


class QueryResponse(BaseModel):
    answer: str
    status: AnswerStatus | None = None
    attempts: int = 1
    sources: list[dict] = []
    claims: list[dict] | None = None
    verification: dict | None = None
