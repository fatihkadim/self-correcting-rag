from enum import Enum
from pydantic import BaseModel


class AnswerStatus(str, Enum):
    VERIFIED = "verified"                      # tüm/yeterli iddialar doğrulandı
    REPAIRED = "repaired"                      # cevap onarıldı ve onarılan hali doğrulandı
    PARTIALLY_VERIFIED = "partially_verified"  # deneme hakkı bitti; çürütülmüş iddia yok ama tamamı doğrulanamadı
    UNVERIFIED = "unverified"                  # iddialar çıkarılamadı, doğrulama yapılamadı
    NO_ANSWER = "no_answer"                    # güvenilir cevap bulunamadı


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    status: AnswerStatus | None = None
    attempts: int = 1
    sources: list[dict] = []
    claims: list[dict] | None = None
    verification: dict | None = None
