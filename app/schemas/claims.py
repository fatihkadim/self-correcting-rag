from pydantic import Field
from pydantic import BaseModel
from enum import Enum

class ClaimType(str,Enum):
    FACTUAL : "factual"
    TEMPORAL : "temporal"
    CASUAL : "casual"

class Claim(BaseModel):
    claim : str = Field(description="Doğrulanabilir iddia metni")
    type : ClaimType

class ClaimExtractionResult(BaseModel):
    claims: list[Claim]
    original_answer: str
    claim_count: int