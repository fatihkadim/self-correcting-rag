from pydantic import Field, BaseModel
from enum import Enum

class ClaimType(str, Enum):
    FACTUAL = "factual"
    TEMPORAL = "temporal"
    CAUSAL = "causal"

class Claim(BaseModel):
    claim: str = Field(description="Doğrulanabilir iddia metni")
    type: ClaimType

class ClaimExtractionResult(BaseModel):
    claims: list[Claim]
    original_answer: str
    claim_count: int