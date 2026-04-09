from pydantic import Field, BaseModel
from enum import Enum

class VerificationStatus(str, Enum):
    SUPPORTED = "supported"
    REFUTED = "refuted"
    UNKNOWN = "unknown"

class ClaimVerification(BaseModel):
    claim: str 
    status: VerificationStatus
    confidence: float
    evidence: list[str]
    reasoning: str

class VerificationResult(BaseModel):
    verifications: list[ClaimVerification]
    total_claims: int
    supported_counts: int
    refuted_counts:int
    unknown_counts: int
    overall_confidence: float