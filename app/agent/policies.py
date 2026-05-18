from enum import Enum
from app.schemas.verification import VerificationResult

class Decisions(str, Enum):
    ACCEPT = "accept"
    REPAIR = "repair"
    RETRY = "retry"

class PolicyEngine():
    def __init__(self):
        pass
    
    def decide(self,result: VerificationResult) -> Decisions:
        
        if result.total_claims == 0:
            return Decisions.ACCEPT
        elif result.supported_counts == result.total_claims:
            return Decisions.ACCEPT
        elif result.supported_counts == 0:
            return Decisions.RETRY
        else:
            return Decisions.REPAIR
        
        pass
    