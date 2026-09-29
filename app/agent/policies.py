from enum import Enum
from app.schemas.verification import VerificationResult, VerificationStatus
from app.core.config import settings

class Decisions(str, Enum):
    ACCEPT = "accept"
    REPAIR = "repair"
    RETRY = "retry"

# Eşik değerleri
HIGH_CONF_ACCEPT_RATIO = 0.85

class PolicyEngine():
    def __init__(self):
        pass
    
    def decide(self,result: VerificationResult) -> Decisions:
        
        if result.total_claims == 0:
            return Decisions.ACCEPT
        elif result.supported_counts == result.total_claims:
            return Decisions.ACCEPT
        
        # Yüksek güvenle desteklenen claim oranı yüksekse kabul et
        # (gereksiz repair/retry'ı önler). Çürütülmüş (refuted) bir iddia
        # varsa oran ne olursa olsun kabul edilmez.
        high_conf_supported = sum(
            1 for v in result.verifications
            if v.status == VerificationStatus.SUPPORTED and v.confidence >= settings.confidence_threshold
        )
        if result.refuted_counts == 0 and high_conf_supported / result.total_claims >= HIGH_CONF_ACCEPT_RATIO:
            return Decisions.ACCEPT
        
        if result.supported_counts == 0:
            return Decisions.RETRY
        
        return Decisions.REPAIR
    