from app.core.logger import get_logger
from app.generation.llm import LLMClient
from app.schemas.verification import VerificationResult, VerificationStatus
from app.utils.prompts import PromptLoader

class AnswerRepair:
    def __init__(self, llm_client: LLMClient | None = None):
        self.llm = llm_client or LLMClient()
        self.logger = get_logger(__name__)

    def repair(self, question: str, verification_result: VerificationResult):
        supported_claims = [v.claim for v in verification_result.verifications if v.status == VerificationStatus.SUPPORTED]

        if not supported_claims:
            self.logger.warning("Hiç doğrulanmış iddia bulunamadı. Yeni cevap üretilemiyor.")
            return "Üzgünüm, sorunuzu yanıtlamak için güvenilir bir kanıt bulunamadı."

        supported_claims_text = "\n".join([f"- {e}" for e in supported_claims])        
        template = PromptLoader.load("repair.txt")
        system_prompt = PromptLoader.load("repair_system.txt")        
        prompt = PromptLoader.format(
            template, 
            question=question, 
            verified_claims=supported_claims_text
        )
        
        response = self.llm.generate(prompt, system_prompt=system_prompt)        

        return response
