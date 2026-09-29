from app.core.logger import get_logger
from app.generation.llm import LLMClient
from app.schemas.verification import VerificationResult, VerificationStatus
from app.utils.prompts import PromptLoader

NO_ANSWER_MESSAGE = "The provided sources do not contain a reliable answer to this question."


class AnswerRepair:
    def __init__(self, llm_client: LLMClient | None = None):
        self.llm = llm_client or LLMClient()
        self.logger = get_logger(__name__)

    def repair(self, question: str, verification_result: VerificationResult,
               original_answer: str = "") -> str:
        supported = [v for v in verification_result.verifications if v.status == VerificationStatus.SUPPORTED]
        rejected = [v for v in verification_result.verifications if v.status != VerificationStatus.SUPPORTED]

        if not supported:
            self.logger.warning("No verified claims found. Cannot generate a new answer.")
            return NO_ANSWER_MESSAGE

        # Yalnızca supported claim'lerin kanıtları (sırayı koruyarak tekilleştir)
        supporting_evidence = list(dict.fromkeys(e for v in supported for e in v.evidence))

        supported_claims_text = "\n".join(f"- {v.claim}" for v in supported)
        rejected_claims_text = "\n".join(f"- {v.claim}" for v in rejected) or "(none)"
        evidence_text = (
            "\n".join(f"[{i+1}] {e}" for i, e in enumerate(supporting_evidence))
            if supporting_evidence else "(none)"
        )

        template = PromptLoader.load("repair.txt")
        system_prompt = PromptLoader.load("repair_system.txt")
        prompt = PromptLoader.format(
            template,
            question=question,
            verified_claims=supported_claims_text,
            rejected_claims=rejected_claims_text,
            original_answer=original_answer,
            supporting_evidence=evidence_text,
        )
        return self.llm.generate(prompt, system_prompt=system_prompt)
