from app.utils.prompts import PromptLoader
from app.utils.llm_json import parse_llm_json
from app.generation.llm import LLMClient
from app.core.logger import get_logger


class LLMJudge():
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.logger = get_logger(__name__)

    def judge(self, claim: str, evidence: list[str]) -> dict | None:
        template = PromptLoader.load("verify_user.txt")
        system = PromptLoader.load("verify_system.txt")
        evidence_text = "\n".join([f"[{i+1}] {e}" for i, e in enumerate(evidence)])
        prompt = PromptLoader.format(template, claim=claim, evidence=evidence_text)

        try:
            response = self.llm.generate(prompt, system, json_mode=True)
            parsed = parse_llm_json(response)
        except Exception as e:
            self.logger.warning("Judge başarısız: %s", e)
            return None

        if not isinstance(parsed, dict):
            self.logger.warning("Judge beklenmeyen format: %r", type(parsed).__name__)
            return None
        return parsed
