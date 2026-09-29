from pydantic import ValidationError

from app.generation.llm import LLMClient
from app.schemas.claims import Claim, ClaimType, ClaimExtractionResult
from app.utils.prompts import PromptLoader
from app.utils.llm_json import parse_llm_json
from app.core.logger import get_logger


class ClaimExtractor():
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.logger = get_logger(__name__)

    def extract(self, answer: str) -> ClaimExtractionResult | None:
        """Cevaptaki doğrulanabilir iddiaları çıkarır.

        Tamamen ayrıştırılamayan çıktıda None döner. Tek tek bozuk öğeler
        atlanır; bilinmeyen `type` değerleri 'factual' kabul edilir.
        """
        template = PromptLoader.load("claim_extract.txt")
        system_prompt = PromptLoader.load("claim_extract_system.txt")
        user_prompt = PromptLoader.format(template, answer=answer)

        try:
            response = self.llm.generate(user_prompt, system_prompt=system_prompt, json_mode=True)
            parsed = parse_llm_json(response)
        except Exception as e:
            self.logger.warning("Claim extraction başarısız: %s", e)
            return None

        items = parsed.get("claims") if isinstance(parsed, dict) else parsed
        if not isinstance(items, list):
            self.logger.warning("Claim extraction beklenmeyen format: %r", type(parsed).__name__)
            return None

        claims = [c for c in (self._to_claim(item) for item in items) if c is not None]
        if items and not claims:
            self.logger.warning("LLM %d öğe döndürdü ama hiçbiri geçerli claim değil", len(items))
            return None

        return ClaimExtractionResult(claims=claims, original_answer=answer, claim_count=len(claims))

    def _to_claim(self, item) -> Claim | None:
        if not isinstance(item, dict):
            return None
        text = str(item.get("claim", "")).strip()
        if not text:
            return None
        claim_type = str(item.get("type", "")).lower()
        if claim_type not in {t.value for t in ClaimType}:
            claim_type = ClaimType.FACTUAL.value
        try:
            return Claim(claim=text, type=claim_type)
        except ValidationError:
            return None
