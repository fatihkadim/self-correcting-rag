from app.generation.llm import LLMClient
from app.utils.prompts import PromptLoader
from app.core.logger import get_logger


class QueryRewriter:
    def __init__(self, llm_client: LLMClient | None = None):
        self.llm = llm_client or LLMClient()
        self.logger = get_logger(__name__)

    def rewrite(self, question: str, attempt: int) -> str:
        template = PromptLoader.load("query_rewrite.txt")
        system_prompt = PromptLoader.load("query_rewrite_system.txt")
        prompt = PromptLoader.format(template, question=question, attempt=attempt)

        try:
            rewritten = self.llm.generate(prompt, system_prompt=system_prompt)
            rewritten = rewritten.strip()
            self.logger.info(
                "Query rewritten: attempt=%d original='%s' rewritten='%s'",
                attempt, question, rewritten,
            )
            return rewritten
        except Exception as e:
            self.logger.warning("Query rewrite başarısız: %s — orijinal soru kullanılacak.", e)
            return question
