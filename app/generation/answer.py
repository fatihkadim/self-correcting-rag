from app.generation.llm import LLMClient
from app.schemas.retrieval import RetrievalResult
from app.utils.prompts import PromptLoader


class AnswerGenerator:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def generate(self, question: str, retrieval_result: RetrievalResult) -> str:
        template = PromptLoader.load("answer_user.txt")
        system_prompt = PromptLoader.load("answer_system.txt")
        context = "\n".join([f"[{i+1}] {chunk.content}" for i, chunk in enumerate(retrieval_result.chunks)])
        user_prompt = PromptLoader.format(template, context=context, question=question)
        return self.llm.generate(user_prompt, system_prompt)



