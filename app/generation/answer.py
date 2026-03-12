from app.generation.llm import LLMClient
from app.schemas.retrieval import RetrievalResult

ANSWER_SYSTEM_PROMPT = "Sen bir araştırma asistanısın. SADECE verilen bilgilere dayanarak cevap ver. Bilmiyorsan 'bilmiyorum' de."

class AnswerGenerator:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def generate(self, question: str, retrieval_result: RetrievalResult) -> str:
        context = "\n".join([f"Kaynak {i+1}: {chunk.content}" for i, chunk in enumerate(retrieval_result.chunks)])
        user_prompt = f"Kaynaklar:\n{context}\n\nSoru: {question}"
        return self.llm.generate(user_prompt, ANSWER_SYSTEM_PROMPT)



