from app.retrieval.retriever import QdrantRetriever
from app.generation.answer import AnswerGenerator
from app.generation.llm import LLMClient
from app.claims.extractor import ClaimExtractor
from app.verification.verifier import VerificationEngine
from app.schemas.query import QueryRequest
from app.agent.controller import SelfCorrectionController
from app.evaluation.metrics import MetricsEvaluator
import json

class BaselineEvaluator:
    def __init__(self):
        self.llm = LLMClient()
        self.retriever = QdrantRetriever()
        self.answer_generator = AnswerGenerator(llm_client=self.llm)
        self.claim_extractor = ClaimExtractor(llm_client=self.llm)
        self.verifier = VerificationEngine(retriever=self.retriever)        
        self.scr_agent = SelfCorrectionController()
        self.metrics_evaluator = MetricsEvaluator()

    def run_classic_rag(self, question: str) -> dict:
        retrieval_result = self.retriever.search(question)
        answer = self.answer_generator.generate(question, retrieval_result)        
        claims_result = self.claim_extractor.extract(answer)
        verification_result = None
        if claims_result and claims_result.claims:
            verification_result = self.verifier.verify_claims(claims_result.claims)
            
        return {
            "answer": answer,
            "verification": verification_result.model_dump() if verification_result else None
        }

    def run_comparison(self, questions: list[str]):
        classic_results = []
        scr_results = []

        print(f"Toplam {len(questions)} soru için karşılaştırma başlatılıyor...\n")

        for idx, q in enumerate(questions):
            print(f"Soru {idx+1}: {q}")
            
            # Klasik RAG
            print("Klasik RAG çalışıyor...")
            classic_res = self.run_classic_rag(q)
            classic_results.append(classic_res)
            
            # Self-Correcting RAG
            print("Self-Correcting RAG çalışıyor...")
            request = QueryRequest(question=q)
            scr_res = self.scr_agent.run(request)
            scr_results.append(scr_res.model_dump())
            
            print("-" * 50)

        print("\n--- KLASİK RAG SONUÇLARI ---")
        classic_metrics = self.metrics_evaluator.evaluate(classic_results)
        self.metrics_evaluator.print_report(classic_metrics)

        print("\n--- SELF-CORRECTING RAG SONUÇLARI ---")
        scr_metrics = self.metrics_evaluator.evaluate(scr_results)
        self.metrics_evaluator.print_report(scr_metrics)


if __name__ == "__main__":
    evaluator = BaselineEvaluator()
    
    test_questions = [
        "What is attention in transformer models?",
        "How does the encoder work in the Transformer architecture?",
        "What is the capital of France? (Bu sorunun cevabı PDF'te yok, model uyduracak mı görelim)"
    ]
    
    evaluator.run_comparison(test_questions)
