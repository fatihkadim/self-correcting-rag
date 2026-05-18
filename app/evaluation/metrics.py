from pydantic import BaseModel
from typing import List

class EvaluationMetrics(BaseModel):
    total_queries: int
    avg_claim_accuracy: float
    avg_refuted_rate: float
    avg_unknown_rate: float
    total_claims_processed: int

class MetricsEvaluator:
    def evaluate(self, responses: List[dict]) -> EvaluationMetrics:

        total_queries = len(responses)
        total_claims_processed = 0
        total_supported = 0
        total_refuted = 0
        total_unknown = 0
        
        for response in responses:
            verification = response.get("verification")
            if verification and isinstance(verification, dict):
                total_claims = verification.get("total_claims", 0)
                if total_claims > 0:
                    total_claims_processed += total_claims
                    total_supported += verification.get("supported_counts", 0)
                    total_refuted += verification.get("refuted_counts", 0)
                    total_unknown += verification.get("unknown_counts", 0)

        if total_claims_processed > 0:
            avg_claim_accuracy = (total_supported / total_claims_processed) * 100
            avg_refuted_rate = (total_refuted / total_claims_processed) * 100
            avg_unknown_rate = (total_unknown / total_claims_processed) * 100
        else:
            avg_claim_accuracy = 0.0
            avg_refuted_rate = 0.0
            avg_unknown_rate = 0.0

        return EvaluationMetrics(
            total_queries=total_queries,
            avg_claim_accuracy=round(avg_claim_accuracy, 2),
            avg_refuted_rate=round(avg_refuted_rate, 2),
            avg_unknown_rate=round(avg_unknown_rate, 2),
            total_claims_processed=total_claims_processed
        )

    def print_report(self, metrics: EvaluationMetrics):
        print("\n" + "="*40)
        print("SELF-CORRECTING RAG DEĞERLENDİRME RAPORU")
        print("="*40)
        print(f"Toplam İşlenen Sorgu Sayısı: {metrics.total_queries}")
        print(f"Toplam Çıkarılan İddia (Claim): {metrics.total_claims_processed}")
        print(f"Doğruluk Oranı (Supported) : %{metrics.avg_claim_accuracy}")
        print(f"Uydurma Oranı (Refuted)    : %{metrics.avg_refuted_rate}")
        print(f"Bilinmeyen Oranı (Unknown) : %{metrics.avg_unknown_rate}")
        print("="*40 + "\n")
