"""
RAGAS Evaluation Module
========================
Evaluates both Classic RAG and Self-Correcting RAG pipelines using RAGAS metrics:
  - Faithfulness: Is the answer grounded in the retrieved context?
  - Answer Relevancy: Is the answer relevant to the question?
  - Context Precision: Are the relevant documents ranked higher?
  - Context Recall: Does the context cover the ground truth?

Usage:
    python -m app.evaluation.ragas_eval
"""

import json
import asyncio
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from ragas import EvaluationDataset, SingleTurnSample, evaluate, RunConfig
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithoutReference,
    LLMContextRecall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from app.retrieval.retriever import QdrantRetriever
from app.generation.answer import AnswerGenerator
from app.generation.llm import LLMClient
from app.agent.controller import SelfCorrectionController
from app.schemas.query import QueryRequest
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

# ── Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_DATASET_PATH = PROJECT_ROOT / "data" / "eval" / "eval_dataset.json"
EVAL_OUTPUT_DIR = PROJECT_ROOT / "data" / "eval"
CHECKPOINT_PATH = EVAL_OUTPUT_DIR / "_checkpoint.json"

# ── Retry Config ─────────────────────────────────────────────────────
MAX_RETRIES_PER_QUESTION = 3
RETRY_BASE_DELAY = 5  # saniye


def _retry_with_backoff(func, *args, max_retries=MAX_RETRIES_PER_QUESTION, **kwargs):
    """Herhangi bir fonksiyonu exponential backoff ile yeniden dener.

    OpenAI'ın boş yanıt dönmesi (JSONDecodeError), rate limit, bağlantı
    hataları gibi geçici sorunları yakalar ve yeniden dener.
    """
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_error = e
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            logger.warning(
                f"API hatası (deneme {attempt}/{max_retries}): "
                f"{type(e).__name__}: {e} — {delay}s sonra tekrar denenecek..."
            )
            print(
                f"    ⚠ API hatası (deneme {attempt}/{max_retries}): "
                f"{type(e).__name__} — {delay}s bekleniyor..."
            )
            time.sleep(delay)

    logger.error(f"Tüm denemeler başarısız: {type(last_error).__name__}: {last_error}")
    if last_error is not None:
        raise last_error
    raise RuntimeError("Retry denemesi yapılamadı (max_retries ≤ 0)")


class RagasEvaluator:
    """Runs RAGAS evaluation on both Classic RAG and Self-Correcting RAG."""

    def __init__(self):
        # ── Pipeline components ──────────────────────────────────────
        self.llm = LLMClient()
        self.retriever = QdrantRetriever()
        self.answer_generator = AnswerGenerator(llm_client=self.llm)
        self.scr_controller = SelfCorrectionController()

        # ── RAGAS LLM & Embeddings wrappers ──────────────────────────
        self.ragas_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model=settings.model_name,
                api_key=settings.openai_api_key,
                temperature=0,
            )
        )
        self.ragas_embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(api_key=settings.openai_api_key)
        )

    # ── 1. Load Eval Dataset ────────────────────────────────────────
    def load_dataset(self, path: Optional[Path] = None) -> list[dict]:
        path = path or EVAL_DATASET_PATH
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} evaluation samples from {path}")
        return data

    # ── 2. Classic RAG Response Collection ──────────────────────────
    def _run_classic_rag(self, question: str) -> dict:
        """Run a single question through Classic RAG (no self-correction)."""
        retrieval_result = self.retriever.search(question)
        answer = self.answer_generator.generate(question, retrieval_result)
        contexts = [chunk.content for chunk in retrieval_result.chunks]
        return {
            "question": question,
            "answer": answer,
            "contexts": contexts,
        }

    # ── 3. Self-Correcting RAG Response Collection ──────────────────
    def _run_scr_rag(self, question: str) -> dict:
        """Run a single question through Self-Correcting RAG."""
        request = QueryRequest(question=question)
        response = self.scr_controller.run(request)
        # Extract context texts from sources
        contexts = [src.get("content", "") for src in response.sources]
        return {
            "question": question,
            "answer": response.answer,
            "contexts": contexts,
        }

    # ── Checkpoint Helpers ──────────────────────────────────────────
    def _save_checkpoint(
        self,
        classic_results: list[dict],
        scr_results: list[dict],
        completed_idx: int,
    ):
        """Her başarılı soru sonrası ara sonuçları diske yazar."""
        checkpoint = {
            "completed_idx": completed_idx,
            "timestamp": datetime.now().isoformat(),
            "classic_results": classic_results,
            "scr_results": scr_results,
        }
        CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)
        logger.debug(f"Checkpoint kaydedildi: soru #{completed_idx}")

    def _load_checkpoint(self) -> Optional[dict]:
        """Varsa önceki checkpoint'i yükler."""
        if CHECKPOINT_PATH.exists():
            with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
            logger.info(
                f"Checkpoint bulundu: {checkpoint['completed_idx']} soru tamamlanmış "
                f"({checkpoint['timestamp']})"
            )
            return checkpoint
        return None

    def _clear_checkpoint(self):
        """Evaluation tamamlandıktan sonra checkpoint dosyasını temizler."""
        if CHECKPOINT_PATH.exists():
            CHECKPOINT_PATH.unlink()
            logger.debug("Checkpoint temizlendi.")

    # ── 4. Collect All Responses ────────────────────────────────────
    def collect_responses(
        self, eval_data: list[dict]
    ) -> tuple[list[dict], list[dict]]:
        """Run all questions through both pipelines. Returns (classic, scr) results.

        - Her soru retry ile korunur (JSONDecodeError, RateLimitError vb.)
        - Her başarılı soru sonrası checkpoint kaydedilir
        - Önceki checkpoint varsa kaldığı yerden devam eder
        """
        classic_results = []
        scr_results = []
        start_idx = 0

        # Checkpoint varsa kaldığı yerden devam et
        checkpoint = self._load_checkpoint()
        if checkpoint:
            classic_results = checkpoint["classic_results"]
            scr_results = checkpoint["scr_results"]
            start_idx = checkpoint["completed_idx"]
            print(f"\n✓ Checkpoint'ten devam ediliyor: soru #{start_idx + 1}'den itibaren\n")

        total = len(eval_data)
        failed_questions = []

        for idx in range(start_idx, total):
            sample = eval_data[idx]
            question = sample["question"]
            ground_truth = sample["ground_truth"]
            display_idx = idx + 1
            print(f"\n[{display_idx}/{total}] Processing: {question[:80]}...")

            # ── Classic RAG (retry ile) ──
            print("  - Classic RAG running...")
            try:
                classic_res = _retry_with_backoff(self._run_classic_rag, question)
                classic_res["ground_truth"] = ground_truth
            except Exception as e:
                logger.error(f"Classic RAG başarısız (soru #{display_idx}): {e}")
                print(f"    ✗ Classic RAG başarısız — atlanıyor: {type(e).__name__}")
                classic_res = {
                    "question": question,
                    "answer": f"[HATA: {type(e).__name__}]",
                    "contexts": [],
                    "ground_truth": ground_truth,
                }
                failed_questions.append({"idx": display_idx, "pipeline": "classic", "error": str(e)})

            classic_results.append(classic_res)

            # ── Self-Correcting RAG (retry ile) ──
            print("  - Self-Correcting RAG running...")
            try:
                scr_res = _retry_with_backoff(self._run_scr_rag, question)
                scr_res["ground_truth"] = ground_truth
            except Exception as e:
                logger.error(f"SCR RAG başarısız (soru #{display_idx}): {e}")
                print(f"    ✗ SCR RAG başarısız — atlanıyor: {type(e).__name__}")
                scr_res = {
                    "question": question,
                    "answer": f"[HATA: {type(e).__name__}]",
                    "contexts": [],
                    "ground_truth": ground_truth,
                }
                failed_questions.append({"idx": display_idx, "pipeline": "scr", "error": str(e)})

            scr_results.append(scr_res)

            # ── Checkpoint kaydet ──
            self._save_checkpoint(classic_results, scr_results, idx + 1)

            # API rate limit'e çarpmamak için sorular arasında kısa bekleme
            if idx < total - 1:
                time.sleep(1)

        # Sonuçları özetle
        if failed_questions:
            print(f"\n⚠ {len(failed_questions)} API hatası atlandı:")
            for fq in failed_questions:
                print(f"  - Soru #{fq['idx']} ({fq['pipeline']}): {fq['error'][:100]}")

        logger.info(
            f"Collected {len(classic_results)} classic + {len(scr_results)} SCR responses "
            f"({len(failed_questions)} hata atlandı)"
        )
        return classic_results, scr_results

    # ── 5. Build RAGAS Dataset ──────────────────────────────────────
    def build_ragas_dataset(self, responses: list[dict]) -> EvaluationDataset:
        """Convert pipeline responses to RAGAS EvaluationDataset."""
        samples = []
        for r in responses:
            sample = SingleTurnSample(
                user_input=r["question"],
                response=r["answer"],
                retrieved_contexts=r["contexts"],
                reference=r["ground_truth"],
            )
            samples.append(sample)
        return EvaluationDataset(samples=samples)

    # ── 6. Run RAGAS Evaluation ─────────────────────────────────────
    def run_metrics(self, dataset: EvaluationDataset) -> dict:
        """Evaluate a dataset with RAGAS metrics. Returns per-sample + aggregate scores."""
        metrics = [
            Faithfulness(llm=self.ragas_llm),
            ResponseRelevancy(llm=self.ragas_llm, embeddings=self.ragas_embeddings),
            LLMContextPrecisionWithoutReference(llm=self.ragas_llm),
            LLMContextRecall(llm=self.ragas_llm),
        ]

        try:
            result = evaluate(
                dataset=dataset,
                metrics=metrics,
                run_config=RunConfig(max_workers=1, timeout=180, max_retries=15),
            )
        except Exception as e:
            logger.error(f"RAGAS evaluate() hatası: {type(e).__name__}: {e}")
            print(f"\n✗ RAGAS evaluation başarısız: {type(e).__name__}: {e}")
            raise

        return result

    # ── 7. Build Report ─────────────────────────────────────────────
    def build_report(
        self,
        classic_result,
        scr_result,
        classic_responses: list[dict],
        scr_responses: list[dict],
    ) -> dict:
        """Build a structured JSON report from both evaluation results."""
        classic_df = classic_result.to_pandas()
        scr_df = scr_result.to_pandas()

        metric_keys = [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ]

        # Map report keys → olası RAGAS sütun adları (öncelik sırasıyla)
        column_map = {
            "faithfulness": ["faithfulness"],
            "answer_relevancy": ["answer_relevancy", "response_relevancy"],
            "context_precision": [
                "LLMContextPrecisionWithoutReference",
                "context_precision",
                "llm_context_precision_without_reference",
            ],
            "context_recall": ["context_recall", "LLMContextRecall", "llm_context_recall"],
        }

        def get_metric_value(df, metric_name):
            """Get metric value using column_map with fallback to fuzzy match."""
            # Önce column_map'teki bilinen isimleri dene
            candidates = column_map.get(metric_name, [metric_name])
            for col_name in candidates:
                if col_name in df.columns:
                    return round(float(df[col_name].mean()), 4)

            # Fallback: kısmi eşleşme
            normalized = metric_name.replace("_", "")
            for col in df.columns:
                if normalized in col.lower().replace("_", "").replace(" ", ""):
                    return round(float(df[col].mean()), 4)
            return None

        classic_scores = {}
        scr_scores = {}
        for key in metric_keys:
            classic_scores[key] = get_metric_value(classic_df, key)
            scr_scores[key] = get_metric_value(scr_df, key)

        # Per-question details
        per_question = []
        for i, (c_resp, s_resp) in enumerate(
            zip(classic_responses, scr_responses)
        ):
            detail = {
                "question": c_resp["question"],
                "ground_truth": c_resp["ground_truth"],
                "classic_rag": {
                    "answer": c_resp["answer"],
                    "num_contexts": len(c_resp["contexts"]),
                },
                "self_correcting_rag": {
                    "answer": s_resp["answer"],
                    "num_contexts": len(s_resp["contexts"]),
                },
            }
            # Add per-question metric scores if available
            for key in metric_keys:
                val_c = get_metric_value(classic_df.iloc[[i]].reset_index(drop=True), key)
                val_s = get_metric_value(scr_df.iloc[[i]].reset_index(drop=True), key)
                if val_c is not None:
                    detail["classic_rag"][key] = val_c
                if val_s is not None:
                    detail["self_correcting_rag"][key] = val_s
            per_question.append(detail)

        report = {
            "timestamp": datetime.now().isoformat(),
            "model": settings.model_name,
            "num_questions": len(classic_responses),
            "classic_rag": classic_scores,
            "self_correcting_rag": scr_scores,
            "per_question_details": per_question,
        }
        return report

    # ── 8. Save Report ──────────────────────────────────────────────
    def save_report(self, report: dict, output_dir: Optional[Path] = None) -> Path:
        """Save the report as a timestamped JSON file."""
        output_dir = output_dir or EVAL_OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = output_dir / f"ragas_report_{timestamp}.json"

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Report saved: {filepath}")
        return filepath

    # ── 9. Print Comparison Table ───────────────────────────────────
    @staticmethod
    def print_comparison_table(report: dict):
        """Print a side-by-side comparison table to the console."""
        classic = report["classic_rag"]
        scr = report["self_correcting_rag"]

        print("\n" + "=" * 60)
        print("  RAGAS EVALUATION RESULTS")
        print(f"  Model: {report['model']}  |  Questions: {report['num_questions']}")
        print("=" * 60)

        header = f"{'Metrik':<25} {'Klasik RAG':>12} {'Self-Correcting':>16}"
        print(header)
        print("-" * 60)

        metric_labels = {
            "faithfulness": "Faithfulness",
            "answer_relevancy": "Answer Relevancy",
            "context_precision": "Context Precision",
            "context_recall": "Context Recall",
        }

        for key, label in metric_labels.items():
            c_val = classic.get(key)
            s_val = scr.get(key)
            c_str = f"{c_val:.4f}" if c_val is not None else "N/A"
            s_str = f"{s_val:.4f}" if s_val is not None else "N/A"

            # Indicate improvement with ASCII
            indicator = ""
            if c_val is not None and s_val is not None:
                diff = s_val - c_val
                if diff > 0.005:
                    indicator = "  ( + )"
                elif diff < -0.005:
                    indicator = "  ( - )"
                else:
                    indicator = "  ( = )"

            print(f"  {label:<23} {c_str:>12} {s_str:>14}{indicator}")

        print("=" * 60)
        print("  + = SCR better  |  - = Classic better  |  = = Similar")
        print("=" * 60 + "\n")

    # ── Main Runner ─────────────────────────────────────────────────
    def run(self, dataset_path: Optional[Path] = None):
        """Full evaluation pipeline: load → collect → evaluate → report."""
        print("\nRAGAS Evaluation Starting...\n")

        # Step 1: Load eval dataset
        eval_data = self.load_dataset(dataset_path)

        # Step 2: Collect responses from both pipelines
        classic_responses, scr_responses = self.collect_responses(eval_data)

        # Step 3: Build RAGAS datasets
        print("\nBuilding RAGAS datasets...")
        classic_dataset = self.build_ragas_dataset(classic_responses)
        scr_dataset = self.build_ragas_dataset(scr_responses)

        # Step 4: Run RAGAS evaluation
        print("\nRunning RAGAS metrics on Classic RAG...")
        classic_result = self.run_metrics(classic_dataset)

        print("\nRunning RAGAS metrics on Self-Correcting RAG...")
        scr_result = self.run_metrics(scr_dataset)

        # Step 5: Build & save report
        print("\nBuilding report...")
        report = self.build_report(
            classic_result, scr_result, classic_responses, scr_responses
        )

        filepath = self.save_report(report)

        # Step 6: Print results
        self.print_comparison_table(report)

        # Başarılı tamamlandı — checkpoint'i temizle
        self._clear_checkpoint()

        print(f"Full report saved: {filepath}")
        return report


# ── CLI Entry Point ─────────────────────────────────────────────────
if __name__ == "__main__":
    evaluator = RagasEvaluator()
    evaluator.run(PROJECT_ROOT / "data" / "eval" / "eval_dataset_extended.json")
