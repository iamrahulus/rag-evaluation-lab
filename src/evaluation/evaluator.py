"""
RAG Pipeline Evaluator.
Runs golden test cases through the pipeline and scores results across
five evaluation dimensions using LLM-as-judge.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.chunking.chunking_strategies import SimpleChunker
from src.config import settings
from src.llm.ollama import OllamaLLM
from src.rag.pipeline import RAGPipeline
from src.vectorstore.milvus_store import MilvusStore
from concurrent.futures import ThreadPoolExecutor, as_completed

from .metrics import (
    answer_correctness,
    answer_relevance,
    context_relevance,
    faithfulness,
    hallucination_rate,
)
from .test_cases import TEST_CASES


@dataclass
class TestResult:
    test_id: str
    question: str
    expected_answer: str
    actual_answer: str
    retrieved_context: str
    scores: dict[str, float] = field(default_factory=dict)
    metric_details: dict[str, Any] = field(default_factory=dict)

    @property
    def overall_score(self) -> float:
        if not self.scores:
            return 0.0
        return sum(self.scores.values()) / len(self.scores)


@dataclass
class EvalReport:
    timestamp: str
    model: str
    total_cases: int
    results: list[TestResult]

    @property
    def aggregate_scores(self) -> dict[str, float]:
        if not self.results:
            return {}
        metrics = self.results[0].scores.keys()
        return {
            metric: sum(r.scores.get(metric, 0) for r in self.results) / len(self.results)
            for metric in metrics
        }

    @property
    def overall_score(self) -> float:
        scores = self.aggregate_scores
        return sum(scores.values()) / len(scores) if scores else 0.0


class RAGEvaluator:
    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline

    def _get_context(self, question: str) -> str:
        """Retrieve context chunks for a question without generating an answer."""
        question_embedding = self.pipeline.llm.get_embeddings(question)
        results = self.pipeline.vector_store.retrieve(
            query_embedding=question_embedding,
            limit=settings.DEFAULT_TOP_K,
        )
        return "\n\n".join(str(hit["text"]) for hit in results)

    def run_test_case(self, test_case: dict) -> TestResult:
        print(f"  Running: {test_case['id']} — {test_case['question'][:60]}...")

        # Get answer and context from pipeline
        actual_answer = self.pipeline.query(test_case["question"])
        retrieved_context = self._get_context(test_case["question"])

        result = TestResult(
            test_id=test_case["id"],
            question=test_case["question"],
            expected_answer=test_case["expected_answer"],
            actual_answer=actual_answer,
            retrieved_context=retrieved_context,
        )

        # Run all metrics
        print(f"    Scoring metrics...")

        metrics_results = {
            "answer_relevance": answer_relevance(test_case["question"], actual_answer),
            "faithfulness": faithfulness(retrieved_context, actual_answer),
            "answer_correctness": answer_correctness(test_case["expected_answer"], actual_answer),
            "hallucination_rate": hallucination_rate(retrieved_context, actual_answer),
            "context_relevance": context_relevance(test_case["question"], retrieved_context),
        }

        result.scores = {k: v["score"] for k, v in metrics_results.items()}
        result.metric_details = {k: v["raw"] for k, v in metrics_results.items()}

        print(f"    Overall score: {result.overall_score:.2f}")
        return result
    
    def run_test_case_parallel(self, test_case: dict) -> TestResult:
        print(f"  Running: {test_case['id']} — {test_case['question'][:60]}...")

        # Get answer and context from pipeline
        actual_answer = self.pipeline.query(test_case["question"])
        retrieved_context = self._get_context(test_case["question"])

        result = TestResult(
            test_id=test_case["id"],
            question=test_case["question"],
            expected_answer=test_case["expected_answer"],
            actual_answer=actual_answer,
            retrieved_context=retrieved_context,
        )

        print(f"    Scoring metrics in parallel...")

        # Define all metric calls as (name, fn, args) tuples
        metric_calls = {
            "answer_relevance":  (answer_relevance,  (test_case["question"], actual_answer)),
            "faithfulness":      (faithfulness,       (retrieved_context, actual_answer)),
            "answer_correctness":(answer_correctness, (test_case["expected_answer"], actual_answer)),
            "hallucination_rate":(hallucination_rate, (retrieved_context, actual_answer)),
            "context_relevance": (context_relevance,  (test_case["question"], retrieved_context)),
        }

        # Fire all metric calls in parallel
        metrics_results = {}
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(fn, *args): name
                for name, (fn, args) in metric_calls.items()
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    metrics_results[name] = future.result()
                except Exception as e:
                    print(f"    Metric {name} failed: {e}")
                    metrics_results[name] = {"score": 0.0, "raw": str(e), "metric": name}

        result.scores = {k: v["score"] for k, v in metrics_results.items()}
        result.metric_details = {k: v["raw"] for k, v in metrics_results.items()}

        print(f"    Overall score: {result.overall_score:.2f}")
        return result

    def run(self, test_cases: list[dict] = None) -> EvalReport:
        if test_cases is None:
            test_cases = TEST_CASES

        print(f"\n{'='*60}")
        print(f"RAG Pipeline Evaluation")
        print(f"Model: {settings.LLM_MODEL}")
        print(f"Test cases: {len(test_cases)}")
        print(f"{'='*60}\n")

        results = []
        for tc in test_cases:
            result = self.run_test_case(tc)
            results.append(result)
            print()

        report = EvalReport(
            timestamp=datetime.now().isoformat(),
            model=settings.LLM_MODEL,
            total_cases=len(test_cases),
            results=results,
        )

        self._print_report(report)
        return report
    
    def run_parallel(self, test_cases: list[dict] = None) -> EvalReport:
        print("Parallel evaluation mode enabled — running metrics for each test case in parallel. And test cases in parallel too.")
        if test_cases is None:
            test_cases = TEST_CASES

        print(f"\n{'='*60}")
        print(f"RAG Pipeline Evaluation — {len(test_cases)} test cases")
        print(f"{'='*60}\n")

        # Run test cases in parallel too
        results = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self.run_test_case_parallel, tc): tc["id"]
                for tc in test_cases
            }
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"Test case failed: {e}")

        # Sort results back into original order
        id_order = {tc["id"]: i for i, tc in enumerate(test_cases)}
        results.sort(key=lambda r: id_order[r.test_id])

        report = EvalReport(
            timestamp=datetime.now().isoformat(),
            model=settings.LLM_MODEL,
            total_cases=len(test_cases),
            results=results,
        )

        self._print_report(report)
        return report

    def _print_report(self, report: EvalReport):
        print(f"\n{'='*60}")
        print(f"EVALUATION REPORT — {report.timestamp}")
        print(f"{'='*60}")
        print(f"\nAGGREGATE SCORES (across {report.total_cases} test cases):")
        print(f"{'-'*40}")

        for metric, score in report.aggregate_scores.items():
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            print(f"  {metric:<25} {bar} {score:.2f}")

        print(f"\n  {'OVERALL':<25} {'':20} {report.overall_score:.2f}")

        print(f"\n\nPER-TEST RESULTS:")
        print(f"{'-'*40}")
        for result in report.results:
            print(f"\n  [{result.test_id}] {result.question[:70]}")
            print(f"  Overall: {result.overall_score:.2f}")
            for metric, score in result.scores.items():
                status = "✓" if score >= 0.6 else "✗"
                print(f"    {status} {metric:<25} {score:.2f}")

        print(f"\n{'='*60}")

    def save_report(self, report: EvalReport, path: str = "eval_report.json"):
        data = {
            "timestamp": report.timestamp,
            "model": report.model,
            "total_cases": report.total_cases,
            "overall_score": report.overall_score,
            "aggregate_scores": report.aggregate_scores,
            "results": [
                {
                    "test_id": r.test_id,
                    "question": r.question,
                    "expected_answer": r.expected_answer,
                    "actual_answer": r.actual_answer,
                    "overall_score": r.overall_score,
                    "scores": r.scores,
                    "metric_details": r.metric_details,
                }
                for r in report.results
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nReport saved to {path}")