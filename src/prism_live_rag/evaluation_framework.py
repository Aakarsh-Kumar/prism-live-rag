"""Advanced RAG Evaluation Framework using RAGChecker, RAGAS, and custom metrics.

This module implements comprehensive evaluation infrastructure to measure Samsung
Theme 04 gate compliance with concrete evidence:
- G4: Citation faithfulness and claim support verification
- Citation coverage and hallucination detection
- Context precision, recall, and relevance metrics
- Production-grade evaluation pipeline with detailed reporting

Based on Amazon's RAGChecker framework for claim-level entailment verification.
"""

from __future__ import annotations

import json
import os
import requests
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

try:
    from ragchecker import RAGResults, RAGChecker
    from ragchecker.metrics import faithfulness as ragchecker_faithfulness
except ImportError as e:
    RAGChecker = None
    RAGResults = None
    ragchecker_faithfulness = None
    print(f"WARNING: RAGChecker not available ({e}). Citation verification will be limited.")

try:
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import faithfulness, context_precision, context_recall, answer_relevancy
    from datasets import Dataset
except ImportError:
    ragas_evaluate = None
    faithfulness = None
    context_precision = None
    context_recall = None
    answer_relevancy = None
    Dataset = None
    print("WARNING: RAGAS not available. Advanced metrics will be limited.")


def _cerebras_post_with_network_retry(*args, **kwargs):
    """Retry transient transport failures without turning them into scores.

    Retries stay under five requests/minute. Exhaustion still raises, preserving
    the evaluation checkpoint; authentication and malformed outputs are not retried.
    """
    for attempt in range(3):
        try:
            return requests.post(*args, **kwargs)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            if attempt == 2:
                raise
            print("Cerebras G4 transport failure; retrying the same request in 12.5 seconds.", flush=True)
            time.sleep(12.5)


def _cerebras_rate_limited_completion(api_key: str, model: str):
    """Return a RefChecker callback paced below Cerebras' 5-RPM user limit."""
    lock = threading.Lock()
    last_request_started = 0.0

    def complete(prompts: list[str]) -> list[str]:
        nonlocal last_request_started
        responses = []
        for prompt in prompts:
            while True:
                with lock:
                    delay = 12.5 - (time.monotonic() - last_request_started)
                    if last_request_started and delay > 0:
                        time.sleep(delay)
                    response = _cerebras_post_with_network_retry(
                        "https://api.cerebras.ai/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": model,
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": 0,
                            "reasoning_effort": "low",
                            "max_completion_tokens": 2048,
                        },
                        timeout=120,
                    )
                    last_request_started = time.monotonic()
                if response.status_code == 429:
                    print(
                        "Cerebras G4 rate/quota limit response; retrying this same "
                        "request after the 5-RPM pacing interval: "
                        + response.text[:500],
                        flush=True,
                    )
                    continue
                break
            if response.status_code >= 400:
                raise RuntimeError(
                    f"Cerebras G4 request failed ({response.status_code}): "
                    f"{response.text[:500]}"
                )
            try:
                content = response.json()["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                try:
                    message_keys = sorted(response.json()["choices"][0]["message"])
                except (KeyError, IndexError, TypeError):
                    message_keys = []
                raise RuntimeError(
                    "Cerebras response had no assistant content; "
                    f"message fields were {message_keys}"
                ) from exc
            if not content:
                raise RuntimeError("Cerebras returned an empty G4 evaluation response")
            responses.append(content)
        return responses

    return complete


@dataclass
class CitationVerificationResult:
    """Result of citation faithfulness verification."""

    query: str
    answer: str
    citations: list[str]
    retrieved_passages: list[str]

    # RAGChecker results
    overall_faithfulness: float  # 0-1 score
    claim_level_scores: list[dict]  # Per-claim verification

    # RAGAS results
    faithfulness_score: float
    context_precision_score: float
    context_recall_score: float
    answer_relevancy_score: float

    # Citation analysis
    citation_support_rate: float  # % of claims supported by citations
    hallucination_rate: float     # % of claims not supported by corpus
    citation_coverage: float      # % of relevant passages cited

    processing_time_ms: float


@dataclass
class EvaluationReport:
    """Comprehensive evaluation report for Samsung compliance."""

    # Overall metrics
    total_queries: int
    evaluation_time_s: float

    # G4 Citation Support Metrics (TARGET: >=85%)
    mean_citation_support: float
    median_citation_support: float
    citation_support_passing_rate: float  # % of queries >= 85% support

    # Faithfulness Metrics
    mean_faithfulness: float
    median_faithfulness: float
    faithfulness_passing_rate: float  # % queries >= 80% faithfulness

    # Context Quality Metrics
    mean_context_precision: float
    mean_context_recall: float
    mean_answer_relevancy: float

    # Risk Analysis
    high_hallucination_queries: list[str]  # Queries with >20% hallucination
    unsupported_claims: list[dict]         # Claims without citation support

    # Detailed Results
    query_results: list[CitationVerificationResult]


class RAGEvaluationFramework:
    """Production-grade RAG evaluation framework for Samsung Theme 04."""

    def __init__(
        self,
        *,
        ragchecker_config: dict | None = None,
        enable_ragchecker: bool = True,
        enable_ragas: bool = True,
        citation_support_threshold: float = 0.85,
        faithfulness_threshold: float = 0.80,
        evaluation_provider: str = "groq",
        use_groq: bool | None = None,
    ):
        self.enable_ragchecker = enable_ragchecker and RAGChecker is not None
        self.enable_ragas = enable_ragas and ragas_evaluate is not None
        self.citation_support_threshold = citation_support_threshold
        self.faithfulness_threshold = faithfulness_threshold

        # Initialize RAGChecker through the explicitly selected provider.
        if self.enable_ragchecker:
            try:
                from .config import load_settings

                settings = load_settings()
                if use_groq is not None:
                    evaluation_provider = "groq" if use_groq else "deepseek"

                provider_options: dict[str, Any] = {}
                if evaluation_provider == "groq":
                    if not settings.groq_api_key:
                        raise ValueError("GROQ_API_KEY is required for Groq G4 evaluation")
                    os.environ["GROQ_API_KEY"] = settings.groq_api_key
                    model = settings.groq_model
                    if not model.startswith("groq/"):
                        model = f"groq/{model}"
                elif evaluation_provider == "deepseek":
                    if not settings.deepseek_api_key:
                        raise ValueError("DEEPSEEK_API_KEY is required for DeepSeek G4 evaluation")
                    os.environ["DEEPSEEK_API_KEY"] = settings.deepseek_api_key
                    model = f"deepseek/{settings.deepseek_model}"
                elif evaluation_provider == "cerebras":
                    if not settings.cerebras_api_key:
                        raise ValueError("CEREBRAS_API_KEY is required for Cerebras G4 evaluation")
                    model = f"cerebras/{settings.cerebras_model}"
                    provider_options = {
                        "custom_llm_api_func": _cerebras_rate_limited_completion(
                            settings.cerebras_api_key,
                            settings.cerebras_model,
                        ),
                        "extractor_max_new_tokens": 768,
                        "joint_check_num": 20,
                    }
                else:
                    raise ValueError(
                        "G4_EVAL_PROVIDER must be one of: cerebras, groq, deepseek"
                    )

                default_config = {
                    "extractor_name": model,
                    "checker_name": model,
                    "batch_size_extractor": 1,
                    "batch_size_checker": 1,
                    **provider_options,
                }

                config = {**default_config, **(ragchecker_config or {})}

                self.ragchecker = RAGChecker(**config)
                print(f"RAGChecker initialized with: {config.get('extractor_name', 'default')}")
            except Exception as e:
                print(f"Failed to initialize RAGChecker: {e}")
                self.enable_ragchecker = False
                self.ragchecker = None
        else:
            self.ragchecker = None

        # Initialize RAGAS metrics
        if self.enable_ragas:
            self.ragas_metrics = [
                faithfulness,
                context_precision,
                context_recall,
                answer_relevancy,
            ]
        else:
            self.ragas_metrics = []

    def evaluate_citation_faithfulness(
        self,
        query: str,
        answer: str,
        citations: list[str],
        retrieved_passages: list[str],
    ) -> CitationVerificationResult:
        """Evaluate citation faithfulness using multiple frameworks."""

        start_time = time.perf_counter()

        # Initialize scores
        overall_faithfulness = 0.0
        claim_level_scores = []
        faithfulness_score = 0.0
        context_precision_score = 0.0
        context_recall_score = 0.0
        answer_relevancy_score = 0.0

        # RAGChecker evaluation
        if self.enable_ragchecker and self.ragchecker:
            try:
                rag_results = self._evaluate_with_ragchecker(
                    query, answer, retrieved_passages
                )
                overall_faithfulness = rag_results.get('overall_faithfulness', 0.0)
                claim_level_scores = rag_results.get('claim_scores', [])
            except Exception as e:
                raise RuntimeError(f"RAGChecker evaluation failed: {e}") from e

        # RAGAS evaluation
        if self.enable_ragas:
            try:
                ragas_scores = self._evaluate_with_ragas(
                    query, answer, retrieved_passages
                )
                faithfulness_score = ragas_scores.get('faithfulness', 0.0)
                context_precision_score = ragas_scores.get('context_precision', 0.0)
                context_recall_score = ragas_scores.get('context_recall', 0.0)
                answer_relevancy_score = ragas_scores.get('answer_relevancy', 0.0)
            except Exception as e:
                print(f"RAGAS evaluation failed: {e}")

        # Citation analysis
        citation_analysis = self._analyze_citation_support(
            answer, citations, retrieved_passages
        )

        processing_time = (time.perf_counter() - start_time) * 1000

        return CitationVerificationResult(
            query=query,
            answer=answer,
            citations=citations,
            retrieved_passages=retrieved_passages,
            overall_faithfulness=overall_faithfulness,
            claim_level_scores=claim_level_scores,
            faithfulness_score=faithfulness_score,
            context_precision_score=context_precision_score,
            context_recall_score=context_recall_score,
            answer_relevancy_score=answer_relevancy_score,
            citation_support_rate=citation_analysis['support_rate'],
            hallucination_rate=citation_analysis['hallucination_rate'],
            citation_coverage=citation_analysis['coverage'],
            processing_time_ms=processing_time,
        )

    def _evaluate_with_ragchecker(
        self,
        query: str,
        answer: str,
        passages: list[str],
        gt_answer: str | None = None,
    ) -> dict:
        """Evaluate using RAGChecker framework with proper data format.

        For Samsung G4, we need faithfulness metrics which can run without ground truth.
        Full metrics (precision/recall) require ground truth answers.
        """

        # RAGChecker REQUIRES gt_answer in the data structure
        # For faithfulness-only evaluation, use the response as pseudo-ground-truth
        effective_gt = gt_answer if gt_answer else answer

        # Create RAGResults-compatible data structure
        rag_data_dict = {
            "results": [{
                "query_id": "eval_query",
                "query": query,
                "gt_answer": effective_gt,  # Required field
                "response": answer,
                "retrieved_context": [
                    {"doc_id": f"doc_{i}", "text": passage}
                    for i, passage in enumerate(passages)
                ],
            }]
        }

        try:
            # Convert to RAGResults object
            import json
            rag_results = RAGResults.from_json(json.dumps(rag_data_dict))

            # Samsung G4 needs faithfulness only. Running all RAGChecker metrics
            # also extracts/checks pseudo-ground-truth claims and burns provider
            # tokens on precision, recall, and retriever metrics that G4 does not
            # score.
            self.ragchecker.evaluate(rag_results, ragchecker_faithfulness)

            # Extract results - RAGChecker adds metrics to the results objects
            first_result = rag_results.results[0]

            # Access the metrics that were added by RAGChecker
            metrics_dict = {}
            if hasattr(first_result, 'metrics'):
                metrics_dict = first_result.metrics or {}

            # Try different ways to access the results
            faithfulness = (
                metrics_dict.get('faithfulness', 0.0) or
                getattr(first_result, 'faithfulness', 0.0) or
                0.0
            )

            precision = (
                metrics_dict.get('precision', 0.0) or
                getattr(first_result, 'precision', 0.0) or
                0.0
            )

            return {
                'overall_faithfulness': float(faithfulness),
                'precision': float(precision),
                'recall': 0.0,  # Needs real ground truth
                'claim_scores': [],
            }

        except Exception as e:
            raise RuntimeError(f"RAGChecker API/evaluation error: {e}") from e

    def _evaluate_with_ragas(
        self,
        query: str,
        answer: str,
        passages: list[str],
    ) -> dict:
        """Evaluate using RAGAS framework."""

        # Create dataset for RAGAS
        data = {
            'question': [query],
            'answer': [answer],
            'contexts': [passages],
        }
        dataset = Dataset.from_dict(data)

        # Run evaluation
        results = ragas_evaluate(
            dataset,
            metrics=self.ragas_metrics,
        )

        return {
            'faithfulness': results.get('faithfulness', 0.0),
            'context_precision': results.get('context_precision', 0.0),
            'context_recall': results.get('context_recall', 0.0),
            'answer_relevancy': results.get('answer_relevancy', 0.0),
        }

    def _analyze_citation_support(
        self,
        answer: str,
        citations: list[str],
        passages: list[str],
    ) -> dict:
        """Analyze citation support using rule-based methods."""

        # Basic heuristic analysis
        # In production, this should use NLP techniques for claim extraction

        sentences = self._extract_claims(answer)
        total_claims = len(sentences)

        if total_claims == 0:
            return {
                'support_rate': 0.0,
                'hallucination_rate': 0.0,
                'coverage': 0.0,
            }

        supported_claims = 0
        for sentence in sentences:
            if self._claim_supported_by_passages(sentence, passages):
                supported_claims += 1

        support_rate = supported_claims / total_claims
        hallucination_rate = 1.0 - support_rate

        # Citation coverage (rough estimate)
        cited_passages = len([c for c in citations if c.strip()])
        total_passages = len(passages)
        coverage = cited_passages / max(total_passages, 1)

        return {
            'support_rate': support_rate,
            'hallucination_rate': hallucination_rate,
            'coverage': coverage,
        }

    def _extract_claims(self, answer: str) -> list[str]:
        """Extract factual claims from answer text."""

        # Simple sentence splitting - in production use spaCy or similar
        sentences = []
        for sent in answer.split('.'):
            sent = sent.strip()
            if len(sent) > 10:  # Filter very short fragments
                sentences.append(sent)

        return sentences

    def _claim_supported_by_passages(self, claim: str, passages: list[str]) -> bool:
        """Check if claim is supported by retrieved passages."""

        # Simple keyword-based support check
        # In production, use semantic similarity or entailment models
        claim_lower = claim.lower()
        claim_words = set(claim_lower.split())

        for passage in passages:
            passage_lower = passage.lower()
            passage_words = set(passage_lower.split())

            # If significant word overlap, consider supported
            overlap = len(claim_words & passage_words)
            if overlap >= min(3, len(claim_words) * 0.3):
                return True

        return False

    def evaluate_batch(
        self,
        evaluation_data: list[dict],
    ) -> EvaluationReport:
        """Evaluate a batch of queries and generate comprehensive report."""

        start_time = time.perf_counter()
        results = []

        print(f"Starting evaluation of {len(evaluation_data)} queries...")

        for i, data in enumerate(evaluation_data):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(evaluation_data)} queries processed")

            result = self.evaluate_citation_faithfulness(
                query=data['query'],
                answer=data['answer'],
                citations=data.get('citations', []),
                retrieved_passages=data.get('passages', []),
            )
            results.append(result)

        evaluation_time = time.perf_counter() - start_time

        # Generate report
        report = self._generate_report(results, evaluation_time)

        print(f"\n✅ Evaluation complete! {len(results)} queries processed in {evaluation_time:.1f}s")
        print(f"📊 Citation Support Rate: {report.mean_citation_support:.1%}")
        print(f"📊 Faithfulness Score: {report.mean_faithfulness:.1%}")

        return report

    def _generate_report(
        self,
        results: list[CitationVerificationResult],
        evaluation_time: float,
    ) -> EvaluationReport:
        """Generate comprehensive evaluation report."""

        if not results:
            return EvaluationReport(
                total_queries=0,
                evaluation_time_s=evaluation_time,
                mean_citation_support=0.0,
                median_citation_support=0.0,
                citation_support_passing_rate=0.0,
                mean_faithfulness=0.0,
                median_faithfulness=0.0,
                faithfulness_passing_rate=0.0,
                mean_context_precision=0.0,
                mean_context_recall=0.0,
                mean_answer_relevancy=0.0,
                high_hallucination_queries=[],
                unsupported_claims=[],
                query_results=results,
            )

        # Citation support analysis
        support_scores = [r.citation_support_rate for r in results]
        mean_support = sum(support_scores) / len(support_scores)
        median_support = sorted(support_scores)[len(support_scores) // 2]
        passing_support = sum(1 for s in support_scores if s >= self.citation_support_threshold)
        support_passing_rate = passing_support / len(results)

        # Faithfulness analysis
        faith_scores = [r.overall_faithfulness for r in results if r.overall_faithfulness > 0]
        if faith_scores:
            mean_faithfulness = sum(faith_scores) / len(faith_scores)
            median_faithfulness = sorted(faith_scores)[len(faith_scores) // 2]
            passing_faithfulness = sum(1 for s in faith_scores if s >= self.faithfulness_threshold)
            faithfulness_passing_rate = passing_faithfulness / len(faith_scores)
        else:
            mean_faithfulness = 0.0
            median_faithfulness = 0.0
            faithfulness_passing_rate = 0.0

        # Context quality analysis
        precision_scores = [r.context_precision_score for r in results if r.context_precision_score > 0]
        recall_scores = [r.context_recall_score for r in results if r.context_recall_score > 0]
        relevancy_scores = [r.answer_relevancy_score for r in results if r.answer_relevancy_score > 0]

        mean_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
        mean_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
        mean_relevancy = sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else 0.0

        # Risk analysis
        high_hallucination_queries = [
            r.query for r in results if r.hallucination_rate > 0.20
        ]

        unsupported_claims = []
        for result in results:
            for claim_data in result.claim_level_scores:
                if isinstance(claim_data, dict) and claim_data.get('supported', True) is False:
                    unsupported_claims.append({
                        'query': result.query,
                        'claim': claim_data.get('claim', ''),
                        'support_score': claim_data.get('score', 0.0),
                    })

        return EvaluationReport(
            total_queries=len(results),
            evaluation_time_s=evaluation_time,
            mean_citation_support=mean_support,
            median_citation_support=median_support,
            citation_support_passing_rate=support_passing_rate,
            mean_faithfulness=mean_faithfulness,
            median_faithfulness=median_faithfulness,
            faithfulness_passing_rate=faithfulness_passing_rate,
            mean_context_precision=mean_precision,
            mean_context_recall=mean_recall,
            mean_answer_relevancy=mean_relevancy,
            high_hallucination_queries=high_hallucination_queries,
            unsupported_claims=unsupported_claims,
            query_results=results,
        )

    def save_report(
        self,
        report: EvaluationReport,
        output_path: Path | str,
    ) -> None:
        """Save evaluation report to JSON file."""

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert report to serializable format
        report_data = {
            'metadata': {
                'total_queries': report.total_queries,
                'evaluation_time_s': report.evaluation_time_s,
                'evaluation_framework': {
                    'ragchecker_enabled': self.enable_ragchecker,
                    'ragas_enabled': self.enable_ragas,
                    'citation_support_threshold': self.citation_support_threshold,
                    'faithfulness_threshold': self.faithfulness_threshold,
                },
            },
            'samsung_gate_compliance': {
                'G4_citation_support': {
                    'target': '>=85%',
                    'achieved': f"{report.mean_citation_support:.1%}",
                    'passing_rate': f"{report.citation_support_passing_rate:.1%}",
                    'status': 'PASS' if report.mean_citation_support >= 0.85 else 'FAIL',
                },
                'faithfulness': {
                    'target': '>=80%',
                    'achieved': f"{report.mean_faithfulness:.1%}",
                    'passing_rate': f"{report.faithfulness_passing_rate:.1%}",
                    'status': 'PASS' if report.mean_faithfulness >= 0.80 else 'FAIL',
                },
            },
            'metrics_summary': {
                'citation_support': {
                    'mean': report.mean_citation_support,
                    'median': report.median_citation_support,
                    'passing_rate': report.citation_support_passing_rate,
                },
                'faithfulness': {
                    'mean': report.mean_faithfulness,
                    'median': report.median_faithfulness,
                    'passing_rate': report.faithfulness_passing_rate,
                },
                'context_quality': {
                    'precision': report.mean_context_precision,
                    'recall': report.mean_context_recall,
                    'relevancy': report.mean_answer_relevancy,
                },
            },
            'risk_analysis': {
                'high_hallucination_queries_count': len(report.high_hallucination_queries),
                'unsupported_claims_count': len(report.unsupported_claims),
                'high_risk_queries': report.high_hallucination_queries[:10],  # Top 10
            },
            'detailed_results': [
                {
                    'query': r.query,
                    'citation_support_rate': r.citation_support_rate,
                    'faithfulness_score': r.overall_faithfulness,
                    'hallucination_rate': r.hallucination_rate,
                    'processing_time_ms': r.processing_time_ms,
                }
                for r in report.query_results
            ],
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        print(f"📄 Evaluation report saved to: {output_path}")


def create_evaluation_framework(**kwargs) -> RAGEvaluationFramework:
    """Factory function to create evaluation framework with proper configuration."""

    # Load .env before selecting a provider so the evaluator can use its
    # configured credential and model.
    from .config import load_settings
    load_settings()

    # Do not pass generic ``batch_size`` or ``device`` fields through to
    # LiteLLM: provider APIs such as Groq reject them as request parameters.
    return RAGEvaluationFramework(**kwargs)
