"""
Evaluation metrics for the RAG pipeline.
Uses LLM-as-judge with structured JSON output for robust, parseable scoring.
Scores are 0.0 to 1.0 for all metrics.
"""

import json
from typing import Any

import httpx

from src.config import settings

_judge_client = httpx.Client(base_url=settings.OLLAMA_BASE_URL, timeout=600.0)


def _normalise(score: float) -> float:
    """Normalise score to 0-1 range regardless of LLM output scale."""
    if score > 2.0:
        score = score / 10.0
    return min(1.0, max(0.0, score))

def _judge(prompt: str, metric_name: str) -> dict[str, Any]:
    """Call the LLM judge with JSON mode and return parsed response."""
    payload = {
        "model": settings.LLM_JUDGE_MODEL,
        "prompt": prompt,
        "stream": False,
        "temperature": 0.0,
        "format": "json",
    }
    response = _judge_client.post("/api/generate", json=payload)
    response.raise_for_status()
    raw = response.json()["response"].strip()
    try:
        return dict(json.loads(raw))
    except json.JSONDecodeError:
        return {"score": 0.5, "reason": raw, "metric": metric_name}


def answer_relevance(question: str, answer: str) -> dict[str, Any]:
    """
    Measures how well the answer addresses the question.
    Score: 0 = completely irrelevant, 1 = fully addresses the question.
    """
    prompt = f"""You are an expert evaluator assessing a RAG pipeline.

Question: {question}

Answer: {answer}

Evaluate how well the answer addresses the question. Consider:
- Does the answer directly respond to what was asked?
- Is the answer on-topic?
- Does it provide the information the question was seeking?

Respond with ONLY a JSON object in this exact format:
{{
    "score": <float between 0.0 and 1.0>,
    "reason": "<one sentence explanation>"
}}"""
    result = _judge(prompt, "answer_relevance")
    return {
        "score": _normalise(float(result.get("score", 0.5))),
        "raw": result.get("reason", ""),
        "metric": "answer_relevance",
    }


def faithfulness(context: str, answer: str) -> dict[str, Any]:
    """
    Measures whether the answer is grounded in the retrieved context.
    Score: 0 = answer contradicts or ignores context, 1 = fully grounded.
    """
    prompt = f"""You are an expert evaluator assessing a RAG pipeline.

Retrieved Context:
{context}

Generated Answer:
{answer}

Evaluate whether the answer is faithful to and grounded in the context provided.
Consider:
- Are the facts in the answer supported by the context?
- Does the answer introduce information not present in the context?
- Does the answer contradict anything in the context?

Respond with ONLY a JSON object in this exact format:
{{
    "score": <float between 0.0 and 1.0>,
    "reason": "<one sentence explanation>"
}}"""
    result = _judge(prompt, "faithfulness")
    return {
        "score": _normalise(float(result.get("score", 0.5))),
        "raw": result.get("reason", ""),
        "metric": "faithfulness",
    }


def answer_correctness(expected: str, actual: str) -> dict[str, Any]:
    """
    Measures semantic similarity between expected and actual answer.
    Score: 0 = completely wrong, 1 = matches expected answer.
    """
    prompt = f"""You are an expert evaluator assessing a RAG pipeline.

Expected Answer:
{expected}

Actual Answer:
{actual}

Evaluate how closely the actual answer matches the expected answer semantically.
Consider:
- Are the key facts present in both answers?
- Do they convey the same meaning even if worded differently?
- Are important details from the expected answer missing?

Respond with ONLY a JSON object in this exact format:
{{
    "score": <float between 0.0 and 1.0>,
    "reason": "<one sentence explanation>"
}}"""
    result = _judge(prompt, "answer_correctness")
    return {
        "score": _normalise(float(result.get("score", 0.5))),
        "raw": result.get("reason", ""),
        "metric": "answer_correctness",
    }


def hallucination_rate(context: str, answer: str) -> dict[str, Any]:
    """
    Detects whether the answer contains hallucinated facts not in context.
    Score: 0 = heavily hallucinated, 1 = no hallucination detected.
    Higher is better — this is a faithfulness/grounding score.
    """
    prompt = f"""You are an expert evaluator checking for hallucination in AI responses.

Retrieved Context:
{context}

Generated Answer:
{answer}

Identify any specific claims, facts, numbers, or details in the answer that
CANNOT be verified from the context provided. These are hallucinations.

Respond with ONLY a JSON object in this exact format:
{{
    "score": <float between 0.0 and 1.0 where 1.0 means no hallucinations>,
    "hallucinations_found": ["<hallucinated claim 1>", "<hallucinated claim 2>"],
    "reason": "<one sentence summary>"
}}"""
    result = _judge(prompt, "hallucination_rate")
    return {
        "score": _normalise(float(result.get("score", 0.5))),
        "raw": result.get("reason", ""),
        "hallucinations": result.get("hallucinations_found", []),
        "metric": "hallucination_rate",
    }


def context_relevance(question: str, context: str) -> dict[str, Any]:
    """
    Measures whether the retrieved context is relevant to the question.
    Score: 0 = irrelevant chunks retrieved, 1 = highly relevant context.
    """
    prompt = f"""You are an expert evaluator assessing retrieval quality in a RAG pipeline.

Question: {question}

Retrieved Context:
{context}

Evaluate how relevant the retrieved context is for answering the question.
Consider:
- Does the context contain information needed to answer the question?
- Is the context on-topic?
- Would a good answer to the question be possible using only this context?

Respond with ONLY a JSON object in this exact format:
{{
    "score": <float between 0.0 and 1.0>,
    "reason": "<one sentence explanation>"
}}"""
    result = _judge(prompt, "context_relevance")
    return {
        "score": _normalise(float(result.get("score", 0.5))),
        "raw": result.get("reason", ""),
        "metric": "context_relevance",
    }