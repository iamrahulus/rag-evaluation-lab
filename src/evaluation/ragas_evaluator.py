from src.evaluation.robust_llm import RobustOllamaEmbeddings, RobustOllamaInstructorLLM
from ragas import evaluate

# from evaluation.metrics import answer_relevance, faithfulness
from src.chunking.chunking_strategies import SimpleChunker
from src.llm.ollama import OllamaLLM
from ragas.metrics import discrete_metric
from ragas.metrics.result import MetricResult
from src.rag.pipeline import RAGPipeline
from src.vectorstore.milvus_store import MilvusStore
from src.evaluation.test_cases import TEST_CASES
from src.evaluation.evaluator import TestResult, EvalReport
from datetime import datetime
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,      # not ResponseRelevancy
    ContextPrecision,
    AnswerCorrectness,
    ContextRecall,        # useful addition — measures retrieval coverage
    FactualCorrectness,   # stronger than AnswerCorrectness for grounded eval
)
from datasets import Dataset
from ragas import experiment
@experiment()
async def run_experiment(row):
    # 1. Get answer and context from your pipeline
    print("Running test case:", row["id"])
    answer = pipeline.query(row["question"])
    context = pipeline.retrieve(row["question"])

    # 2. Score each metric manually — THIS is what calls the judge
    faith_score = await faithfulness.ascore(
        llm=ragas_llm,
        response=answer,
        retrieved_contexts=[context],
    )

    relevancy_score = await answer_relevancy.ascore(
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        user_input=row["question"],
        response=answer,
    )

    precision_score = await context_precision.ascore(
        llm=ragas_llm,
        user_input=row["question"],
        retrieved_contexts=[context],
        reference=row["expected_answer"],
    )
    print("Done scoring metrics.")
    print(type(precision_score))
        # 3. Return everything for the CSV
    return {
        **row,
        "response": answer,
        "faithfulness": faith_score.value,
        "answer_relevancy": relevancy_score.value,
        "context_precision": precision_score.value,
        "faithfulness_reason": faith_score.reason,
    }

def build_dataset(test_cases: list, pipeline: RAGPipeline) -> Dataset:
    """Run pipeline on all test cases and build RAGAS dataset."""
    samples = []
    for test_case in test_cases:
        print(f"  Running: {test_case['id']} — {test_case['question'][:60]}...")
        actual_answer = pipeline.query(test_case["question"])
        retrieved_context = pipeline.retrieve(test_case["question"])
        samples.append({
            "user_input": test_case["question"],
            "reference": test_case["expected_answer"],
            "response": actual_answer,
            "retrieved_contexts": [retrieved_context],
        })
    return Dataset.from_list(samples)


def run_test_cases(test_cases: list[dict], pipeline: RAGPipeline):

    dataset = build_dataset(test_cases, pipeline)

    from langchain_ollama import ChatOllama
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_ollama import OllamaEmbeddings
    from ragas.embeddings import embedding_factory

    from openai import AsyncOpenAI
    from ragas.llms import llm_factory

    # Ollama's OpenAI-compatible endpoint
    client = AsyncOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # required by openai client, value ignored by Ollama
    )

    base_llm = llm_factory("llama3.2", client=client)
    ragas_llm = RobustOllamaInstructorLLM(client=base_llm.client, model=base_llm.model, provider=base_llm.provider)  # pass httpx client directly to avoid instructor retry loop
    from ragas.embeddings import OpenAIEmbeddings
    ragas_embeddings = RobustOllamaEmbeddings(
        model="nomic-embed-text",   
        base_url="http://localhost:11434"  # direct to Ollama embed endpoint
    )
#    answer_relevancy = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
#    faithfulness = Faithfulness(llm=ragas_llm)
#    context_precision = ContextPrecision(llm=ragas_llm)

    # Use this instead — these ARE instances of Metric
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
    )
    from ragas import evaluate as ragas_evaluate
    # And evaluate() receives llm/embeddings at the call level
    from ragas.run_config import RunConfig
    results = ragas_evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=RunConfig(
            timeout=300,      # 5 minutes per job — generous for local CPU
            max_retries=3,    # retry on timeout before giving up
            max_workers=1,    # sequential — don't overwhelm local Ollama
        )
    )


    print(results)

if __name__ == "__main__":
    # Example usage
    from src.rag.pipeline import RAGPipeline
    pipeline = RAGPipeline(chunker=SimpleChunker(), llm=OllamaLLM(), vector_store=MilvusStore())  # Initialize with actual components in real use
    test_cases = TEST_CASES
    results = run_test_cases(test_cases, pipeline)
