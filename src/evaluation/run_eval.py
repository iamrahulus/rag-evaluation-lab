"""
Run the RAG pipeline evaluation. - This is the entry point for a standalone evaluation run. 
It initializes the RAG pipeline, loads documents, and executes the evaluation suite defined in `evaluator.py`. 
The results are saved to `eval_report.json`.
Usage: python -m src.evaluation.run_eval
"""


from src.chunking.chunking_strategies import SimpleChunker
from src.config import settings
from src.llm.ollama import OllamaLLM
from src.rag.pipeline import RAGPipeline
from src.vectorstore.milvus_store import MilvusStore

from .evaluator import RAGEvaluator


def main() -> None:
    print("Initialising RAG pipeline for evaluation...")

    llm = OllamaLLM()
    chunker = SimpleChunker()
    vector_store = MilvusStore()
    pipeline = RAGPipeline(llm=llm, chunker=chunker, vector_store=vector_store)

    # Load and ingest documents
    print("Loading documents...")
    with open(settings.CASE_STUDIES_PATH, "r", encoding="utf-8") as f:
        import json as _json
        case_studies = _json.load(f)
    documents = [study["content"] for study in case_studies]
    pipeline.add_documents(documents)
    print(f"Ingested {len(documents)} documents.\n")

    # Run evaluation
    evaluator = RAGEvaluator(pipeline=pipeline)
    report = evaluator.run()
    evaluator.save_report(report, "eval_report.json")


if __name__ == "__main__":
    main()
