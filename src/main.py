import argparse
import json
from typing import List

from pymilvus import connections

from src.chunking.chunking_strategies import SimpleChunker
from src.config import settings
from src.llm.ollama import OllamaLLM
from src.rag.pipeline import RAGPipeline
from src.vectorstore.milvus_store import MilvusStore

parser = argparse.ArgumentParser(description="Run the RAG pipeline")
parser.add_argument(
    "--eval", action="store_true", default=False, help="Run evaluation after ingestion"
)
parser.add_argument(
    "--parallel",
    action="store_true",
    default=False,
    help="Run evaluation in parallel mode (only applicable if --eval is set)",
)
# Another can be added for selecting chunking strategy if needed, e.g. --chunker simple|advanced
args = parser.parse_args()


def main() -> None:
    llm = OllamaLLM()
    # Candidate to Initialize chunker
    # chunker = AdvancedChunker()
    chunker = SimpleChunker()
    vector_store = MilvusStore()

    pipeline = RAGPipeline(llm=llm, chunker=chunker, vector_store=vector_store)

    try:
        with open(settings.CASE_STUDIES_PATH, "r", encoding="utf-8") as f:
            case_studies = json.load(f)
            documents: List[str] = [study["content"] for study in case_studies]

        print(f"Processing {len(documents)} documents...")
        if vector_store.is_empty():
            pipeline.add_documents(documents)
        else:
            print("Collection already has data, skipping ingestion.")
        print("Documents stored")
        if args.eval:
            print("Running evaluation...")
            from src.evaluation.evaluator import RAGEvaluator

            evaluator = RAGEvaluator(pipeline=pipeline)
            if args.parallel:
                report = evaluator.run_parallel()
            else:
                report = evaluator.run()
            evaluator.save_report(report, "eval_report.json")
            return
        # Example questions to test the system
        questions = [
            "What work did Equal Experts do for IG group?",
            "What was demonstrated in the Forrester study?",
        ]

        # Query each question
        for question in questions:
            print(f"\nQ: {question}")
            answer = pipeline.query(question, top_k=3, temperature=0.7)
            print(f"A: {answer}")

    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        llm.close()
        connections.disconnect("default")  # explicit Milvus cleanup before shutdown


if __name__ == "__main__":
    main()
