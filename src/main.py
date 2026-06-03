import json
from typing import List

from src.config import settings
from src.llm.ollama import OllamaLLM
from src.rag.pipeline import RAGPipeline
from src.vectorstore.milvus_store import MilvusStore


def main() -> None:
    llm = OllamaLLM()
    # Candidate to Initialize chunker
    chunker = None
    vector_store = MilvusStore()

    pipeline = RAGPipeline(llm=llm, chunker=chunker, vector_store=vector_store)

    try:
        with open(settings.CASE_STUDIES_PATH, "r", encoding="utf-8") as f:
            case_studies = json.load(f)
            documents: List[str] = [study["content"] for study in case_studies]

        print(f"Processing {len(documents)} documents...")
        pipeline.add_documents(documents)
        print("Documents stored")

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


if __name__ == "__main__":
    main()
