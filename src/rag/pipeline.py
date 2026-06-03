from typing import Dict, List

from src.chunking.chunking_strategies import BaseChunker
from src.config import settings
from src.llm.base import BaseLLM
from src.vectorstore.milvus_store import MilvusStore


class RAGPipeline:
    def __init__(self, llm: BaseLLM, chunker: BaseChunker, vector_store: MilvusStore):
        self.llm = llm
        self.chunker = chunker
        self.vector_store = vector_store

        # Initialize vector store collection
        self.vector_store.create_collection()

    def add_documents(self, documents: List[str]) -> None:
        """
        TODO: Candidate to implement document processing pipeline:
        1. Chunk documents into smaller pieces
        2. Generate embeddings for chunks
        3. Store chunks and embeddings in vector store

        For testing purposes, we'll use a simple fixed text.
        """
        # Fixed text for testing
        fixed_texts = [
            "Equal Experts helped HMRC support the economy during COVID-19 by building new services in four weeks.",
            "Equal Experts worked with Pret A Manger to develop their digital platform and mobile app.",
            "The Forrester study showed $61 million in benefits from reduced costs and increased revenues.",
        ]
        # Fixed embeddings (768-dimensional vectors filled with 0.1)
        fixed_embeddings = [[0.1] * settings.EMBEDDING_DIMENSION for _ in fixed_texts]

        # Store fixed data
        self.vector_store.insert_embeddings(
            embeddings=fixed_embeddings, texts=fixed_texts
        )

    def query(
        self,
        question: str,
        top_k: int = settings.DEFAULT_TOP_K,
        temperature: float = settings.DEFAULT_TEMPERATURE,
    ) -> str:
        """Query the RAG pipeline with a question."""

        question_embedding = self.llm.get_embeddings(question)
        results: List[Dict[str, float]] = self.vector_store.retrieve(
            query_embedding=question_embedding, limit=top_k
        )
        context = "\n\n".join(str(hit["text"]) for hit in results)
        prompt = f"""Use the following context to answer the question. If you cannot answer this question based on the context, say "I cannot answer this question based on the available information."

Context:
{context}

Question: {question}

Answer:"""
        answer = self.llm.generate(prompt=prompt, temperature=temperature)
        return answer
