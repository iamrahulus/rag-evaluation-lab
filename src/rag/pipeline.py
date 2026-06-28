import os
from typing import List, Optional

from src.chunking.chunking_strategies import BaseChunker
from src.config import settings
from src.llm.base import BaseLLM
from src.vectorstore.milvus_store import MilvusStore
from src.vectorstore.sparse_encoder import SparseEncoder


class RAGPipeline:
    def __init__(self, llm: BaseLLM, chunker: BaseChunker, vector_store: MilvusStore, sparse_encoder: Optional[SparseEncoder] = None) -> None:
        self.llm = llm
        self.chunker = chunker
        self.vector_store = vector_store
        self.bm25 = sparse_encoder

        self.vector_store.create_collection()

        # Restore BM25 model fitted during a previous ingestion run so that
        # --eval and standalone queries use hybrid retrieval without re-ingesting.
        if self.bm25 is not None and os.path.exists(settings.BM25_MODEL_PATH):
            print(f"Loading BM25 model from {settings.BM25_MODEL_PATH}")
            self.bm25.load(settings.BM25_MODEL_PATH)

    def add_documents(self, documents: List[str]) -> None:
        """Chunk documents, generate dense embeddings and sparse BM25 vectors, then store."""
        chunks: List[str] = []
        for doc in documents:
            doc_chunks = self.chunker.chunk_text(doc)
            chunks.extend(doc_chunks)
            print(f"Processed document into {len(doc_chunks)} chunks.")
        print(f"Total chunks to insert: {len(chunks)}")
        # Fit BM25 on full chunk corpus so IDF is accurate across all documents
        embeddings = [self.llm.get_embeddings(chunk) for chunk in chunks]
        if self.bm25 is not None:
            print("Fitting BM25 encoder on document chunks...")
            self.bm25.fit(chunks)
            sparse_vectors = self.bm25.encode_documents(chunks)
            self.bm25.save(settings.BM25_MODEL_PATH)
            print(f"BM25 model saved to {settings.BM25_MODEL_PATH}")
        else:
            print("No BM25 encoder provided, skipping sparse vector generation.")
            sparse_vectors = None
        print("Inserting embeddings into vector store...")
        self.vector_store.insert_embeddings(
            embeddings=embeddings,
            texts=chunks,
            sparse_vectors=sparse_vectors, #None check is done in insert_embeddings
        )

    def retrieve(self, question: str, top_k: int = settings.DEFAULT_TOP_K) -> str:
        """Retrieve relevant context using hybrid search when BM25 is available."""
        print(f"Retrieving context for question: {question}")
        question_embedding = self.llm.get_embeddings(question)
        if self.bm25 is not None and self.bm25.is_fitted:
            query_sparse = self.bm25.encode_query(question)
            results = self.vector_store.hybrid_retrieve(
                query_embedding=question_embedding,
                query_sparse=query_sparse,
                limit=top_k,
            )
        else:
            results = self.vector_store.retrieve(
                query_embedding=question_embedding,
                limit=top_k,
            )

        return "\n\n".join(str(hit["text"]) for hit in results)

    def query(
        self,
        question: str,
        top_k: int = settings.DEFAULT_TOP_K,
        temperature: float = settings.DEFAULT_TEMPERATURE,
    ) -> str:
        """Query the RAG pipeline — retrieve context then generate answer."""
        print(f"Querying RAG pipeline for question: {question}")
        context = self.retrieve(question, top_k)
        prompt = f"""Use the following context to answer the question. If you cannot answer based on the context, say "I cannot answer this question based on the available information."

        Context:
        {context}

        Question: {question}

        Answer:"""
        return self.llm.generate(prompt=prompt, temperature=temperature)
