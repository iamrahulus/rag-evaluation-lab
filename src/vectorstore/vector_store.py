from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class VectorStore(ABC):
    """
    Store               upsert                          search(mode=)           Sparse encoder in pipeline?
    ─────────────────── ──────────────────────────────  ──────────────────────  ───────────────────────────
    Milvus              dense + sparse vecs as fields   hybrid_search + RRF     Yes — compute externally, pass as query_sparse
    Elasticsearch       text + dense field              single knn+query        No  — native BM25 via query_text
    Azure AI Search     text + dense field              single hybrid query     No  — native via query_text
    Pinecone            dense + sparse vecs             single hybrid request   Yes — compute externally, pass as query_sparse
    pgvector            dense only                      ANN only                No  — no sparse support
    Chroma              dense only                      ANN only                No
    """

    @abstractmethod
    def upsert(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        sparse_vectors: Optional[List[Dict[int, float]]] = None,
        metadata: Optional[List[Dict]] = None,
    ) -> None: ...

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        limit: int,
        query_text: Optional[str] = None,
        query_sparse: Optional[Dict[int, float]] = None,
        mode: str = "hybrid",
    ) -> List[Dict]:
        """
        query_text   — raw text for stores that do BM25 natively (ES, Azure AI Search)
        query_sparse — pre-computed sparse vector for stores that need it (Milvus, Pinecone)
        mode         — "dense" | "sparse" | "hybrid"
        """
        ...

    @abstractmethod
    def is_empty(self) -> bool: ...

    @abstractmethod
    def drop_collection(self) -> None: ...

    def create_collection(self, dim: int, **kwargs) -> None:
        pass  # no-op for serverless stores (Pinecone Serverless, etc.)

    def close(self) -> None:
        pass
