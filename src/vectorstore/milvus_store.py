from typing import Dict, List, Optional

from pymilvus import AnnSearchRequest, DataType, MilvusClient, RRFRanker
from pymilvus.client.types import LoadState

from src.config import settings
from src.vectorstore.vector_store import VectorStore


class MilvusStore(VectorStore):
    def __init__(self) -> None:
        self.collection_name = settings.COLLECTION_NAME
        self._client: Optional[MilvusClient] = None

    # ── connection ────────────────────────────────────────────────────────────

    def _get_client(self) -> MilvusClient:
        if self._client is None:
            self._client = MilvusClient(
                uri=settings.MILVUS_DB_PATH,
                # Disable keepalive pings when no RPCs are in flight.
                # The default (True) causes GOAWAY from milvus_lite during long idle periods
                # (e.g. while generating embeddings), which kills the channel.
                grpc_options={"grpc.keepalive_permit_without_calls": False},
            )
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    # ── collection lifecycle ──────────────────────────────────────────────────

    def create_collection(self, dim: int = settings.EMBEDDING_DIMENSION, **kwargs) -> None:
        """Create collection with dense + sparse fields. No-op if already populated and schema is current."""
        client = self._get_client()

        if client.has_collection(self.collection_name):
            fields = client.describe_collection(self.collection_name)["fields"]
            has_sparse = any(f["name"] == "sparse" for f in fields)
            stats = client.get_collection_stats(self.collection_name)
            if stats["row_count"] > 0 and has_sparse:
                return
            client.drop_collection(self.collection_name)

        schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field(field_name="id",        datatype=DataType.INT64,          is_primary=True)
        schema.add_field(field_name="text",      datatype=DataType.VARCHAR,         max_length=65535)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR,    dim=dim)
        schema.add_field(field_name="sparse",    datatype=DataType.SPARSE_FLOAT_VECTOR)

        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(field_name="embedding", index_type="AUTOINDEX",            metric_type="IP")
        index_params.add_index(field_name="sparse",    index_type="SPARSE_INVERTED_INDEX", metric_type="IP")

        client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )

    def is_empty(self) -> bool:
        client = self._get_client()
        if not client.has_collection(self.collection_name):
            return True
        stats = client.get_collection_stats(self.collection_name)
        return bool(stats["row_count"] == 0)

    def drop_collection(self) -> None:
        client = self._get_client()
        if client.has_collection(self.collection_name):
            client.drop_collection(self.collection_name)

    def _ensure_collection_loaded(self) -> None:
        """Load the collection into memory if needed. Only required before search."""
        client = self._get_client()
        if not client.has_collection(self.collection_name):
            return
        state = client.get_load_state(collection_name=self.collection_name)["state"]
        if state != LoadState.Loaded:
            client.load_collection(self.collection_name)

    # ── write ─────────────────────────────────────────────────────────────────

    def upsert(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        sparse_vectors: Optional[List[Dict[int, float]]] = None,
        metadata: Optional[List[Dict]] = None,
        batch_size: int = 200,
    ) -> None:
        """Insert texts, dense embeddings, and optional pre-computed sparse BM25 vectors."""
        if len(embeddings) != len(texts):
            raise ValueError("Number of embeddings and texts must match")
        if sparse_vectors is not None and len(sparse_vectors) != len(texts):
            raise ValueError("Number of sparse vectors and texts must match")

        # Re-create collection if it was dropped (e.g. --clean) or lost between init and insert.
        self.create_collection(dim=len(embeddings[0]))

        client = self._get_client()
        entities: List[Dict] = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            entity: Dict = {"text": text, "embedding": embedding}
            if sparse_vectors is not None:
                entity["sparse"] = sparse_vectors[i]
            entities.append(entity)

        for start in range(0, len(entities), batch_size):
            batch = entities[start : start + batch_size]
            client.insert(collection_name=self.collection_name, data=batch)
            print(f"Inserted {min(start + batch_size, len(entities))}/{len(entities)} entities.")

    # ── read ──────────────────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: List[float],
        limit: int,
        query_text: Optional[str] = None,
        query_sparse: Optional[Dict[int, float]] = None,
        mode: str = "hybrid",
    ) -> List[Dict]:
        """
        mode="dense"  — ANN only.
        mode="hybrid" — dense ANN + sparse BM25 via RRF. Requires query_sparse (pre-computed
                        by the pipeline's SparseEncoder). Falls back to dense if not provided.
        mode="sparse" — sparse BM25 only. Requires query_sparse.
        query_text is accepted but unused — Milvus BM25 is encoded externally by the pipeline.
        """
        self._ensure_collection_loaded()
        client = self._get_client()

        if mode == "dense" or (mode == "hybrid" and query_sparse is None):
            results = client.search(
                collection_name=self.collection_name,
                data=[query_embedding],
                anns_field="embedding",
                search_params={"metric_type": "IP"},
                limit=limit,
                output_fields=["text"],
            )
            return [{"text": hit["entity"]["text"], "score": hit["distance"]} for hit in results[0]]

        if query_sparse is None:
            raise ValueError(f"query_sparse is required for mode='{mode}'")

        if mode == "sparse":
            results = client.search(
                collection_name=self.collection_name,
                data=[query_sparse],
                anns_field="sparse",
                search_params={"metric_type": "IP"},
                limit=limit,
                output_fields=["text"],
            )
            return [{"text": hit["entity"]["text"], "score": hit["distance"]} for hit in results[0]]

        if mode == "hybrid":
            dense_req = AnnSearchRequest(
                data=[query_embedding], anns_field="embedding",
                param={"metric_type": "IP"}, limit=limit,
            )
            sparse_req = AnnSearchRequest(
                data=[query_sparse], anns_field="sparse",
                param={"metric_type": "IP"}, limit=limit,
            )
            results = client.hybrid_search(
                collection_name=self.collection_name,
                reqs=[dense_req, sparse_req],
                ranker=RRFRanker(k=60),
                limit=limit,
                output_fields=["text"],
            )
            return [{"text": hit["entity"]["text"], "score": hit["distance"]} for hit in results[0]]

        raise ValueError(f"Unknown search mode: '{mode}'")
