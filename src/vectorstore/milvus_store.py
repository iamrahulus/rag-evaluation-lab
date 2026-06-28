from typing import Dict, List, Optional

from pymilvus import AnnSearchRequest, DataType, MilvusClient, RRFRanker
from pymilvus.client.types import LoadState

from src.config import settings


class MilvusStore:
    def __init__(self) -> None:
        self.collection_name = settings.COLLECTION_NAME
        self._client: Optional[MilvusClient] = None

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

    def _ensure_collection_loaded(self) -> None:
        """Load the collection if it is not already in the Loaded state.

        Only needed before search/hybrid_search — insert does not require loading.
        No-ops if the collection does not exist yet.
        """
        client = self._get_client()
        if not client.has_collection(self.collection_name):
            return
        state = client.get_load_state(collection_name=self.collection_name)["state"]
        if state != LoadState.Loaded:
            client.load_collection(self.collection_name)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def is_empty(self) -> bool:
        """Check if the collection is empty or does not exist."""
        client = self._get_client()
        if not client.has_collection(self.collection_name):
            return True
        stats = client.get_collection_stats(self.collection_name)
        return bool(stats["row_count"] == 0)

    def drop_collection(self) -> None:
        """Drop the collection if it exists."""
        client = self._get_client()
        if client.has_collection(self.collection_name):
            client.drop_collection(self.collection_name)

    def create_collection(self, dim: int = settings.EMBEDDING_DIMENSION) -> None:
        """Create a new collection with dense and sparse vector fields."""
        client = self._get_client()

        if client.has_collection(self.collection_name):
            fields = client.describe_collection(self.collection_name)["fields"]
            has_sparse = any(f["name"] == "sparse" for f in fields)
            stats = client.get_collection_stats(self.collection_name)
            if stats["row_count"] > 0 and has_sparse:
                return
            client.drop_collection(self.collection_name)

        schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)

        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="IP")
        index_params.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")

        client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )

    def insert_embeddings(
        self,
        embeddings: List[List[float]],
        texts: List[str],
        sparse_vectors: Optional[List[Dict[int, float]]] = None,
        batch_size: int = 200,
    ) -> None:
        """Insert texts, dense embeddings, and optional sparse BM25 vectors."""
        if len(embeddings) != len(texts):
            raise ValueError("Number of embeddings and texts must match")
        if sparse_vectors is not None and len(sparse_vectors) != len(texts):
            raise ValueError("Number of sparse vectors and texts must match")

        # Re-create the collection if it was dropped (e.g. --clean flag) or
        # never fully created due to a server restart between init and insert.
        self.create_collection(dim=len(embeddings[0]))

        client = self._get_client()

        entities = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            entity: Dict[str, object] = {"text": text, "embedding": embedding}
            if sparse_vectors is not None:
                entity["sparse"] = sparse_vectors[i]
            entities.append(entity)

        for start in range(0, len(entities), batch_size):
            batch = entities[start : start + batch_size]
            client.insert(collection_name=self.collection_name, data=batch)
            print(f"Inserted {min(start + batch_size, len(entities))}/{len(entities)} entities.")

    def retrieve(
        self, query_embedding: List[float], limit: int
    ) -> List[Dict[str, float]]:
        """Dense-only retrieval using the query embedding."""
        self._ensure_collection_loaded()
        client = self._get_client()
        results = client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            anns_field="embedding",
            search_params={"metric_type": "IP"},
            limit=limit,
            output_fields=["text"],
        )
        return [{"text": hit["entity"]["text"], "score": hit["distance"]} for hit in results[0]]

    def hybrid_retrieve(
        self,
        query_embedding: List[float],
        query_sparse: Dict[int, float],
        limit: int,
    ) -> List[Dict[str, float]]:
        """Hybrid retrieval combining dense ANN and sparse BM25 via RRF ranking."""
        self._ensure_collection_loaded()
        client = self._get_client()

        dense_req = AnnSearchRequest(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "IP"},
            limit=limit,
        )
        sparse_req = AnnSearchRequest(
            data=[query_sparse],
            anns_field="sparse",
            param={"metric_type": "IP"},
            limit=limit,
        )

        results = client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(k=60),
            limit=limit,
            output_fields=["text"],
        )
        return [{"text": hit["entity"]["text"], "score": hit["distance"]} for hit in results[0]]
