from typing import Dict, List

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from src.config import settings


class MilvusStore:
    def __init__(self) -> None:
        self.collection_name = settings.COLLECTION_NAME
        self._connect()

    def _connect(self) -> None:
        """Establish connection to Milvus"""
        connections.connect(uri=settings.MILVUS_DB_PATH)

    def create_collection(self, dim: int = settings.EMBEDDING_DIMENSION) -> None:
        """Create a new collection."""
        if utility.has_collection(self.collection_name):
            Collection(self.collection_name).drop()

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        ]

        schema = CollectionSchema(fields=fields, description="Document store")
        collection = Collection(name=self.collection_name, schema=schema)

        index_params = {"metric_type": "IP", "index_type": "AUTOINDEX"}
        collection.create_index(field_name="embedding", index_params=index_params)
        collection.load()

    def insert_embeddings(
        self, embeddings: List[List[float]], texts: List[str]
    ) -> None:
        """Insert text and their embeddings into the collection."""
        if len(embeddings) != len(texts):
            raise ValueError("Number of embeddings and texts must match")

        collection = Collection(self.collection_name)

        entities = [
            {"text": text, "embedding": embedding}
            for text, embedding in zip(texts, embeddings)
        ]

        collection.insert(entities)
        collection.flush()

    def retrieve(
        self, query_embedding: List[float], limit: int
    ) -> List[Dict[str, float]]:
        """Retrieve similar documents using the query embedding."""
        collection = Collection(self.collection_name)

        search_params = {"metric_type": "IP", "params": {}}

        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=limit,
            output_fields=["text"],
        )

        hits = []
        for hit in results[0]:
            hits.append({"text": hit.entity.get("text"), "score": hit.score})

        return hits
