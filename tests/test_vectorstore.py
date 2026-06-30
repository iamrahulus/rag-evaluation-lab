from unittest.mock import MagicMock, patch

import pytest

from src.vectorstore.milvus_store import MilvusStore


@pytest.fixture
def mock_client():
    with patch("src.vectorstore.milvus_store.MilvusClient") as mock_cls:
        client = MagicMock()
        mock_cls.return_value = client
        mock_cls.create_schema.return_value = MagicMock()
        mock_cls.prepare_index_params.return_value = MagicMock()
        client.has_collection.return_value = False
        client.get_collection_stats.return_value = {"row_count": 0}
        yield client


def test_milvus_store_init(mock_client):
    MilvusStore()
    mock_client.assert_not_called()


def test_milvus_store_create_collection(mock_client):
    store = MilvusStore()
    store.create_collection()

    mock_client.has_collection.assert_called_once()
    mock_client.create_collection.assert_called_once()


def test_milvus_store_upsert(mock_client):
    store = MilvusStore()

    embeddings = [[1.0, 2.0], [3.0, 4.0]]
    texts = ["text1", "text2"]

    store.upsert(texts=texts, embeddings=embeddings)

    mock_client.insert.assert_called_once()


def test_milvus_store_upsert_with_sparse(mock_client):
    store = MilvusStore()

    embeddings = [[1.0, 2.0], [3.0, 4.0]]
    texts = ["text1", "text2"]
    sparse = [{0: 0.5, 1: 0.3}, {2: 0.8}]

    store.upsert(texts=texts, embeddings=embeddings, sparse_vectors=sparse)

    call_data = mock_client.insert.call_args[1]["data"]
    assert call_data[0]["sparse"] == sparse[0]
    assert call_data[1]["sparse"] == sparse[1]


def test_milvus_store_upsert_validation(mock_client):
    store = MilvusStore()

    with pytest.raises(ValueError, match="Number of embeddings and texts must match"):
        store.upsert(texts=["text1", "text2"], embeddings=[[1.0, 2.0]])


def test_milvus_store_search_dense(mock_client):
    store = MilvusStore()

    mock_client.search.return_value = [[
        {"entity": {"text": "test text"}, "distance": 0.95}
    ]]

    results = store.search([1.0, 2.0], limit=1, mode="dense")

    assert len(results) == 1
    assert results[0]["text"] == "test text"
    assert results[0]["score"] == 0.95
    mock_client.search.assert_called_once()


def test_milvus_store_search_hybrid(mock_client):
    store = MilvusStore()

    mock_client.hybrid_search.return_value = [[
        {"entity": {"text": "hybrid result"}, "distance": 0.88}
    ]]

    results = store.search(
        [1.0, 2.0], limit=1,
        query_sparse={0: 0.5, 3: 0.2},
        mode="hybrid",
    )

    assert results[0]["text"] == "hybrid result"
    mock_client.hybrid_search.assert_called_once()


def test_milvus_store_search_hybrid_falls_back_to_dense_without_sparse(mock_client):
    """When mode=hybrid but no query_sparse provided, falls back to dense rather than raising."""
    store = MilvusStore()

    mock_client.search.return_value = [[
        {"entity": {"text": "dense fallback"}, "distance": 0.75}
    ]]

    results = store.search([1.0, 2.0], limit=1, mode="hybrid")

    assert results[0]["text"] == "dense fallback"
    mock_client.hybrid_search.assert_not_called()
