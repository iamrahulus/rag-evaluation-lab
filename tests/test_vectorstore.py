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


def test_milvus_store_insert_embeddings(mock_client):
    store = MilvusStore()

    embeddings = [[1.0, 2.0], [3.0, 4.0]]
    texts = ["text1", "text2"]

    store.insert_embeddings(embeddings, texts)

    mock_client.insert.assert_called_once()


def test_milvus_store_retrieve(mock_client):
    store = MilvusStore()

    mock_client.search.return_value = [[
        {"entity": {"text": "test text"}, "distance": 0.95}
    ]]

    results = store.retrieve([1.0, 2.0], limit=1)

    assert len(results) == 1
    assert results[0]["text"] == "test text"
    assert results[0]["score"] == 0.95
    mock_client.search.assert_called_once()


def test_milvus_store_insert_embeddings_validation(mock_client):
    store = MilvusStore()

    with pytest.raises(ValueError, match="Number of embeddings and texts must match"):
        store.insert_embeddings([[1.0, 2.0]], ["text1", "text2"])
