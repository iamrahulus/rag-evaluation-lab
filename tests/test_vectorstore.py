from unittest.mock import MagicMock, patch

import pytest

from src.vectorstore.milvus_store import MilvusStore


@pytest.fixture
def mock_milvus():
    with patch("src.vectorstore.milvus_store.connections") as mock_conn, patch(
        "src.vectorstore.milvus_store.Collection"
    ) as mock_coll, patch("src.vectorstore.milvus_store.utility") as mock_utility:

        mock_utility.has_collection.return_value = False
        mock_collection = MagicMock()
        mock_coll.return_value = mock_collection

        yield {
            "connections": mock_conn,
            "collection": mock_collection,
            "utility": mock_utility,
        }


def test_milvus_store_init(mock_milvus):
    MilvusStore()
    mock_milvus["connections"].connect.assert_called_once()


def test_milvus_store_create_collection(mock_milvus):
    store = MilvusStore()
    store.create_collection()

    # Check if collection exists
    mock_milvus["utility"].has_collection.assert_called_once()

    # Check collection creation
    mock_milvus["collection"].create_index.assert_called_once()
    mock_milvus["collection"].load.assert_called_once()


def test_milvus_store_insert_embeddings(mock_milvus):
    store = MilvusStore()

    embeddings = [[1.0, 2.0], [3.0, 4.0]]
    texts = ["text1", "text2"]

    store.insert_embeddings(embeddings, texts)

    # Check insert was called with correct entities
    mock_milvus["collection"].insert.assert_called_once()
    mock_milvus["collection"].flush.assert_called_once()


def test_milvus_store_retrieve(mock_milvus):
    store = MilvusStore()

    # Mock search results
    mock_hit = MagicMock()
    mock_hit.entity.get.return_value = "test text"
    mock_hit.score = 0.95
    mock_milvus["collection"].search.return_value = [[mock_hit]]

    results = store.retrieve([1.0, 2.0], limit=1)

    assert len(results) == 1
    assert results[0]["text"] == "test text"
    assert results[0]["score"] == 0.95
    mock_milvus["collection"].search.assert_called_once()


def test_milvus_store_insert_embeddings_validation(mock_milvus):
    store = MilvusStore()

    # Test mismatched lengths
    with pytest.raises(ValueError, match="Number of embeddings and texts must match"):
        store.insert_embeddings([[1.0, 2.0]], ["text1", "text2"])
