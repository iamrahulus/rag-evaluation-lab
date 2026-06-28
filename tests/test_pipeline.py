from unittest.mock import Mock, patch

import pytest

from src.rag.pipeline import RAGPipeline


@pytest.fixture
def mock_components():
    llm = Mock()
    chunker = Mock()
    vector_store = Mock()

    # Setup mock returns
    llm.get_embeddings.return_value = [0.1, 0.2, 0.3]
    llm.generate.return_value = "test answer"
    vector_store.retrieve.return_value = [
        {"text": "relevant text 1", "score": 0.9},
        {"text": "relevant text 2", "score": 0.8},
    ]

    return {"llm": llm, "chunker": chunker, "vector_store": vector_store}


def test_pipeline_query(mock_components):
    pipeline = RAGPipeline(
        llm=mock_components["llm"],
        chunker=mock_components["chunker"],
        vector_store=mock_components["vector_store"],
    )

    result = pipeline.query("test question")

    mock_components["llm"].get_embeddings.assert_called_once_with("test question")
    mock_components["vector_store"].retrieve.assert_called_once()
    mock_components["llm"].generate.assert_called_once()
    assert result == "test answer"


def test_add_documents_saves_bm25_model(mock_components):
    """BM25 model is persisted after ingestion so --eval sessions can load it."""
    bm25 = Mock()
    chunker = mock_components["chunker"]
    chunker.chunk_text.return_value = ["chunk1", "chunk2"]

    with patch("src.rag.pipeline.os.path.exists", return_value=False):
        pipeline = RAGPipeline(
            llm=mock_components["llm"],
            chunker=chunker,
            vector_store=mock_components["vector_store"],
            sparse_encoder=bm25,
        )
        pipeline.add_documents(["doc1"])

    bm25.fit.assert_called_once()
    bm25.save.assert_called_once()


def test_init_loads_bm25_model_when_file_exists(mock_components):
    """BM25 model is auto-loaded at startup so hybrid search works without re-ingesting."""
    bm25 = Mock()
    bm25.is_fitted = False

    with patch("src.rag.pipeline.os.path.exists", return_value=True):
        RAGPipeline(
            llm=mock_components["llm"],
            chunker=mock_components["chunker"],
            vector_store=mock_components["vector_store"],
            sparse_encoder=bm25,
        )

    bm25.load.assert_called_once()


def test_init_skips_bm25_load_when_no_file(mock_components):
    """No load attempt is made when the model file does not exist yet."""
    bm25 = Mock()

    with patch("src.rag.pipeline.os.path.exists", return_value=False):
        RAGPipeline(
            llm=mock_components["llm"],
            chunker=mock_components["chunker"],
            vector_store=mock_components["vector_store"],
            sparse_encoder=bm25,
        )

    bm25.load.assert_not_called()
