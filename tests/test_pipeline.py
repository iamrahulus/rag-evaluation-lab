from unittest.mock import Mock

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
