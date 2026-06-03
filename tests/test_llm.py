from unittest.mock import Mock, patch

import pytest

from src.llm.ollama import OllamaLLM


@pytest.fixture
def mock_ollama():
    with patch("httpx.Client") as mock_client:
        client_instance = Mock()
        mock_client.return_value = client_instance

        # Create separate response mocks
        generate_response = Mock()
        generate_response.json.return_value = {"response": "test response"}

        embeddings_response = Mock()
        embeddings_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}

        # Configure post method to return different responses based on endpoint
        def mock_post(url, **kwargs):
            if url == "/api/generate":
                return generate_response
            elif url == "/api/embeddings":
                return embeddings_response
            raise ValueError(f"Unexpected URL: {url}")

        client_instance.post = Mock(side_effect=mock_post)

        llm = OllamaLLM()
        yield llm
        llm.close()


def test_ollama_generate(mock_ollama):
    response = mock_ollama.generate("test prompt")
    assert response == "test response"

    mock_ollama.client.post.assert_called_with(
        "/api/generate",
        json={
            "model": mock_ollama.model,
            "prompt": "test prompt",
            "stream": False,
            "temperature": 0.7,
            "max_tokens": 2048,
        },
    )


def test_ollama_get_embeddings(mock_ollama):
    embeddings = mock_ollama.get_embeddings("test text")
    assert embeddings == [0.1, 0.2, 0.3]

    mock_ollama.client.post.assert_called_with(
        "/api/embeddings",
        json={"model": mock_ollama.embedding_model, "prompt": "test text"},
    )


def test_ollama_error_handling(mock_ollama):
    # Test error response
    with patch.object(mock_ollama.client, "post") as mock_post:
        error_response = Mock()
        error_response.raise_for_status.side_effect = Exception("API Error")
        mock_post.return_value = error_response
        with pytest.raises(Exception):
            mock_ollama.generate("test prompt")
