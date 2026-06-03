from typing import Any, Dict, List, Optional

import httpx

from src.config import settings

from .base import BaseLLM


class OllamaLLM(BaseLLM):
    def __init__(
        self,
        model: str = settings.LLM_MODEL,
        embedding_model: str = settings.EMBEDDING_MODEL,
        base_url: Optional[str] = None,
    ) -> None:
        self.model = model
        self.embedding_model = embedding_model
        self.base_url = base_url or settings.OLLAMA_BASE_URL
        self.client = httpx.Client(base_url=self.base_url, timeout=60.0)

    def generate(
        self,
        prompt: str,
        temperature: float = settings.DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = settings.MAX_TOKENS,
        **kwargs: Any
    ) -> str:
        """Generate text using Ollama API."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }

        response = self.client.post("/api/generate", json=payload)
        response.raise_for_status()

        result: Dict[str, str] = response.json()
        return result["response"]

    def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings using Ollama API."""
        payload = {"model": self.embedding_model, "prompt": text}

        response = self.client.post("/api/embeddings", json=payload)
        response.raise_for_status()

        result: Dict[str, List[float]] = response.json()
        return result["embedding"]

    def close(self) -> None:
        self.client.close()
