from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseLLM(ABC):
    """Base class for LLM implementations."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> str:
        """Generate text from the LLM."""
        pass

    @abstractmethod
    def get_embeddings(self, text: str) -> list[float]:
        """Get embeddings for the input text."""
        pass
