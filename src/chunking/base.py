from abc import ABC, abstractmethod
from typing import List


class BaseChunker(ABC):
    """Base class for implementing different chunking strategies."""

    @abstractmethod
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks according to the strategy."""
        pass
