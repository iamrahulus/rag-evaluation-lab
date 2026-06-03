from abc import abstractmethod
from typing import List

from src.config import settings

from .base import BaseChunker


class SimpleChunker(BaseChunker):
    def __init__(
        self,
        chunk_size: int = settings.DEFAULT_CHUNK_SIZE,
        overlap: int = settings.DEFAULT_CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap

    @abstractmethod
    def chunk_text(self, text: str) -> List[str]:
        """Simple chunking to be implemented."""
        pass


class AdvancedChunker(BaseChunker):
    @abstractmethod
    def chunk_text(self, text: str) -> List[str]:
        """Advanced chunking to be implemented."""
        pass
