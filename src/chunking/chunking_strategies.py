from typing import List
from abc import abstractmethod

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

    def chunk_text(self, text: str) -> List[str]:
        """Simple chunking to be implemented."""
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.overlap):
            chunk = text[i : i + self.chunk_size]
            chunks.append(chunk)
        return chunks


class AdvancedChunker(BaseChunker):
    def __init__(
        self,
        chunk_size: int = settings.DEFAULT_CHUNK_SIZE,
        overlap: int = settings.DEFAULT_CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    """
    Paragraph-aware chunker. Splits on natural paragraph boundaries
    rather than fixed character counts, preserving semantic coherence.
    Falls back to sentence boundaries if paragraphs exceed max_chunk_size.
    """
    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences on punctuation boundaries."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s for s in sentences if s.strip()]

    def _chunk_large_paragraph(self, para: str) -> list[str]:
        """Fall back to sentence-boundary chunking for oversized paragraphs."""
        sentences = self._split_into_sentences(para)
        chunks = []
        current = ""

        for sentence in sentences:
            if len(current) + len(sentence) <= self.chunk_size:
                current += sentence + " "
            else:
                if current:
                    chunks.append(current.strip())
                current = sentence + " "

        if current:
            chunks.append(current.strip())

        return chunks    

    def chunk_text(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        current_chunk = ""
        overlap_buffer = ""  # carries tail of previous chunk

        for para in paragraphs:
            if len(para) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    overlap_buffer = self._get_overlap(current_chunk)
                    current_chunk = ""
                for sentence_chunk in self._chunk_large_paragraph(para):
                    chunks.append(sentence_chunk)
                overlap_buffer = self._get_overlap(para)

            elif len(overlap_buffer) + len(current_chunk) + len(para) + 2 <= self.chunk_size:
                current_chunk += para + "\n\n"

            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    overlap_buffer = self._get_overlap(current_chunk)
                # Start new chunk with overlap from previous + current para
                current_chunk = overlap_buffer + para + "\n\n"

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _get_overlap(self, text: str) -> str:
        """Return the last overlap_size characters of text as overlap context."""
        if not self.overlap or len(text) <= self.overlap:
            return ""
        # Trim to last overlap_size chars but start at a word boundary
        tail = text[-self.overlap:]
        first_space = tail.find(" ")
        if first_space > 0:
            tail = tail[first_space + 1:]
        return tail + " "