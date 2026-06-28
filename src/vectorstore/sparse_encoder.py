from abc import ABC, abstractmethod


class SparseEncoder(ABC):

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Return True once fit() has been called."""
        ...

    @abstractmethod
    def fit(self, corpus: list[str]) -> None:
        """Compute corpus statistics — must be called before encode_documents."""
        ...

    @abstractmethod
    def encode_documents(self, texts: list[str]) -> list[dict]:
        """Encode chunks at ingestion time."""
        ...

    @abstractmethod
    def encode_queries(self, texts: list[str]) -> list[dict]:
        """Encode queries at retrieval time."""
        ...

    def encode_query(self, text: str) -> dict:
        """Encode a single query string. Delegates to encode_queries."""
        return self.encode_queries([text])[0]

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist model state — IDF weights and vocabulary."""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Restore model state for query-time encoding."""
        ...
