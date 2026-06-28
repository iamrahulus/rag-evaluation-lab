import scipy.sparse
from pymilvus.model.sparse import BM25EmbeddingFunction

from .sparse_encoder import SparseEncoder


class MilvusBM25Encoder(SparseEncoder):

    def __init__(self) -> None:
        self._model = BM25EmbeddingFunction()
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, corpus: list[str]) -> None:
        self._model.fit(corpus)
        self._fitted = True

    @staticmethod
    def _sparse_to_dict_list(mat: scipy.sparse.csr_array) -> list[dict]:
        """Convert a scipy csr_array (one row per text) to a list of {col: val} dicts."""
        csr = mat.tocsr()
        result = []
        for i in range(csr.shape[0]):
            start, end = int(csr.indptr[i]), int(csr.indptr[i + 1])
            result.append({
                int(c): float(v)
                for c, v in zip(csr.indices[start:end], csr.data[start:end])
            })
        return result

    def encode_documents(self, texts: list[str]) -> list[dict]:
        return self._sparse_to_dict_list(self._model.encode_documents(texts))

    def encode_queries(self, texts: list[str]) -> list[dict]:
        return self._sparse_to_dict_list(self._model.encode_queries(texts))

    def save(self, path: str) -> None:
        self._model.save(path)

    def load(self, path: str) -> None:
        self._model.load(path)
        self._fitted = True
