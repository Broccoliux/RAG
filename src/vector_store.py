import numpy as np

from src.models import Chunk, RetrievalResult


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._embeddings: np.ndarray | None = None

            def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have exactly one embedding")

        if not chunks:
            return

 new_embeddings = np.asarray(embeddings, dtype=np.float32)

        if new_embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array")

        if not np.isfinite(new_embeddings).all():
            raise ValueError("Embeddings must contain only finite values")

        if np.any(np.linalg.norm(new_embeddings, axis=1) == 0):
            raise ValueError("Embeddings cannot be zero vectors")

        if self._embeddings is not None:
            if new_embeddings.shape[1] != self._embeddings.shape[1]:
                raise ValueError("Embedding dimensions must match")

            self._embeddings = np.vstack(
                [self._embeddings, new_embeddings]
            )
        else:
            self._embeddings = new_embeddings

        self._chunks.extend(chunks)


          def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        if not self._chunks or self._embeddings is None:
            return []

        query = np.asarray(query_embedding, dtype=np.float32)

        if query.ndim != 1:
            raise ValueError("Query embedding must be a 1D vector")

        if query.shape[0] != self._embeddings.shape[1]:
            raise ValueError("Query and stored embedding dimensions must match")

        if not np.isfinite(query).all():
            raise ValueError("Query embedding must contain only finite values")

query_norm = np.linalg.norm(query)

        if query_norm == 0:
            raise ValueError("Query embedding cannot be a zero vector")

        query = query / query_norm

        stored_norms = np.linalg.norm(self._embeddings, axis=1)
        normalized_embeddings = self._embeddings / stored_norms[:, None]

        scores = normalized_embeddings @ query

        result_count = min(top_k, len(self._chunks))
        candidate_indices = np.argpartition(
            scores, len(scores) - result_count
        )[-result_count:]

          ranked_indices = candidate_indices[
            np.argsort(scores[candidate_indices])[::-1]
        ]

        return [
            RetrievalResult(
                chunk=self._chunks[index],
                score=float(scores[index]),
            )
            for index in ranked_indices
        ]
