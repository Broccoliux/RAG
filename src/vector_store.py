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


        
