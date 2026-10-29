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

