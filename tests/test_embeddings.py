from unittest.mock import Mock, patch

import pytest

from src.embeddings import LocalEmbeddingModel

def test_embad_texts_returns_vectors():
  fake_model = Mock()
  fake_model.encode.return_value.tolist.return_value = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]


    with patch(
        "src.embeddings.SentenceTransformer",
        return_value=fake_model,
