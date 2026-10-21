from unittest.mock import Mock, patch

import pytest

from src.embeddings import LocalEmbeddingModel


def test_embed_texts_returns_vectors():
    fake_model = Mock()
    fake_model.encode.return_value.tolist.return_value = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    with patch(
        "src.embeddings.SentenceTransformer",
        return_value=fake_model,
    ):
        model = LocalEmbeddingModel()

    vectors = model.embed_texts(["first text", "second text"])

    assert len(vectors) == 2
    assert len(vectors[0]) == 3
    assert len(vectors[1]) == 3


def test_embed_texts_returns_empty_list_for_empty_input():
    with patch("src.embeddings.SentenceTransformer"):
        model = LocalEmbeddingModel()

    assert model.embed_texts([]) == []
    model.model.encode.assert_not_called()


def test_embed_query_rejects_blank_input():
    with patch("src.embeddings.SentenceTransformer"):
        model = LocalEmbeddingModel()

    with pytest.raises(ValueError, match="Query cannot be empty"):
        model.embed_query("   ")


def test_embed_query_returns_single_vector():
    fake_model = Mock()
    fake_model.encode.return_value.tolist.return_value = [
        [0.1, 0.2, 0.3]
    ]

    with patch(
        "src.embeddings.SentenceTransformer",
        return_value=fake_model,
    ):
        model = LocalEmbeddingModel()

    vector = model.embed_query("What is RAG?")

    assert vector == [0.1, 0.2, 0.3]
