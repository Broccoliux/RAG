import pytest

from src.embeddings import LocalEmbeddingModel


@pytest.fixture(scope="module")
def embedding_model():
    return LocalEmbeddingModel()


def test_embed_texts_returns_384_dimensional_vectors(embedding_model):
    vectors = embedding_model.embed_texts(
        ["RAG retrieves relevant information.", "Language models generate answers."]
    )

    assert len(vectors) == 2
    assert all(len(vector) == 384 for vector in vectors)


def test_embeddings_are_normalized(embedding_model):
    vector = embedding_model.embed_query("What is RAG?")
    magnitude = sum(value**2 for value in vector) ** 0.5

    assert magnitude == pytest.approx(1.0, abs=1e-5)


def test_embed_texts_returns_empty_list_for_empty_input(embedding_model):
    assert embedding_model.embed_texts([]) == []


def test_embed_query_rejects_blank_input(embedding_model):
    with pytest.raises(ValueError, match="Query cannot be empty"):
        embedding_model.embed_query("   ")
