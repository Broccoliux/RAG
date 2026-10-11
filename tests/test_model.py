from src.models import Document, Chunk, RetrievalResult, Citation, Answer
from src.chunking import chunk_document
import pytest
from pydantic import ValidationError

from pathlib import Path

from src.ingestion.text_loader import load_text_file


def test_document_defaults():
    doc = Document(
        id="doc-1",
        content="Hello RAG",
        source="notes.md",
    )
    assert doc.metadata == {}


def test_chunk_links_to_document():
    chunk = Chunk(
        id="chunk-1",
        document_id="doc-1",
        content="Hello RAG",
    )
    assert chunk.document_id == "doc-1"


def test_retrieval_result_stores_score():
    chunk = Chunk(
        id="chunk-1",
        document_id="doc-1",
        content="Hello RAG",
    )
    result = RetrievalResult(chunk=chunk, score=0.9)
    assert result.score == 0.9


def test_citation_stores_page():
    citation = Citation(
        document_id="doc-1",
        source="notes.pdf",
        chunk_id="chunk-1",
        page=3,
    )
    assert citation.page == 3


def test_answer_defaults_to_empty_citations():
    answer = Answer(
        question="What is RAG?",
        content="RAG retrieves relevant information.",
    )
    assert answer.citations == []

def test_document_rejects_missing_required_fields():
    with pytest.raises(ValidationError):
        Document(id="doc-1", source="notes.md")


def test_text_loader_reads_file(tmp_path):
    file_path = tmp_path / "example.txt"
    file_path.write_text("Hello RAG", encoding="utf-8")

    doc = load_text_file(str(file_path))

    assert doc.content == "Hello RAG"
    assert doc.source == "example.txt"
    assert doc.metadata["file_type"] == ".txt"


def test_text_loader_rejects_missing_file(tmp_path):
    missing_file = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError):
        load_text_file(str(missing_file))

def test_text_loader_rejects_empty_file(tmp_path):
    file_path = tmp_path / "empty.txt"
    file_path.write_text("   \n", encoding="utf-8")

    with pytest.raises(ValueError, match="File is empty"):
        load_text_file(str(file_path))

def test_chunker_splits_large_document():
    doc = Document(
        id="test-doc",
        content="RAG retrieves relevant information. " * 200,
        source="test.txt",
    )

    chunks = chunk_document(doc, chunk_size=100, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(chunk.document_id == doc.id for chunk in chunks)
    assert all(chunk.content for chunk in chunks)
    assert chunks[0].id == "test-doc-chunk-0"
    assert chunks[1].id == "test-doc-chunk-1"


def test_chunker_preserves_small_document():
    content = "RAG retrieves relevant information.\n"
    doc = Document(
        id="small-doc",
        content=content,
        source="small.txt",
    )

    chunks = chunk_document(doc)

    assert len(chunks) == 1
    assert chunks[0].content == content
    assert chunks[0].metadata["start_index"] == 0

def test_chunker_preserves_overlap():
    doc = Document(
        id="overlap-doc",
        content=" ".join(f"word{i}" for i in range(300)),
        source="overlap.txt",
    )
    chunks = chunk_document(doc, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1

    first_end = chunks[0].metadata["start_index"] + len(chunks[0].content)
    second_start = chunks[1].metadata["start_index"]
    assert second_start < first_end


def test_chunker_preserves_metadata():
    doc = Document(
        id="metadata-doc",
        content="RAG retrieves relevant information. " * 200,
        source="metadata.txt",
        metadata={"file_type": ".txt"},
    )
    chunks = chunk_document(doc, chunk_size=100, chunk_overlap=20)

    assert len(chunks) > 1

    for index, chunk in enumerate(chunks):
        assert chunk.metadata["source"] == doc.source
        assert chunk.metadata["file_type"] == ".txt"
        assert chunk.metadata["chunk_index"] == index
        assert isinstance(chunk.metadata["start_index"], int)
        assert chunk.metadata["start_index"] >= 0
