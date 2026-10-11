from src.models import Document, Chunk, RetrievalResult, Citation, Answer

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
