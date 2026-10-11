from src.models import Document, Chunk, RetrievalResult, Citation, Answer

def test_document_model():
    doc = Document(
        id="doc-1",
        content=" Hello RAG",
        source="notes.md",
    )
