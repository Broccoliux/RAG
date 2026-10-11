from typing import Any

from pydantic import BaseModel, Field

class Document(BaseModel):
    id: str
    content: str
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)

class Chunk(BaseModel):
    id: str
    document_id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)

class RetrievalResult(BaseModel):
    chunk: Chunk
    score: float

class Citation(BaseModel):
    document_id: str
    source: str
    chunk_id: str
    page: int | None = None


class Answer(BaseModel):
    question: str
    content: str
    citations: list[Citation] = Field(default_factory=list)
