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
