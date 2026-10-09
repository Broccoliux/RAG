from typing import Any

from pydantic import BaseModel, Field
class Document(BaseModel):
    id: str
    content: str
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)
