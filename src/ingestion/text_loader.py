from pathlib import Path
import hashlib
from src.models import Document


def load_text_file(file_path: str) -> Document:
    path = Path(file_path)

    content = path.read_text(encoding="utf-8")

    if not content.strip():
      raise ValueError(f"File is empty: {path}")

    return Document(
        id=hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest(),
        content=content,
        source=path.name,
        metadata={"file_type": path.suffix.lower()},
    )
