from pathlib import Path

from src.models import Document


def load_text_file(file_path: str) -> Document:
    path = Path(file_path)

    content = path.read_text(encoding="utf-8")

    return Document(
        id=path.stem,
        content=content,
        source=path.name,
        metadata={"file_type": path.suffix.lower()},
    )
