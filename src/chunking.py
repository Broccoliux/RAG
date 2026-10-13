from tiktoken import get_encoding
from src.models import Chunk, Document


_ENCODING = get_encoding("cl100k_base")
_SEPARATORS = ["\n\n", "\n", " ", ""]


def _token_count(text: str) -> int:
    return len(_ENCODING.encode(text))


def _split_range(
    text: str,
    start: int,
    end: int,
    separators: list[str],
    chunk_size: int,
) -> list[tuple[int, int]]:
    """Split a source range while retaining exact character positions."""
    if start >= end:
        return []

    if _token_count(text[start:end]) <= chunk_size:
        return [(start, end)]

    separator = next(
        (item for item in separators if item == "" or item in text[start:end]),
        "",
    )

    if separator == "":
        ranges = [(position, position + 1) for position in range(start, end)]
        for left, right in ranges:
            if _token_count(text[left:right]) > chunk_size:
                raise ValueError(
                    "chunk_size is too small to fit a single character"
                )
        return ranges

    remaining_separators = separators[separators.index(separator) + 1 :]
    parts = []
    cursor = start

    while True:
        found = text.find(separator, cursor, end)
        if found == -1:
            break

        boundary = found + len(separator)
        parts.append((cursor, boundary))
        cursor = boundary

    if cursor < end:
        parts.append((cursor, end))

    result = []

    for left, right in parts:
        if _token_count(text[left:right]) <= chunk_size:
            result.append((left, right))
        else:
            result.extend(
                _split_range(
                    text,
                    left,
                    right,
                    remaining_separators,
                    chunk_size,
                )
            )

    return result


def _merge_ranges(
    text: str,
    ranges: list[tuple[int, int]],
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[int, int]]:
    """Merge adjacent ranges while preserving token limits and overlap."""
    if not ranges:
        return []

    merged = []
    current = []

    for part in ranges:
        if current:
            candidate_start = current[0][0]
            candidate_end = part[1]

            if _token_count(text[candidate_start:candidate_end]) > chunk_size:
                merged.append((current[0][0], current[-1][1]))

                while current and (
                    _token_count(text[current[0][0] : current[-1][1]])
                    > chunk_overlap
                    or _token_count(text[current[0][0] : part[1]])
                    > chunk_size
                ):
                    current.pop(0)

        current.append(part)

    if current:
        merged.append((current[0][0], current[-1][1]))

    return merged


def chunk_document(
    document: Document,
    chunk_size: int = 400,
    chunk_overlap: int = 60,
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    content = document.content
    if not content:
        return []

    ranges = _split_range(
        content,
        0,
        len(content),
        _SEPARATORS,
        chunk_size,
    )
    chunk_ranges = _merge_ranges(
        content,
        ranges,
        chunk_size,
        chunk_overlap,
    )

    chunks = []

    for index, (start, end) in enumerate(chunk_ranges):
        chunk_content = content[start:end]

        chunks.append(
            Chunk(
                id=f"{document.id}-chunk-{index}",
                document_id=document.id,
                content=chunk_content,
                metadata={
                    **document.metadata,
                    "source": document.source,
                    "chunk_index": index,
                    "start_index": start,
                },
            )
        )

    return chunks
