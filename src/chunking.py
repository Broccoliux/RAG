from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.models import Chunk, Document


def _find_overlap(previous: str, current: str, max_tokens: int) -> int:
    """Return the longest matching suffix/prefix within the token limit."""
    from tiktoken import get_encoding

    encoding = get_encoding("cl100k_base")

    for length in range(min(len(previous), len(current)), 0, -1):
        if previous.endswith(current[:length]):
            if len(encoding.encode(current[:length])) <= max_tokens:
                return length

    return 0


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

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        strip_whitespace=False,
    )

    split_documents = splitter.create_documents(
        texts=[document.content],
        metadatas=[document.metadata | {"source": document.source}],
    )

    chunks = []
    previous_start = 0

    for index, split_document in enumerate(split_documents):
        content = split_document.page_content

        if index == 0:
            start_index = document.content.find(content)
        else:
            previous_content = chunks[-1].content
            overlap_length = _find_overlap(
                previous_content, content, chunk_overlap
            )
            start_index = (
                previous_start + len(previous_content) - overlap_length
            )

        if (
            start_index < 0
            or document.content[start_index : start_index + len(content)]
            != content
        ):
            raise ValueError(
                f"Could not determine a reliable offset for chunk {index}"
            )

        chunks.append(
            Chunk(
                id=f"{document.id}-chunk-{index}",
                document_id=document.id,
                content=content,
                metadata={
                    **split_document.metadata,
                    "chunk_index": index,
                    "start_index": start_index,
                },
            )
        )
        previous_start = start_index

    return chunks
