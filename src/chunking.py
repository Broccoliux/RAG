from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.models import Chunk, Document


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

    if not document.content:
        return []

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        strip_whitespace=False,
    )

    text_chunks = splitter.split_text(document.content)

    chunks = []
    search_from = 0

    for index, text in enumerate(text_chunks):
        start = document.content.find(text, search_from)

        if start == -1:
            raise ValueError(
                "Could not map a chunk back to its original document"
            )

        chunks.append(
            Chunk(
                id=f"{document.id}-chunk-{index}",
                document_id=document.id,
                content=text,
                metadata={
                    **document.metadata,
                    "source": document.source,
                    "chunk_index": index,
                    "start_index": start,
                },
            )
        )

        search_from = start + 1

    return chunks
