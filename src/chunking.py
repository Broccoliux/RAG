from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.models import Chunk, Document


def chunk_document(
    document: Document,
    chunk_size: int = 400,
    chunk_overlap: int = 60,
) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        add_start_index=True,
        strip_whitespace=False,
    )

    split_documents = splitter.create_documents(
        texts=[document.content],
        metadatas=[document.metadata | {"source": document.source}],
    )

    chunks = []

    for index, split_document in enumerate(split_documents):
        chunks.append(
            Chunk(
                id=f"{document.id}-chunk-{index}",
                document_id=document.id,
                content=split_document.page_content,
                metadata={
                    **split_document.metadata,
                    "chunk_index": index,
                    "start_index": split_document.metadata["start_index"],
                },
            )
        )

    return chunks
