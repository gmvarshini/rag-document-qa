"""Document ingestion: load raw text files and split them into chunks."""

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import AppConfig, get_config

SOURCE_DIRECTORY = Path("data/source_documents")


def load_and_chunk_documents(
    source_dir: Path = SOURCE_DIRECTORY,
    config: AppConfig | None = None,
) -> list[Document]:
    """Load every text file in a directory and split it into overlapping chunks.

    Each ``.txt`` file under ``source_dir`` is read as UTF-8 and split with
    LangChain's :class:`RecursiveCharacterTextSplitter` using the chunk size and
    overlap from configuration. Every resulting chunk is returned as a LangChain
    :class:`~langchain_core.documents.Document` whose metadata records the source
    filename and the chunk index within that file.

    Args:
        source_dir: Directory containing the ``.txt`` source documents. Defaults
            to ``data/source_documents``.
        config: Optional pipeline configuration. When omitted, configuration is
            loaded from the environment via :func:`~src.config.get_config`.

    Returns:
        list[Document]: The chunked documents, each carrying ``source`` and
        ``chunk_index`` metadata.

    Raises:
        FileNotFoundError: If ``source_dir`` does not exist.
        ValueError: If ``source_dir`` contains no ``.txt`` files.
    """
    if config is None:
        config = get_config()

    if not source_dir.exists():
        raise FileNotFoundError(
            f"Source directory does not exist: {source_dir.resolve()}"
        )

    text_files = sorted(source_dir.glob("*.txt"))
    if not text_files:
        raise ValueError(
            f"No .txt files found in source directory: {source_dir.resolve()}"
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        length_function=len,
    )

    chunked_documents: list[Document] = []
    for text_file in text_files:
        raw_text = text_file.read_text(encoding="utf-8")
        chunks = splitter.split_text(raw_text)
        for chunk_index, chunk in enumerate(chunks):
            chunked_documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": text_file.name,
                        "chunk_index": chunk_index,
                    },
                )
            )

    return chunked_documents
