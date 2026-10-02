"""Document ingestion: load text and PDF files and split them into chunks.

Supported formats:

* ``.txt`` files are read as UTF-8 text.
* ``.pdf`` files are read page by page with ``pypdf``. Each page is split on its
  own, so every chunk can cite the exact page it came from.
"""

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from src.config import AppConfig, get_config

SOURCE_DIRECTORY = Path("data/source_documents")
SUPPORTED_SUFFIXES = {".txt", ".pdf"}


def _read_pages(path: Path) -> list[tuple[str, int | None]]:
    """Return the text of a file as a list of ``(text, page_number)`` pairs.

    Text files are one "page" with no page number. PDF files return one entry
    per page, numbered from 1. Pages with no extractable text (for example a
    scanned image without OCR) are skipped.

    Args:
        path: A ``.txt`` or ``.pdf`` file.

    Returns:
        list[tuple[str, int | None]]: The text and page number of each page.
    """
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        pages = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append((text, page_number))
        return pages
    return [(path.read_text(encoding="utf-8"), None)]


def load_and_chunk_documents(
    source_dir: Path = SOURCE_DIRECTORY,
    config: AppConfig | None = None,
) -> list[Document]:
    """Load every text and PDF file in a directory and split it into chunks.

    Each ``.txt`` file under ``source_dir`` is read as UTF-8, and each ``.pdf``
    file is read page by page. The text is split with
    LangChain's :class:`RecursiveCharacterTextSplitter` using the chunk size and
    overlap from configuration. Every resulting chunk is returned as a LangChain
    :class:`~langchain_core.documents.Document` whose metadata records the source
    filename, the chunk index within that file and, for PDFs, the page number.

    Args:
        source_dir: Directory containing the ``.txt`` source documents. Defaults
            to ``data/source_documents``.
        config: Optional pipeline configuration. When omitted, configuration is
            loaded from the environment via :func:`~src.config.get_config`.

    Returns:
        list[Document]: The chunked documents, each carrying ``source`` and
        ``chunk_index`` metadata, plus ``page`` for chunks from a PDF.

    Raises:
        FileNotFoundError: If ``source_dir`` does not exist.
        ValueError: If ``source_dir`` contains no ``.txt`` or ``.pdf`` files.
    """
    if config is None:
        config = get_config()

    if not source_dir.exists():
        raise FileNotFoundError(
            f"Source directory does not exist: {source_dir.resolve()}"
        )

    source_files = sorted(
        path
        for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not source_files:
        raise ValueError(
            f"No .txt or .pdf files found in source directory: {source_dir.resolve()}"
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        length_function=len,
    )

    chunked_documents: list[Document] = []
    for source_file in source_files:
        chunk_index = 0
        for text, page_number in _read_pages(source_file):
            for chunk in splitter.split_text(text):
                metadata = {"source": source_file.name, "chunk_index": chunk_index}
                if page_number is not None:
                    metadata["page"] = page_number
                chunked_documents.append(
                    Document(page_content=chunk, metadata=metadata)
                )
                chunk_index += 1

    return chunked_documents
