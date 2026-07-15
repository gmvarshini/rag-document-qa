"""Vector store construction and similarity search over document chunks."""

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import AppConfig, get_config


def _build_embeddings(config: AppConfig) -> HuggingFaceEmbeddings:
    """Create the local sentence-transformers embedding function.

    Args:
        config: Pipeline configuration providing the embedding model name.

    Returns:
        HuggingFaceEmbeddings: An embedding function that runs entirely locally.
    """
    return HuggingFaceEmbeddings(model_name=config.embedding_model)


def build_vector_store(
    documents: list[Document],
    config: AppConfig | None = None,
) -> FAISS:
    """Embed the chunked documents, build a FAISS index, and persist it to disk.

    Embeddings are produced with a local sentence-transformers model wrapped by
    LangChain's :class:`HuggingFaceEmbeddings`. The resulting FAISS index is
    saved to the path defined in configuration so later runs can load it instead
    of recomputing every embedding.

    Args:
        documents: The chunked documents to index. Must be non-empty.
        config: Optional pipeline configuration. When omitted, configuration is
            loaded from the environment via :func:`~src.config.get_config`.

    Returns:
        FAISS: The in-memory vector store that was also saved to disk.

    Raises:
        ValueError: If ``documents`` is empty.
    """
    if config is None:
        config = get_config()

    if not documents:
        raise ValueError("Cannot build a vector store from an empty document list.")

    embeddings = _build_embeddings(config)
    vector_store = FAISS.from_documents(documents, embeddings)

    store_path = Path(config.vector_store_path)
    store_path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(store_path))

    return vector_store


def load_vector_store_and_search(
    query: str,
    config: AppConfig | None = None,
) -> list[Document]:
    """Load the persisted FAISS index and return the top matching chunks.

    The saved index is loaded from the configured path and searched by semantic
    similarity against the query. The number of chunks returned is controlled by
    ``top_k_results`` in configuration. Each returned chunk retains its source
    metadata so answers can be cited.

    Args:
        query: The natural language question to search for.
        config: Optional pipeline configuration. When omitted, configuration is
            loaded from the environment via :func:`~src.config.get_config`.

    Returns:
        list[Document]: The top K most similar chunks, each with source metadata.

    Raises:
        FileNotFoundError: If no saved index exists at the configured path.
    """
    if config is None:
        config = get_config()

    store_path = Path(config.vector_store_path)
    if not store_path.exists():
        raise FileNotFoundError(
            f"No vector store found at {store_path.resolve()}. "
            "Build it first by running ingestion and build_vector_store."
        )

    embeddings = _build_embeddings(config)
    vector_store = FAISS.load_local(
        str(store_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    return vector_store.similarity_search(query, k=config.top_k_results)
