"""Application configuration loaded from environment variables.

This module centralizes every tunable parameter for the RAG pipeline so that
the ingestion, retrieval, and generation stages read from a single, validated
source of truth instead of scattering magic numbers across the codebase.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    """Typed application settings for the RAG pipeline.

    Values are read from environment variables (or a local ``.env`` file) at
    instantiation time. Every field falls back to a sensible default so the
    system runs out of the box without any configuration.

    Attributes:
        ollama_model: Name of the local Ollama model used to generate the final
            grounded answer, for example ``"llama3.2"``. The model must already
            be pulled locally with ``ollama pull <model>``.
        chunk_size: Maximum number of characters in each document chunk produced
            during ingestion. Larger chunks preserve more context per chunk but
            reduce retrieval precision.
        chunk_overlap: Number of characters shared between consecutive chunks.
            Overlap keeps sentences that straddle a chunk boundary intact in at
            least one chunk, preventing context loss at the split points.
        vector_store_path: Filesystem path where the FAISS index and its
            metadata are persisted so the store can be reused across runs.
        top_k_results: Number of most similar chunks to retrieve for a query and
            feed to the language model as grounding context.
        embedding_model: Name of the local sentence-transformers model used to
            embed both document chunks and incoming queries.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ollama_model: str = "llama3.2"
    chunk_size: int = 500
    chunk_overlap: int = 50
    vector_store_path: str = "./vector_store"
    top_k_results: int = 4
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"


def get_config() -> AppConfig:
    """Build and return a fresh :class:`AppConfig` instance.

    Returns:
        AppConfig: Settings populated from the environment and ``.env`` file.
    """
    return AppConfig()
