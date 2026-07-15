"""Tests for chunking, vector store creation, and search relevance.

These tests use a small fixture document set written to a temporary directory so
they run fast and do not depend on the project's real source documents. They
exercise the local embedding model, so the first run downloads model weights.
"""

from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.config import AppConfig
from src.ingestion import load_and_chunk_documents
from src.retrieval import build_vector_store, load_vector_store_and_search

FIXTURE_DOCS = {
    "solar.txt": (
        "The sun is a star at the center of the solar system. "
        "Solar panels convert sunlight into electricity using photovoltaic cells. "
        "Photosynthesis in plants also depends on sunlight to produce energy. "
    )
    * 5,
    "oceans.txt": (
        "The Pacific Ocean is the largest and deepest ocean on Earth. "
        "Ocean currents distribute heat around the planet and shape climate. "
        "Marine biology studies the organisms that live in salt water. "
    )
    * 5,
}


@pytest.fixture()
def fixture_config(tmp_path: Path) -> tuple[AppConfig, Path]:
    """Create fixture documents and a config pointing at a temp vector store.

    Args:
        tmp_path: Pytest supplied temporary directory unique to the test.

    Returns:
        tuple[AppConfig, Path]: The configuration and the source directory path.
    """
    source_dir = tmp_path / "source_documents"
    source_dir.mkdir()
    for name, text in FIXTURE_DOCS.items():
        (source_dir / name).write_text(text, encoding="utf-8")

    config = AppConfig(
        chunk_size=120,
        chunk_overlap=20,
        vector_store_path=str(tmp_path / "vector_store"),
        top_k_results=3,
    )
    return config, source_dir


def test_chunking_produces_overlapping_documents(
    fixture_config: tuple[AppConfig, Path],
) -> None:
    """Chunking should split text into multiple Documents with source metadata."""
    config, source_dir = fixture_config

    chunks = load_and_chunk_documents(source_dir=source_dir, config=config)

    assert len(chunks) > len(FIXTURE_DOCS)
    assert all(isinstance(chunk, Document) for chunk in chunks)
    assert all(len(chunk.page_content) <= config.chunk_size for chunk in chunks)
    sources = {chunk.metadata["source"] for chunk in chunks}
    assert sources == set(FIXTURE_DOCS.keys())
    assert all("chunk_index" in chunk.metadata for chunk in chunks)


def test_build_vector_store_persists_index(
    fixture_config: tuple[AppConfig, Path],
) -> None:
    """Building the vector store should write a FAISS index to disk."""
    config, source_dir = fixture_config

    chunks = load_and_chunk_documents(source_dir=source_dir, config=config)
    build_vector_store(chunks, config=config)

    store_path = Path(config.vector_store_path)
    assert (store_path / "index.faiss").exists()
    assert (store_path / "index.pkl").exists()


def test_search_returns_relevant_chunk(
    fixture_config: tuple[AppConfig, Path],
) -> None:
    """A query should retrieve chunks from the topically relevant document."""
    config, source_dir = fixture_config

    chunks = load_and_chunk_documents(source_dir=source_dir, config=config)
    build_vector_store(chunks, config=config)

    results = load_vector_store_and_search(
        "How do solar panels generate electricity?", config=config
    )

    assert len(results) == config.top_k_results
    top_result = results[0]
    assert top_result.metadata["source"] == "solar.txt"
    assert "solar" in top_result.page_content.lower()
