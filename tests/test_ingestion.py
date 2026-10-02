"""Tests for loading text and PDF documents during ingestion.

These tests need no embedding model or language model, so they run fast. The
PDF is created inside the test with reportlab, so no binary file is stored in
the repository.
"""

from pathlib import Path

import pytest
from reportlab.pdfgen import canvas

from src.config import AppConfig
from src.ingestion import load_and_chunk_documents


def _write_pdf(path: Path, pages: list[str]) -> None:
    """Write a small PDF with one line of text per page."""
    pdf = canvas.Canvas(str(path))
    for text in pages:
        pdf.drawString(72, 720, text)
        pdf.showPage()
    pdf.save()


@pytest.fixture()
def config() -> AppConfig:
    """A config with small chunks so the tests stay simple."""
    return AppConfig(chunk_size=200, chunk_overlap=20)


def test_pdf_pages_are_loaded_with_page_numbers(tmp_path: Path, config) -> None:
    """Every PDF page becomes a chunk that remembers its page number."""
    _write_pdf(
        tmp_path / "handbook.pdf",
        ["Employees get 25 vacation days per year.", "The office opens at 8 am."],
    )

    chunks = load_and_chunk_documents(source_dir=tmp_path, config=config)

    assert [chunk.metadata["page"] for chunk in chunks] == [1, 2]
    assert all(chunk.metadata["source"] == "handbook.pdf" for chunk in chunks)
    assert "25 vacation days" in chunks[0].page_content
    assert "8 am" in chunks[1].page_content


def test_text_and_pdf_files_are_loaded_together(tmp_path: Path, config) -> None:
    """A folder can mix .txt and .pdf files; text chunks have no page number."""
    (tmp_path / "notes.txt").write_text("RAG retrieves before it generates.")
    _write_pdf(tmp_path / "guide.pdf", ["FAISS finds similar vectors quickly."])

    chunks = load_and_chunk_documents(source_dir=tmp_path, config=config)

    by_source = {chunk.metadata["source"]: chunk for chunk in chunks}
    assert set(by_source) == {"notes.txt", "guide.pdf"}
    assert "page" not in by_source["notes.txt"].metadata
    assert by_source["guide.pdf"].metadata["page"] == 1


def test_unsupported_files_are_ignored(tmp_path: Path, config) -> None:
    """Only .txt and .pdf files are read; other files are skipped."""
    (tmp_path / "notes.txt").write_text("Only this file should be read.")
    (tmp_path / "image.png").write_bytes(b"not a document")

    chunks = load_and_chunk_documents(source_dir=tmp_path, config=config)

    assert {chunk.metadata["source"] for chunk in chunks} == {"notes.txt"}


def test_empty_folder_raises_a_clear_error(tmp_path: Path, config) -> None:
    """A folder with no supported files gives a helpful error."""
    with pytest.raises(ValueError, match=".txt or .pdf"):
        load_and_chunk_documents(source_dir=tmp_path, config=config)
