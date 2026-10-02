# CLAUDE.md

Notes for working on this repo with Claude Code.

## Project

A local RAG system that answers questions from your own `.txt` and `.pdf` files.
Documents are split into chunks, embedded with sentence-transformers
(`all-MiniLM-L6-v2`), stored in a FAISS index, and the best chunks are given to
Llama 3.2 through Ollama. The model must answer only from those chunks, and the
answer lists its sources. Everything runs on the local machine.

## Layout

- `src/config.py`: settings (model, chunk size 500, overlap 50, top 4 results)
- `src/ingestion.py`: loads text files and PDF pages, splits them into chunks
- `src/retrieval.py`: builds, saves and searches the FAISS index
- `src/generation.py`: builds the grounded prompt and calls Ollama
- `src/main.py`: command line entry point
- `tests/test_ingestion.py`: text and PDF loading, no model needed
- `tests/test_retrieval.py`: chunking, index and search, downloads the embedding model
- `LEARNINGS.md`: chunk overlap, grounding, vector vs keyword search

## Commands

```bash
uv sync                                    # install
uv run pytest                              # tests
uv run ruff check                          # lint
uv run python -m src.main --query "What are the three stages of a RAG pipeline?"
uv run python -m src.main --query "..." --rebuild   # after changing documents
```

Ollama must be running with `llama3.2` pulled to generate answers.

## Rules

- Settings go in `src/config.py` and `.env.example`, never hard coded.
- Every chunk keeps `source` and `chunk_index` in its metadata, plus `page` for
  PDFs. Citations depend on this.
- Documents and questions must use the same embedding model.
- Do not weaken the grounding prompt in `src/generation.py`. If the answer is not
  in the context, the model must reply with the fixed "not present" sentence.
- New file types go through `_read_pages` in `src/ingestion.py`, with a test.
- Tests create their own documents in a temporary folder. Tests that need no
  model go in `test_ingestion.py`.
- Type hints and Google style docstrings on all functions.
- Plain English in docs and comments. No em dashes.

## CI

`.github/workflows/tests.yml` runs ruff and pytest on every push and pull request,
with the embedding model cached between runs.

## Ideas for later

Hybrid search with BM25, a re-ranking step, OCR for scanned PDFs, and a small
evaluation set of questions with known answers.
