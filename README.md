# RAG Document Question Answering

A local Retrieval Augmented Generation (RAG) system that answers questions about
your own text documents. It retrieves the most relevant passages from your files
and asks a local Ollama language model to answer using only those passages, so
answers stay grounded in your source material instead of the model's memory.

Everything runs locally. Embeddings are computed with a sentence-transformers
model, similarity search uses a FAISS index on disk, and generation uses a model
served by Ollama on your machine. No data leaves your computer.

## Why this project exists

Large language models are fluent but they can state false information with
confidence. RAG reduces that risk by grounding the model in retrieved evidence
and by returning the exact sources used, so every answer can be checked against
the documents it came from.

## Architecture

The pipeline has three stages. Ingestion prepares the documents, retrieval finds
the relevant pieces, and generation writes the grounded answer.

```
                        data/source_documents/*.txt
                                   |
                                   v
        +-------------------------------------------------+
        | 1. INGESTION  (src/ingestion.py)                |
        |    load text files                              |
        |    split into overlapping chunks                |
        |    attach source filename metadata              |
        +-------------------------------------------------+
                                   |
                          list[Document] chunks
                                   v
        +-------------------------------------------------+
        | 2. RETRIEVAL  (src/retrieval.py)                |
        |    embed chunks with sentence-transformers      |
        |    build FAISS index and save to disk           |
        |    later: embed query and search top K chunks   |
        +-------------------------------------------------+
                                   |
                     top K relevant chunks + sources
                                   v
        +-------------------------------------------------+
        | 3. GENERATION  (src/generation.py)              |
        |    build a grounded prompt from the chunks      |
        |    call the local Ollama model                  |
        |    return the answer plus cited sources         |
        +-------------------------------------------------+
                                   |
                                   v
                     answer printed with citations
                          (src/main.py, CLI)
```

Configuration for every stage lives in `src/config.py` and is loaded from
environment variables or a local `.env` file.

## Requirements

- Python 3.10 to 3.12
- [uv](https://docs.astral.sh/uv/) for environment and dependency management
- [Ollama](https://ollama.com/) installed and running locally for generation

## Setup with UV

1. Install the dependencies into a managed virtual environment:

   ```bash
   uv sync
   ```

   This reads `pyproject.toml` and `uv.lock` and creates a `.venv` with the
   exact pinned versions.

2. Create your local environment file from the template:

   ```bash
   cp .env.example .env
   ```

   Adjust any values in `.env` if you want different chunk sizes, a different
   model, or a different number of retrieved results.

3. Pull the Ollama model named in your `.env` (the default is `llama3.2`):

   ```bash
   ollama pull llama3.2
   ```

   Make sure the Ollama service is running before you generate answers.

## Adding your own documents

Place any UTF-8 `.txt` files inside `data/source_documents/`. Two example files
are included so the system works immediately. When you add, remove, or edit
documents, rebuild the index on the next query with the `--rebuild` flag so the
vector store reflects your changes:

```bash
uv run python -m src.main --query "your question" --rebuild
```

## Running queries

Ask a question with the `--query` argument:

```bash
uv run python -m src.main --query "What are the three stages of a RAG pipeline?"
```

On the first run the system ingests the documents, builds the FAISS index, and
caches it under `vector_store/`. Later runs reuse that cache and skip straight to
retrieval and generation, which makes them much faster. Use `--rebuild` to force
a fresh index after changing your documents.

The output shows the question, the grounded answer, and the list of source files
and chunks that the answer was based on.

## Running the tests

```bash
uv run pytest
```

The tests use a small in memory fixture document set and cover chunking behavior,
vector store creation, and search relevance. The first test run downloads the
embedding model weights, so it may take a moment.

## Project layout

```
rag-document-qa/
  data/source_documents/   your .txt documents (two examples included)
  src/
    config.py              typed settings loaded from the environment
    ingestion.py           load files and split them into chunks
    retrieval.py           build the FAISS index and search it
    generation.py          build the grounded prompt and call Ollama
    main.py                command line entry point for the pipeline
  tests/
    test_retrieval.py      chunking, index, and search tests
  vector_store/            generated FAISS index (created at runtime, ignored)
```

## How RAG works, explained

Retrieval Augmented Generation joins a search system to a language model. Rather
than trusting the model to recall facts from training, the system looks up
relevant passages from your documents and hands them to the model as context,
with an instruction to answer only from that context.

The flow is straightforward. Your documents are split into small overlapping
chunks. Each chunk is turned into an embedding, which is a list of numbers that
captures its meaning, and all embeddings are stored in a FAISS index. When you
ask a question, the question is embedded the same way and the index returns the
chunks whose meaning is closest to it. Those chunks become the context for the
model, and the model produces an answer plus the sources it used.

This design gives three practical benefits. Answers are grounded in real text
rather than guessed, so hallucination is less likely. Answers are auditable
because the exact sources are returned. And updating knowledge is cheap, since
you only change documents and rebuild the index instead of retraining a model.

For a deeper explanation of the key ideas, see [LEARNINGS.md](LEARNINGS.md).
