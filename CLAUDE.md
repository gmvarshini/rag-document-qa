Instructions for Claude Code
You are building a Retrieval Augmented Generation document question answering system. Follow these steps precisely and in order.
Step 1: Initialize the project
Create the directory structure shown above. Initialize a Git repository. Create a `.gitignore` file that excludes `.env`, `__pycache__`, `.venv`, `*.pyc`, and `.DS_Store`.
Step 2: Set up UV and the virtual environment
Run `uv init` if not already initialized, then `uv venv` to create the virtual environment. Add dependencies using `uv add langchain langchain-community faiss-cpu pydantic pydantic-settings python-dotenv ollama-python` and `uv add --dev pytest ruff`.
Step 3: Build the configuration module
In `src/config.py`, define a Pydantic Settings class named `AppConfig` that inherits from `pydantic_settings.BaseSettings`. It should load the following fields from environment variables: `OLLAMA_MODEL` (default "llama3.2"), `CHUNK_SIZE` (default 500), `CHUNK_OVERLAP` (default 50), `VECTOR_STORE_PATH` (default "./vector_store"), and `TOP_K_RESULTS` (default 4). Include full type hints and a docstring explaining each field's purpose. Create a `.env.example` file documenting these variables and a `.env` file with actual values for local development.
Step 4: Build the ingestion module
In `src/ingestion.py`, write a function `load_and_chunk_documents` that reads all text files from the `data/source_documents` directory, splits them into overlapping chunks using LangChain's `RecursiveCharacterTextSplitter` with the chunk size and overlap from config, and returns a list of LangChain `Document` objects with metadata including source filename. Add a complete docstring with Args, Returns, and Raises sections following the Google docstring style.
Step 5: Build the retrieval module
In `src/retrieval.py`, write a function `build_vector_store` that takes the chunked documents, generates embeddings using a local sentence-transformers model wrapped through LangChain's `HuggingFaceEmbeddings`, builds a FAISS index, and saves it to disk at the configured path. Write a second function `load_vector_store_and_search` that loads the saved FAISS index and performs a similarity search for a given query string, returning the top K matching chunks with their source metadata. Both functions require full docstrings and type hints.
Step 6: Build the generation module
In `src/generation.py`, write a function `generate_grounded_answer` that takes a user question and the retrieved document chunks, constructs a prompt that instructs the LLM to answer using only the provided context and to explicitly say when the answer is not present in the context, sends this prompt to a local Ollama model, and returns the generated answer along with the list of source documents used. Include a docstring explaining the grounding strategy.
Step 7: Build the main entry point
In `src/main.py`, write a command line interface using Python's `argparse` that accepts a `--query` argument, runs the full pipeline from ingestion through generation on first run, caches the vector store for subsequent runs, and prints the answer with cited sources clearly formatted.
Step 8: Write tests
In `tests/test_retrieval.py`, write at least three pytest test cases covering chunking behavior, vector store creation, and search result relevance using a small fixture document set.
Step 9: Write the README
Write a comprehensive `README.md` explaining the project purpose, architecture diagram in text form, setup instructions using UV, how to add your own documents, how to run queries, and a section explaining the RAG pattern conceptually for anyone reading the repository. Do not use em dashes anywhere in the README or code comments. Keep the tone professional and direct.
Step 10: Finalize and publish
Run `ruff check` and fix any linting issues. Stage all files, write a clear commit message describing the project, and push to a new GitHub repository named `rag-document-qa`. Confirm the push succeeded and report the repository URL.
---
Key Concepts to Highlight for Learning
Explain in your own words, either in the README or in a short LEARNINGS.md file, why chunk overlap prevents context loss at chunk boundaries, why grounding matters for reducing hallucination, and how vector similarity search differs from keyword search. This demonstrates conceptual understanding beyond just running commands, which is what interviewers look for.