"""Command line entry point for the RAG document question answering pipeline.

On the first run this ingests and chunks the source documents, builds a FAISS
vector store, and caches it to disk. Subsequent runs reuse the cached store and
skip straight to retrieval and generation. Pass ``--rebuild`` to force a fresh
index after changing the source documents.
"""

import argparse
from pathlib import Path

from src.config import AppConfig, get_config
from src.generation import generate_grounded_answer
from src.ingestion import load_and_chunk_documents
from src.retrieval import build_vector_store, load_vector_store_and_search


def _ensure_vector_store(config: AppConfig, rebuild: bool) -> None:
    """Build the vector store if it is missing or a rebuild was requested.

    Args:
        config: Pipeline configuration.
        rebuild: When True, always rebuild the index from the source documents.
    """
    store_path = Path(config.vector_store_path)
    if store_path.exists() and not rebuild:
        print(f"Using cached vector store at {store_path.resolve()}")
        return

    print("Ingesting and chunking source documents...")
    documents = load_and_chunk_documents(config=config)
    print(f"Produced {len(documents)} chunks. Building vector store...")
    build_vector_store(documents, config=config)
    print(f"Vector store saved to {store_path.resolve()}")


def _print_answer(question: str, answer: str, sources) -> None:
    """Print the answer and its cited sources in a clear, readable format.

    Args:
        question: The question that was asked.
        answer: The generated answer.
        sources: The document chunks used as grounding context.
    """
    separator = "=" * 70
    print(f"\n{separator}")
    print(f"Question: {question}")
    print(separator)
    print(f"\n{answer}\n")
    print(separator)
    print("Sources:")
    if not sources:
        print("  (none)")
    else:
        seen = []
        for chunk in sources:
            label = (
                f"{chunk.metadata.get('source', 'unknown')} "
                f"(chunk {chunk.metadata.get('chunk_index', '?')})"
            )
            if label not in seen:
                seen.append(label)
                print(f"  - {label}")
    print(separator)


def main() -> None:
    """Parse arguments and run the full retrieve-then-generate pipeline."""
    parser = argparse.ArgumentParser(
        description="Ask a question grounded in your local documents using RAG."
    )
    parser.add_argument(
        "--query",
        required=True,
        help="The question to answer using the ingested documents.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuilding the vector store from the source documents.",
    )
    args = parser.parse_args()

    config = get_config()

    _ensure_vector_store(config, rebuild=args.rebuild)

    print("Retrieving relevant chunks...")
    chunks = load_vector_store_and_search(args.query, config=config)

    print("Generating grounded answer...")
    answer, sources = generate_grounded_answer(args.query, chunks, config=config)

    _print_answer(args.query, answer, sources)


if __name__ == "__main__":
    main()
