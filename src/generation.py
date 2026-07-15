"""Grounded answer generation using a local Ollama language model."""

import ollama
from langchain_core.documents import Document

from src.config import AppConfig, get_config

PROMPT_TEMPLATE = """You are a precise assistant that answers questions using \
only the provided context.

Follow these rules strictly:
1. Use only the information in the context below to answer the question.
2. If the answer is not contained in the context, reply exactly: \
"The answer is not present in the provided context."
3. Do not use outside knowledge and do not invent details.

Context:
{context}

Question: {question}

Answer:"""


def _format_context(chunks: list[Document]) -> str:
    """Render retrieved chunks into a numbered, source-labeled context block.

    Args:
        chunks: The retrieved document chunks.

    Returns:
        str: A human readable context string with per-chunk source labels.
    """
    formatted_parts = []
    for index, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source", "unknown")
        formatted_parts.append(f"[{index}] (source: {source})\n{chunk.page_content}")
    return "\n\n".join(formatted_parts)


def generate_grounded_answer(
    question: str,
    chunks: list[Document],
    config: AppConfig | None = None,
) -> tuple[str, list[Document]]:
    """Generate an answer grounded strictly in the retrieved context.

    The grounding strategy has two parts. First, the retrieved chunks are the
    only knowledge the model is allowed to use: the prompt explicitly forbids
    outside knowledge and instructs the model to say when the answer is absent.
    Second, the exact chunks used are returned alongside the answer so the caller
    can display citations and a reader can audit every claim. This constrains the
    model to the supplied evidence and makes hallucination easy to detect.

    Args:
        question: The user's natural language question.
        chunks: The retrieved document chunks used as grounding context.
        config: Optional pipeline configuration. When omitted, configuration is
            loaded from the environment via :func:`~src.config.get_config`.

    Returns:
        tuple[str, list[Document]]: The generated answer and the list of source
        chunks that were provided as context.
    """
    if config is None:
        config = get_config()

    if not chunks:
        return (
            "The answer is not present in the provided context.",
            [],
        )

    context = _format_context(chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    response = ollama.chat(
        model=config.ollama_model,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response["message"]["content"].strip()

    return answer, chunks
