# Key Concepts

This file explains three ideas that are central to how the system works. They
are written out in plain language to show the reasoning behind the design, not
just the commands used to run it.

## Why chunk overlap prevents context loss at boundaries

Documents are too long to embed as a single vector, so they are split into
smaller chunks. The problem is that a naive split can cut a sentence or an idea
in half exactly at a chunk boundary. If a definition starts at the end of one
chunk and finishes at the start of the next, neither chunk contains the whole
thought, and a search might retrieve one half without the other.

Chunk overlap fixes this by making consecutive chunks share a slice of text. With
an overlap of fifty characters, the last fifty characters of one chunk are also
the first fifty characters of the next. Any sentence that straddles a boundary
now appears complete in at least one chunk. The cost is a small amount of
duplicated text and a few extra chunks, which is a cheap price for keeping ideas
intact. Overlap should be large enough to cover a typical sentence but small
relative to the chunk size, so most of each chunk is still new content.

## Why grounding matters for reducing hallucination

A language model generates text by predicting likely continuations. Left to its
own memory, it will happily produce fluent statements that sound correct but are
not, a failure mode called hallucination. This is dangerous precisely because the
wrong answers are so confident and well written.

Grounding means constraining the model to answer from supplied evidence. In this
system the prompt gives the model only the retrieved chunks and instructs it to
use nothing else, and to say clearly when the answer is not in the context. That
turns an open ended memory recall task into a focused reading comprehension task,
which models do far more reliably. Grounding also makes answers auditable: because
the exact source chunks are returned with every answer, a reader can check each
claim against the original text. The combination of constraining the input and
exposing the sources is what makes a RAG answer trustworthy.

## How vector similarity search differs from keyword search

Keyword search matches exact words. If you search for "car" it finds documents
that contain the string "car" and misses documents that only say "automobile" or
"vehicle", even though they mean the same thing. It is fast and precise about
spelling but blind to meaning.

Vector similarity search matches meaning instead of spelling. Every chunk is
converted into an embedding, a list of numbers positioned so that texts with
similar meaning sit close together in that numeric space. A query is embedded the
same way, and the system returns the chunks whose vectors are nearest to the
query vector, measured by cosine similarity or distance. Because "car" and
"automobile" produce nearby vectors, a search for one finds passages about the
other. This semantic matching is what lets a RAG system retrieve relevant context
even when the user's wording does not match the document's wording. The tradeoff
is that vector search depends on the quality of the embedding model and can
return something loosely related rather than an exact term match, so the two
approaches are sometimes combined.
